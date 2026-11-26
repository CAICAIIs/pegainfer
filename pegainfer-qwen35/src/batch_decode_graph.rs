//! Qwen3.5 batched decode 的 CUDA Graph 状态（按 cohort 持有）。

use std::sync::Arc;

use anyhow::Result;
use pegainfer_core::cuda_graph::CudaGraphState;
use pegainfer_core::kv_pool::KvPool;
use pegainfer_core::tensor::DeviceContext;

use super::config::Config35;
use super::config::TensorParallelConfig;
use super::decode_buffers::BatchDecodeBuffers35;
use super::recurrent_state::LinearStatePointerTables;
use super::recurrent_state::RecurrentState;

/// CUDA Graph capture 桶宽，实际 batch 向上对齐到最近桶。
pub(crate) const BATCH_BUCKETS: &[usize] = &[1, 2, 4, 8, 16, 32, 64];

/// 单张 decode graph 的最大宽度（单 cohort 上限）。
pub(crate) const MAX_BATCH: usize = 64;

/// 返回 >= `bs` 的最小 bucket；超过 `MAX_BATCH` 直接 panic。
pub(crate) fn bucket_for(bs: usize) -> usize {
    for &b in BATCH_BUCKETS {
        if b >= bs {
            return b;
        }
    }
    panic!(
        "batch size {bs} exceeds largest bucket {}",
        BATCH_BUCKETS.last().unwrap()
    );
}

/// 单个 cohort 的 CUDA Graph decode 状态。
///
/// 启动时一次性预分配 `slot_capacity` 份 `RecurrentState` 与固定地址的
/// `linear_pointer_tables`；请求从准入到退休持有同一 `slot_idx`，prefill 直接写入
/// `slot_states[slot_idx]`。decode 前按当前 cohort active 列表原地更新 pointer table，
/// padding 行指向共享 `padding_state`，既避免退休时大块 D2D compaction，也避免
/// padding 行覆盖同一 cohort 内正在 chunked prefill 的槽位。
pub(crate) struct BatchDecodeGraphState {
    pub(crate) buffers: BatchDecodeBuffers35,
    pub(crate) slot_states: Vec<RecurrentState>,
    slot_state_ptrs: Vec<Vec<u64>>,
    slot_conv_ptrs: Vec<Vec<u64>>,
    _padding_state: Arc<RecurrentState>,
    padding_state_ptrs: Vec<u64>,
    padding_conv_ptrs: Vec<u64>,
    pub(crate) linear_pointer_tables: LinearStatePointerTables,
    /// 与 `BATCH_BUCKETS` 一一对应的 `CudaGraphState`。
    pub(crate) graphs: Vec<CudaGraphState>,
}

impl BatchDecodeGraphState {
    /// 分配一个 cohort 的 graph state：`slot_capacity` 份 recurrent state、
    /// 对齐到 bucket 的 decode buffer 与 pointer table，以及共享 `padding_state`。
    pub(crate) fn with_capacity_and_padding(
        ctx: &DeviceContext,
        config: &Config35,
        tensor_parallel: TensorParallelConfig,
        kv_pool: &KvPool,
        slot_capacity: usize,
        padding_state: Arc<RecurrentState>,
    ) -> Result<Self> {
        anyhow::ensure!(
            (1..=MAX_BATCH).contains(&slot_capacity),
            "Qwen3.5 cohort slot capacity must be in 1..={MAX_BATCH}, got {slot_capacity}"
        );
        let graph_bucket = bucket_for(slot_capacity);
        let padding_page_id = kv_pool.padding_page_id();
        let max_total_pages = kv_pool.capacity_pages();

        let buffers = BatchDecodeBuffers35::new(
            ctx,
            config,
            tensor_parallel,
            graph_bucket,
            max_total_pages,
            padding_page_id,
        )?;

        let mut slot_states = Vec::with_capacity(slot_capacity);
        let mut slot_state_ptrs = Vec::with_capacity(slot_capacity);
        let mut slot_conv_ptrs = Vec::with_capacity(slot_capacity);
        for _ in 0..slot_capacity {
            let state = RecurrentState::new(ctx, config)?;
            let (sp, cp) = state.layer_device_ptrs(ctx);
            slot_state_ptrs.push(sp);
            slot_conv_ptrs.push(cp);
            slot_states.push(state);
        }

        let (padding_state_ptrs, padding_conv_ptrs) = padding_state.layer_device_ptrs(ctx);
        let linear_pointer_tables =
            LinearStatePointerTables::new_for_capacity(ctx, config, graph_bucket, "Qwen3.5 graph")?;

        let graphs = BATCH_BUCKETS
            .iter()
            .map(|_| CudaGraphState::new())
            .collect();

        Ok(Self {
            buffers,
            slot_states,
            slot_state_ptrs,
            slot_conv_ptrs,
            _padding_state: padding_state,
            padding_state_ptrs,
            padding_conv_ptrs,
            linear_pointer_tables,
            graphs,
        })
    }

    /// 同步本 cohort 当前 active 槽位映射到 GPU pointer table（稳态无变更时零拷贝）。
    pub(crate) fn sync_active_slots(
        &mut self,
        ctx: &DeviceContext,
        active_slots: &[usize],
        padded_bs: usize,
    ) -> Result<()> {
        for &slot_idx in active_slots {
            anyhow::ensure!(
                slot_idx < self.slot_states.len(),
                "Qwen3.5 active slot {slot_idx} out of range 0..{}",
                self.slot_states.len()
            );
        }
        self.linear_pointer_tables.sync_slots(
            ctx,
            &self.slot_state_ptrs,
            &self.slot_conv_ptrs,
            &self.padding_state_ptrs,
            &self.padding_conv_ptrs,
            active_slots,
            padded_bs,
            "Qwen3.5 graph",
        )
    }

    /// 清零指定槽位以供新准入请求原地 prefill。
    pub(crate) fn reset_slot(&mut self, ctx: &DeviceContext, slot_idx: usize) -> Result<()> {
        anyhow::ensure!(
            slot_idx < self.slot_states.len(),
            "Qwen3.5 reset slot {slot_idx} out of range 0..{}",
            self.slot_states.len()
        );
        self.slot_states[slot_idx].reset(ctx)
    }
}
