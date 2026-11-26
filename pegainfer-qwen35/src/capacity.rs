//! Qwen3.5 服务容量模型：统一核算 decode_width、max_resident、recurrent state 池、
//! 双 cohort graph buffer、KV 页数与 workspace；启动前算清账目，放不下直接报错。

use anyhow::Result;
use log::info;

use super::batch_decode_graph;
use super::config::Config35;
use super::config::TensorParallelConfig;
use super::decode_buffers::BatchDecodeBuffers35;
use super::prefill::SCRATCH_ESTIMATE_SEQ;
use super::prefill_buffers::GdrChunkwiseScratch35;
use super::recurrent_state;

/// 单步 decode 默认宽度（与最大 CUDA Graph bucket 一致）。
pub(crate) const DEFAULT_DECODE_WIDTH: usize = batch_decode_graph::MAX_BATCH;

/// 单 GPU 最大 cohort 数（A/B 两张 64 行 decode graph）。
pub(crate) const MAX_SINGLE_GPU_COHORTS: usize = 2;

/// 单 GPU 默认最大驻留请求数（2 × 64 = 128）。
pub(crate) const DEFAULT_SINGLE_GPU_MAX_RESIDENT: usize =
    DEFAULT_DECODE_WIDTH * MAX_SINGLE_GPU_COHORTS;

/// TP 默认最大驻留请求数（保持单 cohort 64）。
pub(crate) const DEFAULT_TP_MAX_RESIDENT: usize = DEFAULT_DECODE_WIDTH;

/// Full-attention KV 分页大小（token/页）。
pub(crate) const KV_PAGE_SIZE: usize = 16;

/// cuBLASLt / FlashInfer / CUDA Graph 运行期 workspace 预留（512 MiB）。
pub(crate) const RUNTIME_WORKSPACE_BYTES: usize = 512 * 1024 * 1024;

/// 剩余可用显存中分配给 KV pool 的比例上限，为 CUDA 驱动碎片留出余量。
const KV_HEADROOM_FRACTION: f64 = 0.90;

/// Qwen3.5 实例的完整显存与并发容量账目。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ServingCapacity {
    /// 单步 decode 宽度上限（1..=64）。
    pub decode_width: usize,
    /// 实例可同时驻留的最大请求数（单 GPU 默认 128，TP 默认 64）。
    pub max_resident: usize,
    /// 启动预分配的请求 recurrent state 槽位总数（等于 `max_resident`）。
    pub state_slots: usize,
    /// 单 GPU CUDA Graph padding 共享槽位数（单 GPU 为 1，TP 为 0）。
    pub padding_state_slots: usize,
    /// 启用的 cohort 数（单 GPU `max_resident > decode_width` 时为 2，否则 1；TP 恒为 1）。
    pub cohort_count: usize,
    /// 每个 cohort 的请求驻留上限（未使用位置为 0）。
    pub cohort_widths: [usize; MAX_SINGLE_GPU_COHORTS],
    /// 每个 cohort 对应的 CUDA Graph bucket 容量（未使用位置为 0）。
    pub cohort_graph_buckets: [usize; MAX_SINGLE_GPU_COHORTS],
    /// 模型 config 声明的最大序列长度（例如 262144）。
    pub max_model_len: usize,
    /// KV 页大小（token/页）。
    pub kv_page_size: usize,
    /// KV 池总页数（含 1 个 padding 页）。
    pub kv_pages: usize,
    /// 请求可用的 KV 页数（`kv_pages - 1`）。
    pub kv_request_pages: usize,
    /// KV 池显存字节数。
    pub kv_bytes: usize,
    /// recurrent state 池显存字节数（含 padding state）。
    pub recurrent_state_bytes: usize,
    /// 所有 cohort 的 graph decode buffer 与 pointer table 显存字节数。
    pub graph_buffers_bytes: usize,
    /// prefill scratch 预留字节数。
    pub prefill_scratch_bytes: usize,
    /// 运行期 workspace 预留字节数。
    pub runtime_workspace_bytes: usize,
    /// 核算时观察到的设备空闲显存字节数。
    pub free_bytes_at_plan: usize,
}

impl ServingCapacity {
    /// 按 topology 返回默认 `max_resident`。
    #[must_use]
    pub fn default_max_resident(decode_width: usize, tp_world_size: usize) -> usize {
        if tp_world_size > 1 {
            decode_width
        } else {
            decode_width.saturating_mul(MAX_SINGLE_GPU_COHORTS)
        }
    }

    /// 校验用户请求的 `(decode_width, max_resident)` 与 TP 拓扑是否合法；不静默降级。
    pub fn validate_request(
        decode_width: usize,
        max_resident: usize,
        tp_world_size: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            tp_world_size >= 1,
            "Qwen3.5 tp_size must be >= 1, got {tp_world_size}"
        );
        anyhow::ensure!(
            (1..=DEFAULT_DECODE_WIDTH).contains(&decode_width),
            "Qwen3.5 decode_width must be in 1..={DEFAULT_DECODE_WIDTH}, got {decode_width}"
        );
        anyhow::ensure!(
            max_resident >= 1,
            "Qwen3.5 max_resident must be >= 1, got {max_resident}"
        );
        if tp_world_size > 1 {
            anyhow::ensure!(
                max_resident <= decode_width,
                "Qwen3.5 TP supports a single cohort only: max_resident ({max_resident}) must be <= decode_width ({decode_width})"
            );
        } else {
            let max_allowed = decode_width * MAX_SINGLE_GPU_COHORTS;
            anyhow::ensure!(
                max_resident <= max_allowed,
                "Qwen3.5 single-GPU max_resident ({max_resident}) exceeds {MAX_SINGLE_GPU_COHORTS} cohorts x decode_width ({decode_width}) = {max_allowed}"
            );
        }
        Ok(())
    }

    /// 在权重加载完成后，按真实空闲显存核算全部预算；放不下直接返回错误。
    pub(crate) fn plan(
        config: &Config35,
        tp: TensorParallelConfig,
        decode_width: usize,
        max_resident: usize,
        free_bytes: usize,
    ) -> Result<Self> {
        Self::validate_request(decode_width, max_resident, tp.world_size)?;

        let (cohort_count, cohort_widths) = if tp.is_sharded() || max_resident <= decode_width {
            (1, [max_resident, 0])
        } else {
            (2, [decode_width, max_resident - decode_width])
        };

        let mut cohort_graph_buckets = [0usize; MAX_SINGLE_GPU_COHORTS];
        for i in 0..cohort_count {
            cohort_graph_buckets[i] = batch_decode_graph::bucket_for(cohort_widths[i]);
        }

        let state_slots = max_resident;
        let padding_state_slots = usize::from(!tp.is_sharded());
        let total_state_slots = state_slots + padding_state_slots;
        let per_request_state_bytes = recurrent_state::bytes_per_request(config);
        let recurrent_state_bytes = total_state_slots
            .checked_mul(per_request_state_bytes)
            .ok_or_else(|| {
                anyhow::anyhow!("Qwen3.5 recurrent-state byte accounting overflowed usize")
            })?;

        let num_full_layers = config.num_full_attention_layers();
        let kv_layout = pegainfer_core::kv_pool::KvLayout::new(
            num_full_layers,
            config.local_num_key_value_heads(tp),
            config.head_dim,
            KV_PAGE_SIZE,
        )?;
        let bytes_per_page = kv_layout
            .page_stride
            .checked_mul(std::mem::size_of::<half::bf16>())
            .ok_or_else(|| anyhow::anyhow!("Qwen3.5 KV page byte size overflowed usize"))?;

        // 每张 graph buffer 含固定张量与 `page_indices_d`（随 kv_pages 线性增长）。
        let graph_buffers_fixed_bytes: usize = (0..cohort_count)
            .map(|i| {
                let bucket = cohort_graph_buckets[i];
                BatchDecodeBuffers35::estimate_fixed_bytes(config, tp, bucket)
                    + recurrent_state::LinearStatePointerTables::estimate_bytes(config, bucket)
            })
            .sum();
        let page_meta_bytes_per_page = cohort_count * std::mem::size_of::<i32>();
        let effective_bytes_per_page = bytes_per_page + page_meta_bytes_per_page;

        let prefill_scratch_bytes =
            GdrChunkwiseScratch35::estimate_bytes(config, SCRATCH_ESTIMATE_SEQ);
        let runtime_workspace_bytes = RUNTIME_WORKSPACE_BYTES;
        let fixed_bytes = recurrent_state_bytes
            .saturating_add(graph_buffers_fixed_bytes)
            .saturating_add(prefill_scratch_bytes)
            .saturating_add(runtime_workspace_bytes);

        let max_model_len = config.max_position_embeddings;
        let min_request_pages = max_model_len.div_ceil(KV_PAGE_SIZE);
        // 额外 1 页供 CUDA Graph padding 使用。
        let min_kv_pages = min_request_pages + 1;
        let min_kv_total_bytes = min_kv_pages
            .checked_mul(effective_bytes_per_page)
            .ok_or_else(|| {
                anyhow::anyhow!("Qwen3.5 minimum KV byte accounting overflowed usize")
            })?;

        let required_bytes = fixed_bytes.saturating_add(min_kv_total_bytes);
        let mode = if tp.is_sharded() {
            format!("tp{}-rank{}", tp.world_size, tp.rank)
        } else {
            "single-gpu".to_string()
        };
        anyhow::ensure!(
            free_bytes >= required_bytes,
            "insufficient device memory for Qwen3.5 ({mode}): {} MiB free, need at least {} MiB \
             [recurrent_state={} MiB ({state_slots} slots + {padding_state_slots} padding), \
             graph_buffers={} MiB ({cohort_count} cohort(s) buckets={:?}), \
             prefill_scratch={} MiB, workspace={} MiB, \
             min_kv={} MiB ({min_kv_pages} pages for max_model_len={max_model_len})]; \
             refusing to silently downgrade max_resident={max_resident} or max_model_len={max_model_len}",
            to_mib(free_bytes),
            to_mib(required_bytes),
            to_mib(recurrent_state_bytes),
            to_mib(graph_buffers_fixed_bytes + min_kv_pages * page_meta_bytes_per_page),
            &cohort_graph_buckets[..cohort_count],
            to_mib(prefill_scratch_bytes),
            to_mib(runtime_workspace_bytes),
            to_mib(min_kv_pages * bytes_per_page),
        );

        let available_for_kv = free_bytes - fixed_bytes;
        let target_kv_bytes = ((available_for_kv as f64) * KV_HEADROOM_FRACTION) as usize;
        let budgeted_kv_bytes = target_kv_bytes
            .max(min_kv_total_bytes)
            .min(available_for_kv);
        let kv_pages = (budgeted_kv_bytes / effective_bytes_per_page).max(min_kv_pages);
        let kv_request_pages = kv_pages - 1;
        let kv_bytes = kv_pages * bytes_per_page;
        let graph_buffers_bytes = graph_buffers_fixed_bytes + kv_pages * page_meta_bytes_per_page;

        Ok(Self {
            decode_width,
            max_resident,
            state_slots,
            padding_state_slots,
            cohort_count,
            cohort_widths,
            cohort_graph_buckets,
            max_model_len,
            kv_page_size: KV_PAGE_SIZE,
            kv_pages,
            kv_request_pages,
            kv_bytes,
            recurrent_state_bytes,
            graph_buffers_bytes,
            prefill_scratch_bytes,
            runtime_workspace_bytes,
            free_bytes_at_plan: free_bytes,
        })
    }

    /// 全部 cohort 中最大的 graph bucket 宽度（用于 cublasLt decode 调优上限）。
    pub(crate) fn max_graph_bucket(&self) -> usize {
        self.cohort_graph_buckets[..self.cohort_count]
            .iter()
            .copied()
            .max()
            .unwrap_or(1)
    }

    /// 打印启动容量账目，便于与实测显存逐项核对。
    pub(crate) fn log_ledger(&self, tp: TensorParallelConfig) {
        let mode = if tp.is_sharded() {
            format!("tp{}-rank{}", tp.world_size, tp.rank)
        } else {
            "single-gpu".to_string()
        };
        let total_planned = self.kv_bytes
            + self.recurrent_state_bytes
            + self.graph_buffers_bytes
            + self.prefill_scratch_bytes
            + self.runtime_workspace_bytes;
        info!(
            "Qwen3.5 capacity ({mode}): decode_width={}, max_resident={}, cohorts={} ({:?}, buckets={:?}), \
             state_slots={} (+{} padding, {} MiB), kv={} pages ({} req pages, {} tokens, {} MiB), \
             graph_buffers={} MiB, prefill_scratch={} MiB, workspace={} MiB, \
             max_model_len={}, planned={} MiB / free={} MiB",
            self.decode_width,
            self.max_resident,
            self.cohort_count,
            &self.cohort_widths[..self.cohort_count],
            &self.cohort_graph_buckets[..self.cohort_count],
            self.state_slots,
            self.padding_state_slots,
            to_mib(self.recurrent_state_bytes),
            self.kv_pages,
            self.kv_request_pages,
            self.kv_request_pages * self.kv_page_size,
            to_mib(self.kv_bytes),
            to_mib(self.graph_buffers_bytes),
            to_mib(self.prefill_scratch_bytes),
            to_mib(self.runtime_workspace_bytes),
            self.max_model_len,
            to_mib(total_planned),
            to_mib(self.free_bytes_at_plan),
        );
    }
}

fn to_mib(bytes: usize) -> usize {
    bytes / (1024 * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LayerType;

    fn qwen35_4b_config() -> Config35 {
        let mut layer_types = Vec::with_capacity(32);
        for i in 0..32 {
            layer_types.push(if i % 4 == 3 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            });
        }
        Config35 {
            hidden_size: 2560,
            intermediate_size: 9216,
            num_hidden_layers: 32,
            vocab_size: 248_320,
            selection_vocab: 248_320,
            rms_norm_eps: 1e-6,
            eos_token_id: 248_044,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 256,
            linear_num_key_heads: 16,
            linear_key_head_dim: 128,
            linear_num_value_heads: 32,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            rope_theta: 10_000_000.0,
            rotary_dim: 64,
            max_position_embeddings: 262_144,
            tie_word_embeddings: true,
            layer_types,
        }
    }

    fn qwen35_27b_config() -> Config35 {
        let mut layer_types = Vec::with_capacity(64);
        for i in 0..64 {
            layer_types.push(if i % 4 == 3 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            });
        }
        Config35 {
            hidden_size: 5120,
            intermediate_size: 17408,
            num_hidden_layers: 64,
            vocab_size: 248_320,
            selection_vocab: 248_320,
            rms_norm_eps: 1e-6,
            eos_token_id: 248_044,
            num_attention_heads: 24,
            num_key_value_heads: 4,
            head_dim: 256,
            linear_num_key_heads: 16,
            linear_key_head_dim: 128,
            linear_num_value_heads: 48,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            rope_theta: 10_000_000.0,
            rotary_dim: 64,
            max_position_embeddings: 262_144,
            tie_word_embeddings: false,
            layer_types,
        }
    }

    #[test]
    fn single_gpu_default_plans_two_64_cohorts_and_262k_kv() {
        let cfg = qwen35_4b_config();
        let free_bytes = 85 * 1024 * 1024 * 1024; // 85 GiB free on H20 after 4B weights
        let cap = ServingCapacity::plan(
            &cfg,
            TensorParallelConfig::default(),
            DEFAULT_DECODE_WIDTH,
            DEFAULT_SINGLE_GPU_MAX_RESIDENT,
            free_bytes,
        )
        .expect("4B on H20 must fit dual 64-row cohorts");

        assert_eq!(cap.decode_width, 64);
        assert_eq!(cap.max_resident, 128);
        assert_eq!(cap.state_slots, 128);
        assert_eq!(cap.padding_state_slots, 1);
        assert_eq!(cap.cohort_count, 2);
        assert_eq!(cap.cohort_widths, [64, 64]);
        assert_eq!(cap.cohort_graph_buckets, [64, 64]);
        assert_eq!(cap.max_model_len, 262_144);
        assert!(cap.kv_request_pages * cap.kv_page_size >= 262_144);
        assert!(
            cap.kv_bytes
                + cap.recurrent_state_bytes
                + cap.graph_buffers_bytes
                + cap.prefill_scratch_bytes
                + cap.runtime_workspace_bytes
                <= free_bytes
        );
    }

    #[test]
    fn single_gpu_27b_tp1_fits_h20_with_128_resident() {
        let cfg = qwen35_27b_config();
        let free_bytes = 42 * 1024 * 1024 * 1024; // ~42 GiB free on 96 GiB H20 after 27B weights
        let cap = ServingCapacity::plan(
            &cfg,
            TensorParallelConfig::default(),
            DEFAULT_DECODE_WIDTH,
            DEFAULT_SINGLE_GPU_MAX_RESIDENT,
            free_bytes,
        )
        .expect("27B TP1 on 96 GiB H20 must fit dual cohorts + one 262144 sequence");

        assert_eq!(cap.decode_width, 64);
        assert_eq!(cap.max_resident, 128);
        assert_eq!(cap.cohort_count, 2);
        assert!(cap.kv_request_pages * cap.kv_page_size >= 262_144);
    }

    #[test]
    fn tp2_plans_single_cohort_64_and_rejects_128() {
        let cfg = qwen35_27b_config();
        let tp2 = TensorParallelConfig {
            rank: 0,
            world_size: 2,
        };
        let free_bytes = 68 * 1024 * 1024 * 1024;
        let cap = ServingCapacity::plan(&cfg, tp2, 64, 64, free_bytes)
            .expect("TP2 must fit single 64-slot cohort");
        assert_eq!(cap.cohort_count, 1);
        assert_eq!(cap.max_resident, 64);
        assert_eq!(cap.state_slots, 64);
        assert_eq!(cap.padding_state_slots, 0);

        let err = ServingCapacity::plan(&cfg, tp2, 64, 128, free_bytes)
            .expect_err("TP2 must fail-fast on max_resident > decode_width")
            .to_string();
        assert!(err.contains("single cohort"), "unexpected error: {err}");
    }

    #[test]
    fn fails_fast_when_memory_cannot_hold_one_max_model_len_sequence() {
        let cfg = qwen35_4b_config();
        // 10 GiB free cannot hold 6.2 GiB recurrent + 3.5 GiB prefill scratch + 8 GiB min KV.
        let free_bytes = 10 * 1024 * 1024 * 1024;
        let err = ServingCapacity::plan(&cfg, TensorParallelConfig::default(), 64, 128, free_bytes)
            .expect_err("must fail-fast instead of silently downgrading")
            .to_string();
        assert!(err.contains("insufficient device memory"), "got: {err}");
        assert!(err.contains("max_model_len=262144"), "got: {err}");
        assert!(err.contains("refusing to silently downgrade"), "got: {err}");
    }

    #[test]
    fn single_gpu_sub_width_uses_one_cohort() {
        let cfg = qwen35_4b_config();
        let free_bytes = 85 * 1024 * 1024 * 1024;
        let cap = ServingCapacity::plan(&cfg, TensorParallelConfig::default(), 8, 8, free_bytes)
            .expect("8/8 should use one cohort");
        assert_eq!(cap.cohort_count, 1);
        assert_eq!(cap.cohort_widths, [8, 0]);
        assert_eq!(cap.cohort_graph_buckets, [8, 0]);
    }

    #[test]
    fn rejects_invalid_decode_width_and_max_resident() {
        assert!(ServingCapacity::validate_request(0, 64, 1).is_err());
        assert!(ServingCapacity::validate_request(65, 65, 1).is_err());
        assert!(ServingCapacity::validate_request(64, 0, 1).is_err());
        assert!(ServingCapacity::validate_request(64, 129, 1).is_err());
        assert!(ServingCapacity::validate_request(32, 65, 1).is_err());
        assert!(ServingCapacity::validate_request(64, 65, 2).is_err());
    }
}
