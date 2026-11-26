//! Recurrent state for Qwen3.5 linear attention layers.
//!
//! Each linear attention layer maintains:
//! - Recurrent state: [num_value_heads, key_head_dim, value_head_dim] f32, V contiguous ([H,K,V])
//! - Conv state: [qkv_dim × (conv_kernel_dim - 1)] bf16

use anyhow::Result;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtrMut;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceVec;

use super::config::Config35;

/// Per-layer recurrent state for a single linear attention layer.
pub(crate) struct LayerRecurrentState {
    /// Recurrent state matrix: [num_value_heads * key_head_dim * value_head_dim] f32
    /// Stored as f32 per mamba_ssm_dtype="float32" in config.
    pub(crate) state: CudaSlice<f32>,
    /// Conv1d state buffer: [qkv_dim * (conv_kernel_dim - 1)] bf16
    /// Stores the last (kernel_dim - 1) inputs for causal conv1d.
    pub(crate) conv_state: DeviceVec,
}

/// Recurrent state for all linear attention layers.
pub(crate) struct RecurrentState {
    pub(crate) layers: Vec<LayerRecurrentState>,
    /// Number of tokens processed so far (for prefill/decode tracking).
    pub(crate) seq_len: usize,
}

/// Device-side tables of per-slot recurrent-state pointers.
///
/// Batched linear decode kernels take a device array of pointers per linear
/// layer. The underlying `CudaSlice` allocations inside `RecurrentState` stay
/// at fixed device addresses for the request lifetime, so these tables should
/// be built once per slot/request and then reused across decode tokens.
pub(crate) struct LinearStatePointerTables {
    pub(crate) state_ptrs: Vec<CudaSlice<u64>>,
    pub(crate) conv_state_ptrs: Vec<CudaSlice<u64>>,
    batch_size: usize,
    last_active_slots: Vec<usize>,
    last_padded_bs: usize,
}

/// Per-layer element counts shared by allocation and reservation:
/// (linear layers, f32 state elements, bf16 conv elements).
fn per_layer_dims(config: &Config35) -> (usize, usize, usize) {
    let num_linear_layers = config.num_hidden_layers - config.num_full_attention_layers();
    let state_size =
        config.linear_num_value_heads * config.linear_key_head_dim * config.linear_value_head_dim;
    let conv_state_size = config.linear_attn_qkv_dim() * (config.linear_conv_kernel_dim - 1);
    (num_linear_layers, state_size, conv_state_size)
}

impl RecurrentState {
    /// Allocate zeroed recurrent state for all linear attention layers.
    pub(crate) fn new(ctx: &DeviceContext, config: &Config35) -> Result<Self> {
        let (num_linear_layers, state_size, conv_state_size) = per_layer_dims(config);

        let mut layers = Vec::with_capacity(num_linear_layers);
        for _ in 0..num_linear_layers {
            let state: CudaSlice<f32> = ctx
                .stream
                .alloc_zeros(state_size)
                .map_err(|e| anyhow::anyhow!("Alloc recurrent state failed: {}", e))?;
            layers.push(LayerRecurrentState {
                state,
                conv_state: DeviceVec::zeros(ctx, conv_state_size)?,
            });
        }

        Ok(Self { layers, seq_len: 0 })
    }

    /// 复用已分配槽位时清零 GPU recurrent/conv state，避免按请求反复 malloc/free。
    pub(crate) fn reset(&mut self, ctx: &DeviceContext) -> Result<()> {
        for layer in &mut self.layers {
            ctx.stream
                .memset_zeros(&mut layer.state)
                .map_err(|e| anyhow::anyhow!("Reset recurrent state failed: {e}"))?;
            ctx.stream
                .memset_zeros(&mut layer.conv_state.data)
                .map_err(|e| anyhow::anyhow!("Reset conv state failed: {e}"))?;
        }
        self.seq_len = 0;
        Ok(())
    }

    /// 提取各 linear layer 的稳定设备指针（仅在启动预分配后读取一次）。
    pub(crate) fn layer_device_ptrs(&self, ctx: &DeviceContext) -> (Vec<u64>, Vec<u64>) {
        use cudarc::driver::DevicePtr;
        let mut state_ptrs = Vec::with_capacity(self.layers.len());
        let mut conv_ptrs = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (sp, _gs) = layer.state.device_ptr(&ctx.stream);
            let (cp, _gc) = layer.conv_state.data.device_ptr(&ctx.stream);
            state_ptrs.push(sp);
            conv_ptrs.push(cp);
        }
        (state_ptrs, conv_ptrs)
    }
}

impl LinearStatePointerTables {
    /// 估算 pointer table 占用的显存字节数。
    pub(crate) fn estimate_bytes(config: &Config35, batch_capacity: usize) -> usize {
        let num_linear_layers = config.num_hidden_layers - config.num_full_attention_layers();
        num_linear_layers * 2 * batch_capacity * std::mem::size_of::<u64>()
    }

    /// 为 CUDA Graph 预分配固定地址的 pointer table，后续仅原地覆写内容。
    pub(crate) fn new_for_capacity(
        ctx: &DeviceContext,
        config: &Config35,
        batch_capacity: usize,
        label: &str,
    ) -> Result<Self> {
        let num_linear_layers = config.num_hidden_layers - config.num_full_attention_layers();
        let mut state_ptrs = Vec::with_capacity(num_linear_layers);
        let mut conv_state_ptrs = Vec::with_capacity(num_linear_layers);
        for layer_idx in 0..num_linear_layers {
            state_ptrs.push(ctx.stream.alloc_zeros(batch_capacity).map_err(|e| {
                anyhow::anyhow!("alloc {label} linear state pointer table {layer_idx}: {e}")
            })?);
            conv_state_ptrs.push(ctx.stream.alloc_zeros(batch_capacity).map_err(|e| {
                anyhow::anyhow!("alloc {label} conv state pointer table {layer_idx}: {e}")
            })?);
        }
        Ok(Self {
            state_ptrs,
            conv_state_ptrs,
            batch_size: batch_capacity,
            last_active_slots: Vec::new(),
            last_padded_bs: 0,
        })
    }

    /// 将 `0..active_slots.len()` 映射到对应 cohort 槽位，其余 `..padded_bs` 指向共享 padding state。
    /// 当 `active_slots` 与 `padded_bs` 未变化时直接跳过 H2D 拷贝。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sync_slots(
        &mut self,
        ctx: &DeviceContext,
        slot_state_ptrs: &[Vec<u64>],
        slot_conv_ptrs: &[Vec<u64>],
        padding_state_ptrs: &[u64],
        padding_conv_ptrs: &[u64],
        active_slots: &[usize],
        padded_bs: usize,
        label: &str,
    ) -> Result<()> {
        anyhow::ensure!(
            padded_bs <= self.batch_size,
            "{label} pointer table capacity {} is smaller than padded batch {padded_bs}",
            self.batch_size
        );
        anyhow::ensure!(
            active_slots.len() <= padded_bs,
            "{label} active slots {} exceed padded batch {padded_bs}",
            active_slots.len()
        );
        if self.last_padded_bs == padded_bs && self.last_active_slots == active_slots {
            return Ok(());
        }

        let num_linear_layers = self.state_ptrs.len();
        let mut row_state_ptrs = Vec::with_capacity(padded_bs);
        let mut row_conv_ptrs = Vec::with_capacity(padded_bs);
        for layer_idx in 0..num_linear_layers {
            row_state_ptrs.clear();
            row_conv_ptrs.clear();
            for &slot_idx in active_slots {
                row_state_ptrs.push(slot_state_ptrs[slot_idx][layer_idx]);
                row_conv_ptrs.push(slot_conv_ptrs[slot_idx][layer_idx]);
            }
            for _ in active_slots.len()..padded_bs {
                row_state_ptrs.push(padding_state_ptrs[layer_idx]);
                row_conv_ptrs.push(padding_conv_ptrs[layer_idx]);
            }
            ctx.stream
                .memcpy_htod(
                    &row_state_ptrs,
                    &mut self.state_ptrs[layer_idx].slice_mut(0..padded_bs),
                )
                .map_err(|e| {
                    anyhow::anyhow!("sync {label} linear state pointer table {layer_idx}: {e}")
                })?;
            ctx.stream
                .memcpy_htod(
                    &row_conv_ptrs,
                    &mut self.conv_state_ptrs[layer_idx].slice_mut(0..padded_bs),
                )
                .map_err(|e| {
                    anyhow::anyhow!("sync {label} conv state pointer table {layer_idx}: {e}")
                })?;
        }

        self.last_active_slots.clear();
        self.last_active_slots.extend_from_slice(active_slots);
        self.last_padded_bs = padded_bs;
        Ok(())
    }

    pub(crate) fn from_recurrent_refs(
        ctx: &DeviceContext,
        config: &Config35,
        recurrent_states: &mut [&mut RecurrentState],
        batch_size: usize,
        label: &str,
    ) -> Result<Self> {
        anyhow::ensure!(
            batch_size <= recurrent_states.len(),
            "{label} pointer table batch {batch_size} exceeds recurrent refs {}",
            recurrent_states.len()
        );
        let num_linear_layers = config.num_hidden_layers - config.num_full_attention_layers();
        let mut linear_state_ptrs = Vec::with_capacity(num_linear_layers);
        let mut linear_conv_state_ptrs = Vec::with_capacity(num_linear_layers);
        for layer_idx in 0..num_linear_layers {
            let mut state_ptrs = Vec::with_capacity(batch_size);
            let mut conv_state_ptrs = Vec::with_capacity(batch_size);
            for slot in recurrent_states.iter_mut().take(batch_size) {
                let state_ptr = {
                    let (ptr, _guard) = slot.layers[layer_idx].state.device_ptr_mut(&ctx.stream);
                    ptr
                };
                let conv_ptr = {
                    let (ptr, _guard) = slot.layers[layer_idx]
                        .conv_state
                        .data
                        .device_ptr_mut(&ctx.stream);
                    ptr
                };
                state_ptrs.push(state_ptr);
                conv_state_ptrs.push(conv_ptr);
            }
            linear_state_ptrs.push(ctx.stream.clone_htod(&state_ptrs).map_err(|e| {
                anyhow::anyhow!("copy {label} linear state pointer table {layer_idx}: {e}")
            })?);
            linear_conv_state_ptrs.push(ctx.stream.clone_htod(&conv_state_ptrs).map_err(|e| {
                anyhow::anyhow!("copy {label} conv state pointer table {layer_idx}: {e}")
            })?);
        }

        Ok(Self {
            state_ptrs: linear_state_ptrs,
            conv_state_ptrs: linear_conv_state_ptrs,
            batch_size,
            last_active_slots: Vec::new(),
            last_padded_bs: 0,
        })
    }

    pub(crate) fn validate_for(
        &self,
        config: &Config35,
        batch_size: usize,
        label: &str,
    ) -> Result<()> {
        let num_linear_layers = config.num_hidden_layers - config.num_full_attention_layers();
        anyhow::ensure!(
            self.batch_size >= batch_size,
            "{label} pointer table capacity {} is smaller than batch {batch_size}",
            self.batch_size
        );
        anyhow::ensure!(
            self.state_ptrs.len() == num_linear_layers
                && self.conv_state_ptrs.len() == num_linear_layers,
            "{label} pointer table layer count mismatch: state={}, conv={}, expected={num_linear_layers}",
            self.state_ptrs.len(),
            self.conv_state_ptrs.len()
        );
        Ok(())
    }
}

/// Device bytes of one request's recurrent state.
pub(crate) fn bytes_per_request(config: &Config35) -> usize {
    let (num_linear_layers, state_size, conv_state_size) = per_layer_dims(config);
    num_linear_layers
        * (state_size * std::mem::size_of::<f32>()
            + conv_state_size * std::mem::size_of::<half::bf16>())
}

impl RecurrentState {
    pub(crate) fn allocation_bytes(config: &Config35) -> usize {
        bytes_per_request(config)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn qwen35_4b_recurrent_allocation_is_49_125_mib() {
        let bytes = 24
            * (32 * 128 * 128 * std::mem::size_of::<f32>()
                + 8192 * 3 * std::mem::size_of::<half::bf16>());
        assert_eq!(bytes, 49 * 1024 * 1024 + 128 * 1024);
    }
}
