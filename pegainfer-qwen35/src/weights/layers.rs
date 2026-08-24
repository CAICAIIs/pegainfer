//! Qwen3.5 weights: layer container types + weight-loading helpers. Split out of
//! weights.rs. Reaches the model/config via `use super::*;`.

use super::*;

/// Full attention layer weights (8 layers in Qwen3.5-4B).
pub(crate) struct FullAttentionLayer {
    /// Q projection including gate: [num_heads * head_dim * 2, hidden_size]
    pub(crate) q_proj: DeviceMatrix,
    /// K projection: [num_kv_heads * head_dim, hidden_size]
    pub(crate) k_proj: DeviceMatrix,
    /// V projection: [num_kv_heads * head_dim, hidden_size]
    pub(crate) v_proj: DeviceMatrix,
    /// Output projection: [hidden_size, num_heads * head_dim]
    pub(crate) o_proj: DeviceMatrix,
    /// QK norm weights: [head_dim] (broadcast to all heads)
    pub(crate) q_norm: DeviceVec,
    pub(crate) k_norm: DeviceVec,
}

/// Linear attention layer weights (24 layers in Qwen3.5-4B).
pub(crate) struct LinearAttentionLayer {
    /// Fused QKV projection: [q_dim + k_dim + v_dim, hidden_size]
    pub(crate) in_proj_qkv: DeviceMatrix,
    /// Z projection (for output gating): [z_dim, hidden_size]
    pub(crate) in_proj_z: DeviceMatrix,
    /// Beta projection: [num_value_heads, hidden_size]
    pub(crate) in_proj_b: DeviceMatrix,
    /// Alpha projection: [num_value_heads, hidden_size]
    pub(crate) in_proj_a: DeviceMatrix,
    /// Depthwise conv1d weight: [qkv_dim * conv_kernel_dim] (flattened from [qkv_dim, 1, 4])
    pub(crate) conv1d_weight: DeviceVec,
    /// dt_bias: [num_value_heads] bf16
    pub(crate) dt_bias: DeviceVec,
    /// A_log: [num_value_heads] f32
    pub(crate) a_log: CudaSlice<f32>,
    /// RMSNorm weight for output normalization: [value_head_dim] f32
    pub(crate) norm_weight: CudaSlice<f32>,
    /// Output projection: [hidden_size, z_dim]
    pub(crate) out_proj: DeviceMatrix,
}

/// Attention layer — either full or linear.
pub(crate) enum LayerKind {
    FullAttention(FullAttentionLayer),
    LinearAttention(LinearAttentionLayer),
}

/// MLP layer weights (shared between both layer types).
#[allow(clippy::struct_field_names)]
pub(crate) struct MLP35 {
    pub(crate) gate_up_proj: DeviceMatrix,
    pub(crate) down_proj: DeviceMatrix,
}

/// Transformer block for Qwen3.5.
pub(crate) struct TransformerBlock35 {
    pub(crate) input_layernorm: DeviceVec,
    pub(crate) attn: LayerKind,
    pub(crate) post_attention_layernorm: DeviceVec,
    pub(crate) mlp: MLP35,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GatedQShardRange {
    pub(crate) row_offset: usize,
    pub(crate) rows: usize,
}

pub(crate) fn full_attention_gated_q_shard_range(
    config: &Config35,
    geometry: LocalGeometry,
) -> GatedQShardRange {
    // HF/PegaInfer kernels interpret q_proj rows as per-head [q, gate] chunks.
    // Keep each local head's q rows adjacent to its gate rows.
    let local_heads = geometry.local_num_attention_heads();
    let head_start = geometry.rank() * local_heads;
    GatedQShardRange {
        row_offset: head_start * config.head_dim * 2,
        rows: local_heads * config.head_dim * 2,
    }
}

pub(crate) fn load_full_attention_gated_q_proj(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
    config: &Config35,
    geometry: LocalGeometry,
) -> Result<DeviceMatrix> {
    if !geometry.is_sharded() {
        return load_tensor_2d(ctx, shards, weight_map, name);
    }

    let range = full_attention_gated_q_shard_range(config, geometry);
    load_tensor_2d_row_shard(ctx, shards, weight_map, name, range.row_offset, range.rows)
}

pub(crate) fn load_tensor_2d_row_shard_if_needed(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
    geometry: LocalGeometry,
    row_offset: usize,
    rows: usize,
) -> Result<DeviceMatrix> {
    if geometry.is_sharded() {
        load_tensor_2d_row_shard(ctx, shards, weight_map, name, row_offset, rows)
    } else {
        load_tensor_2d(ctx, shards, weight_map, name)
    }
}

pub(crate) fn load_tensor_2d_col_shard_if_needed(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
    geometry: LocalGeometry,
    col_offset: usize,
    cols: usize,
) -> Result<DeviceMatrix> {
    if geometry.is_sharded() {
        load_tensor_2d_col_shard(ctx, shards, weight_map, name, col_offset, cols)
    } else {
        load_tensor_2d(ctx, shards, weight_map, name)
    }
}
