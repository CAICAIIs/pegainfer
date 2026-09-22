// Thin C wrappers around FlashInfer's attention kernels.
//
// We include FlashInfer headers (header-only C++) and instantiate only the
// template variants needed: bf16 Q/KV/O, NHD layout, no RoPE, at HEAD_DIM 128
// and 256 — the latter both with and without a sliding-window mask — plus the
// hd512 split-KV decode entry the global family reads through.
//
// FlashInfer's dispatchers internally instantiate multiple GQA group sizes
// (1,2,3,4,8) — this covers both Qwen3-4B (GQA=4) and Qwen3.5-4B (GQA=8).

#include "paged_launch.cuh"

extern "C" {

int paged_attention_decode_cuda(
    void* q, void* output, void* kv_data,
    int64_t k_offset_elems, int64_t v_offset_elems,
    int32_t* page_indices, int32_t* page_indptr, int32_t* last_page_len_d,
    int32_t* request_indices, int32_t* kv_tile_indices, int32_t* kv_chunk_size_ptr,
    int32_t num_qo_heads, int32_t num_kv_heads, int32_t head_dim,
    int32_t page_size, int32_t batch_size, int64_t stride_page,
    float sm_scale, void* stream)
{
  return decode_launch</*HEAD_DIM=*/128, Variant>(
      q, output, kv_data, k_offset_elems, v_offset_elems,
      page_indices, page_indptr, last_page_len_d,
      request_indices, kv_tile_indices, kv_chunk_size_ptr,
      num_qo_heads, num_kv_heads, head_dim, page_size, batch_size,
      stride_page, sm_scale, /*window_left=*/-1, stream);
}

int paged_attention_decode_split_kv_cuda(
    void* q, void* output, void* kv_data,
    int64_t k_offset_elems, int64_t v_offset_elems,
    int32_t* page_indices, int32_t* page_indptr, int32_t* last_page_len_d,
    int32_t* request_indices, int32_t* kv_tile_indices, int32_t* kv_chunk_size_ptr,
    int32_t* o_indptr, uint8_t* block_valid_mask,
    void* tmp_v, float* tmp_s,
    int32_t num_qo_heads, int32_t num_kv_heads, int32_t head_dim,
    int32_t page_size, int32_t batch_size, int32_t padded_batch_size,
    int64_t stride_page, float sm_scale, void* stream)
{
  return decode_split_kv_launch</*HEAD_DIM=*/128, Variant>(
      q, output, kv_data, k_offset_elems, v_offset_elems,
      page_indices, page_indptr, last_page_len_d,
      request_indices, kv_tile_indices, kv_chunk_size_ptr,
      o_indptr, block_valid_mask, tmp_v, tmp_s,
      num_qo_heads, num_kv_heads, head_dim, page_size,
      batch_size, padded_batch_size, stride_page, sm_scale,
      /*window_left=*/-1, stream);
}

// ---------------------------------------------------------------------------
// Paged KV append — writes one K and one V token per request to paged cache.
//
// Must be called AFTER RMSNorm + RoPE on K, and BEFORE the attention decode.
// V is appended as-is (no norm/RoPE).
// ---------------------------------------------------------------------------
int paged_kv_append_cuda(
    void*    kv_data,
    int64_t  k_offset_elems,
    int64_t  v_offset_elems,
    int32_t* page_indices,
    int32_t* page_indptr,
    int32_t* last_page_len_d,
    void*    key,                  // [batch_size * num_kv_heads * head_dim] bf16
    void*    value,                // [batch_size * num_kv_heads * head_dim] bf16
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  page_size,
    int32_t  batch_size,
    int64_t  stride_page,
    void*    stream)
{
  PEGAINFER_FFI_GUARD_BEGIN
    auto paged_kv = make_paged_kv(
        kv_data, k_offset_elems, v_offset_elems,
        page_indices, page_indptr, last_page_len_d,
        num_kv_heads, head_dim, page_size, batch_size, stride_page);

    return static_cast<int>(AppendPagedKVCacheDecode(
        paged_kv,
        reinterpret_cast<DType*>(key),
        reinterpret_cast<DType*>(value),
        reinterpret_cast<cudaStream_t>(stream)));
  PEGAINFER_FFI_GUARD_END(-1)
}

// ---------------------------------------------------------------------------
// Scatter contiguous KV cache into paged layout (one layer at a time).
//
// Source layout (HND per layer): k[head, pos, dim]
//   stride_n = head_dim, stride_h = max_seq_len * head_dim
//
// Called once after prefill to bridge contiguous → paged.
// ---------------------------------------------------------------------------
int paged_kv_scatter_cuda(
    void*    kv_data,
    int64_t  k_offset_elems,
    int64_t  v_offset_elems,
    int32_t* page_indices,
    int32_t* page_indptr,
    int32_t* last_page_len_d,
    void*    src_k,                // contiguous K for this layer [num_kv_heads, max_seq, head_dim]
    void*    src_v,                // contiguous V for this layer [num_kv_heads, max_seq, head_dim]
    int32_t* batch_indices,        // [nnz] = [0, 0, ..., 0]
    int32_t* positions,            // [nnz] = [0, 1, 2, ..., seq_len-1]
    int32_t  nnz,                  // = seq_len
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  page_size,
    int64_t  stride_page,
    int64_t  src_stride_n,         // = head_dim
    int64_t  src_stride_h,         // = max_seq_len * head_dim
    void*    stream)
{
  PEGAINFER_FFI_GUARD_BEGIN
    auto paged_kv = make_paged_kv(
        kv_data, k_offset_elems, v_offset_elems,
        page_indices, page_indptr, last_page_len_d,
        num_kv_heads, head_dim, page_size, /*batch_size=*/1, stride_page);

    return static_cast<int>(AppendPagedKVCache(
        paged_kv,
        reinterpret_cast<DType*>(src_k),
        reinterpret_cast<DType*>(src_v),
        batch_indices,
        positions,
        static_cast<uint32_t>(nnz),
        static_cast<size_t>(src_stride_n),
        static_cast<size_t>(src_stride_h),
        static_cast<size_t>(src_stride_n),   // V has same layout as K
        static_cast<size_t>(src_stride_h),
        reinterpret_cast<cudaStream_t>(stream)));
  PEGAINFER_FFI_GUARD_END(-1)
}

// ---------------------------------------------------------------------------
// Batch prefill with paged KV cache — wraps FlashInfer BatchPrefillWithPagedKVCache.
//
// Reads Q from col-major [q_dim, seq_len] layout (= HiddenStates).
// Reads K/V from paged layout (page-first, NHD within each block).
// No RoPE inside (caller does RoPE beforehand via qk_norm_rope_batched_decode_cuda).
// Causal mask, no split-KV (partition_kv=false).
//
// Plan metadata (request_indices, qo_tile_indices, etc.) is pre-computed by Rust
// and passed as GPU arrays. This avoids per-call GPU allocations.
// ---------------------------------------------------------------------------
// Return the number of Q tiles for given dimensions (needed to size plan arrays).
int32_t batch_prefill_paged_num_tiles(
    int32_t  seq_len,
    int32_t  num_qo_heads,
    int32_t  num_kv_heads,
    int32_t  head_dim)
{
    uint32_t group_size = num_qo_heads / num_kv_heads;
    int64_t packed_qo_len = static_cast<int64_t>(seq_len) * group_size;
    uint32_t cta_tile_q = FA2DetermineCtaTileQ(packed_qo_len, head_dim);
    return static_cast<int32_t>((packed_qo_len + cta_tile_q - 1) / cta_tile_q);
}

int32_t batch_prefill_paged_num_tiles_with_cta_tile_q(
    int32_t  seq_len,
    int32_t  num_qo_heads,
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  cta_tile_q_override)
{
    uint32_t group_size = num_qo_heads / num_kv_heads;
    int64_t packed_qo_len = static_cast<int64_t>(seq_len) * group_size;
    uint32_t cta_tile_q = resolve_prefill_cta_tile_q(
        packed_qo_len, head_dim, cta_tile_q_override);
    if (cta_tile_q == 0) {
        pegainfer_ffi_set_last_error("invalid cta_tile_q override");
        return -1;
    }
    return static_cast<int32_t>((packed_qo_len + cta_tile_q - 1) / cta_tile_q);
}

// Return the CTA tile size for batch prefill planning.
// Rust needs this to compute per-request tile counts that are consistent
// with the kernel dispatch.
int32_t batch_prefill_cta_tile_q(
    int32_t  total_seq_len,
    int32_t  num_qo_heads,
    int32_t  num_kv_heads,
    int32_t  head_dim)
{
    uint32_t group_size = num_qo_heads / num_kv_heads;
    int64_t packed_qo_len = static_cast<int64_t>(total_seq_len) * group_size;
    return static_cast<int32_t>(FA2DetermineCtaTileQ(packed_qo_len, head_dim));
}

int32_t batch_prefill_cta_tile_q_with_override(
    int32_t  total_seq_len,
    int32_t  num_qo_heads,
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  cta_tile_q_override)
{
    uint32_t group_size = num_qo_heads / num_kv_heads;
    int64_t packed_qo_len = static_cast<int64_t>(total_seq_len) * group_size;
    return static_cast<int32_t>(resolve_prefill_cta_tile_q(
        packed_qo_len, head_dim, cta_tile_q_override));
}

int batch_prefill_paged_cuda_with_cta_tile_q(
    void* q, void* output, void* kv_data,
    int64_t k_offset_elems, int64_t v_offset_elems,
    int32_t* page_indices, int32_t* page_indptr, int32_t* last_page_len_d,
    int32_t* q_indptr, int32_t* request_indices, int32_t* qo_tile_indices,
    int32_t* kv_tile_indices, int32_t* kv_chunk_size_ptr, uint32_t* total_num_rows,
    int32_t num_qo_heads, int32_t num_kv_heads, int32_t head_dim,
    int32_t page_size, int32_t seq_len, int32_t batch_size,
    int32_t padded_batch_size, int64_t stride_page, float sm_scale,
    int32_t cta_tile_q_override, void* stream)
{
  return prefill_paged_launch</*HEAD_DIM=*/128, Variant>(
      q, output, kv_data, k_offset_elems, v_offset_elems,
      page_indices, page_indptr, last_page_len_d, q_indptr,
      request_indices, qo_tile_indices, kv_tile_indices, kv_chunk_size_ptr,
      total_num_rows, num_qo_heads, num_kv_heads, head_dim, page_size,
      seq_len, batch_size, padded_batch_size, stride_page, sm_scale,
      cta_tile_q_override, /*window_left=*/-1, stream);
}

int batch_prefill_paged_cuda(
    void*    q,
    void*    output,
    void*    kv_data,
    int64_t  k_offset_elems,
    int64_t  v_offset_elems,
    int32_t* page_indices,
    int32_t* page_indptr,
    int32_t* last_page_len_d,
    int32_t* q_indptr,
    int32_t* request_indices,
    int32_t* qo_tile_indices,
    int32_t* kv_tile_indices,
    int32_t* kv_chunk_size_ptr,
    uint32_t* total_num_rows,
    int32_t  num_qo_heads,
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  page_size,
    int32_t  seq_len,
    int32_t  batch_size,
    int32_t  padded_batch_size,
    int64_t  stride_page,
    float    sm_scale,
    void*    stream)
{
  PEGAINFER_FFI_GUARD_BEGIN
    return batch_prefill_paged_cuda_with_cta_tile_q(
        q, output, kv_data, k_offset_elems, v_offset_elems,
        page_indices, page_indptr, last_page_len_d, q_indptr,
        request_indices, qo_tile_indices, kv_tile_indices,
        kv_chunk_size_ptr, total_num_rows, num_qo_heads, num_kv_heads,
        head_dim, page_size, seq_len, batch_size, padded_batch_size,
        stride_page, sm_scale, /*cta_tile_q_override=*/0, stream);
  PEGAINFER_FFI_GUARD_END(-1)
}

// ---------------------------------------------------------------------------
// Single-request prefill — wraps FlashInfer SinglePrefillWithKVCache.
//
// Reads Q from col-major [q_dim, seq_len] layout (= HiddenStates).
// Reads K/V from contiguous HND cache: k[head, pos, dim].
// No RoPE inside (caller does RoPE beforehand via prefill_attention_prep_cuda).
// Causal mask, no split-KV (tmp=nullptr).
// ---------------------------------------------------------------------------
int single_prefill_cuda(
    // Q and output (HiddenStates col-major: [q_dim, seq_len])
    void*    q,
    void*    output,
    // Contiguous KV cache (HND per-layer: k[head, pos, dim])
    void*    k_cache,
    void*    v_cache,
    // Dimensions
    int32_t  num_qo_heads,
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  seq_len,          // number of Q tokens (qo_len)
    int32_t  kv_len,           // total KV length (start_pos + seq_len)
    int32_t  max_seq_len,      // allocated cache rows (for HND stride)
    float    sm_scale,
    // Stream
    void*    stream)
{
  return single_prefill_launch</*HEAD_DIM=*/128, Variant>(
      q, output, k_cache, v_cache, num_qo_heads, num_kv_heads,
      head_dim, seq_len, kv_len, max_seq_len,
      sm_scale, stream);
}

int single_prefill_nhd_noncausal_cuda(
    // Q and output (HiddenStates token-major: [seq_len, q_dim])
    void*    q,
    void*    output,
    // Contiguous KV cache (HiddenStates token-major: [max_seq_len, kv_dim])
    void*    k_cache,
    void*    v_cache,
    int32_t  num_qo_heads,
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  seq_len,
    int32_t  kv_len,
    int32_t  max_seq_len,
    float    sm_scale,
    void*    stream)
{
  PEGAINFER_FFI_GUARD_BEGIN
    if (q == nullptr || output == nullptr || k_cache == nullptr || v_cache == nullptr ||
        num_qo_heads <= 0 || num_kv_heads <= 0 || head_dim != 128 ||
        seq_len <= 0 || kv_len <= 0 || max_seq_len < kv_len) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    uint32_t q_stride_n  = num_qo_heads * head_dim;
    uint32_t q_stride_h  = head_dim;
    uint32_t kv_stride_n = num_kv_heads * head_dim;
    uint32_t kv_stride_h = head_dim;

    PrefillParamsT params(
        reinterpret_cast<DType*>(q),
        reinterpret_cast<DType*>(k_cache),
        reinterpret_cast<DType*>(v_cache),
        /*maybe_custom_mask=*/nullptr,
        reinterpret_cast<DType*>(output),
        /*lse=*/nullptr,
        /*maybe_alibi_slopes=*/nullptr,
        num_qo_heads,
        num_kv_heads,
        static_cast<uint32_t>(seq_len),
        static_cast<uint32_t>(kv_len),
        q_stride_n,
        q_stride_h,
        kv_stride_n,
        kv_stride_h,
        static_cast<uint32_t>(head_dim),
        /*window_left=*/-1,
        /*logits_soft_cap=*/0.0f,
        sm_scale,
        /*rope_scale=*/1.0f,
        /*rope_theta=*/1e6f);

    return static_cast<int>(
        SinglePrefillWithKVCacheDispatched<
            /*HEAD_DIM_QK=*/128,
            /*HEAD_DIM_VO=*/128,
            PosEncodingMode::kNone,
            /*USE_FP16_QK_REDUCTION=*/false,
            MaskMode::kNone,
            Variant,
            PrefillParamsT>(
            params,
            /*tmp=*/nullptr,
            reinterpret_cast<cudaStream_t>(stream)));
  PEGAINFER_FFI_GUARD_END(-1)
}

// Identical to single_prefill_nhd_noncausal_cuda but dispatched with
// HEAD_DIM_QK/VO=64 — the GLM5.2 DSpark drafter's block attention (64 MHA
// heads x head_dim 64).
int single_prefill_nhd_noncausal_cuda_hd64(
    void*    q,
    void*    output,
    void*    k_cache,
    void*    v_cache,
    int32_t  num_qo_heads,
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  seq_len,
    int32_t  kv_len,
    int32_t  max_seq_len,
    float    sm_scale,
    void*    stream)
{
  PEGAINFER_FFI_GUARD_BEGIN
    if (q == nullptr || output == nullptr || k_cache == nullptr || v_cache == nullptr ||
        num_qo_heads <= 0 || num_kv_heads <= 0 || head_dim != 64 ||
        seq_len <= 0 || kv_len <= 0 || max_seq_len < kv_len) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    uint32_t q_stride_n  = num_qo_heads * head_dim;
    uint32_t q_stride_h  = head_dim;
    uint32_t kv_stride_n = num_kv_heads * head_dim;
    uint32_t kv_stride_h = head_dim;

    PrefillParamsT params(
        reinterpret_cast<DType*>(q),
        reinterpret_cast<DType*>(k_cache),
        reinterpret_cast<DType*>(v_cache),
        /*maybe_custom_mask=*/nullptr,
        reinterpret_cast<DType*>(output),
        /*lse=*/nullptr,
        /*maybe_alibi_slopes=*/nullptr,
        num_qo_heads,
        num_kv_heads,
        static_cast<uint32_t>(seq_len),
        static_cast<uint32_t>(kv_len),
        q_stride_n,
        q_stride_h,
        kv_stride_n,
        kv_stride_h,
        static_cast<uint32_t>(head_dim),
        /*window_left=*/-1,
        /*logits_soft_cap=*/0.0f,
        sm_scale,
        /*rope_scale=*/1.0f,
        /*rope_theta=*/1e6f);

    return static_cast<int>(
        SinglePrefillWithKVCacheDispatched<
            /*HEAD_DIM_QK=*/64,
            /*HEAD_DIM_VO=*/64,
            PosEncodingMode::kNone,
            /*USE_FP16_QK_REDUCTION=*/false,
            MaskMode::kNone,
            Variant,
            PrefillParamsT>(
            params,
            /*tmp=*/nullptr,
            reinterpret_cast<cudaStream_t>(stream)));
  PEGAINFER_FFI_GUARD_END(-1)
}

// Causal variant of single_prefill_nhd_noncausal_cuda: identical NHD token-major
// layout (q/output [seq, q_dim], k/v [max_seq, kv_dim]) but a causal mask, so the
// N query tokens at the tail of the cache attend only positions <= their own.
// Used for EAGLE-3's teacher-forced prefill in one batched forward (query i at
// absolute position kv_len - seq_len + i, FlashInfer's causal alignment).
int single_prefill_nhd_causal_cuda(
    void*    q,
    void*    output,
    void*    k_cache,
    void*    v_cache,
    int32_t  num_qo_heads,
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  seq_len,
    int32_t  kv_len,
    int32_t  max_seq_len,
    float    sm_scale,
    void*    stream)
{
  PEGAINFER_FFI_GUARD_BEGIN
    if (q == nullptr || output == nullptr || k_cache == nullptr || v_cache == nullptr ||
        num_qo_heads <= 0 || num_kv_heads <= 0 || head_dim != 128 ||
        num_qo_heads % num_kv_heads != 0 ||
        seq_len <= 0 || kv_len <= 0 || max_seq_len < kv_len || seq_len > kv_len) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    uint32_t q_stride_n  = num_qo_heads * head_dim;
    uint32_t q_stride_h  = head_dim;
    uint32_t kv_stride_n = num_kv_heads * head_dim;
    uint32_t kv_stride_h = head_dim;

    PrefillParamsT params(
        reinterpret_cast<DType*>(q),
        reinterpret_cast<DType*>(k_cache),
        reinterpret_cast<DType*>(v_cache),
        /*maybe_custom_mask=*/nullptr,
        reinterpret_cast<DType*>(output),
        /*lse=*/nullptr,
        /*maybe_alibi_slopes=*/nullptr,
        num_qo_heads,
        num_kv_heads,
        static_cast<uint32_t>(seq_len),
        static_cast<uint32_t>(kv_len),
        q_stride_n,
        q_stride_h,
        kv_stride_n,
        kv_stride_h,
        static_cast<uint32_t>(head_dim),
        /*window_left=*/-1,
        /*logits_soft_cap=*/0.0f,
        sm_scale,
        /*rope_scale=*/1.0f,
        /*rope_theta=*/1e6f);

    return static_cast<int>(
        SinglePrefillWithKVCacheDispatched<
            /*HEAD_DIM_QK=*/128,
            /*HEAD_DIM_VO=*/128,
            PosEncodingMode::kNone,
            /*USE_FP16_QK_REDUCTION=*/false,
            MaskMode::kCausal,
            Variant,
            PrefillParamsT>(
            params,
            /*tmp=*/nullptr,
            reinterpret_cast<cudaStream_t>(stream)));
  PEGAINFER_FFI_GUARD_END(-1)
}

// ---------------------------------------------------------------------------
// Single-query DECODE over the EAGLE-3 draft's contiguous NHD KV cache.
//
// The chain drafter advances one token per step, so each draft attention is a
// pure decode: exactly ONE query attends the whole [0, kv_len) prefix. This uses
// FlashInfer's dedicated single-query decode path (GEMV-style over the KV),
// which is *structurally* single-query — SingleDecodeParams::get_qo_len() is
// hard-wired to 1 — so, unlike single_prefill_nhd_noncausal_cuda (a prefill
// template forced to qo_len==1), it cannot be silently misused for a multi-query
// batch (a footgun once the draft chain is batched). Same NHD token-major layout
// as the *_nhd_* prefill pair: q/output [1, q_dim], k/v [max_seq_len, kv_dim].
// No RoPE inside (the caller applies eagle3_rope first).
// ---------------------------------------------------------------------------
using DecodeParamsT = SingleDecodeParams<DType, DType, DType>;

int single_decode_nhd_cuda(
    void*    q,            // [1, q_dim] token-major — the single decode query
    void*    output,       // [1, q_dim]
    void*    k_cache,      // [max_seq_len, kv_dim] NHD (k[pos, head, dim])
    void*    v_cache,
    int32_t  num_qo_heads,
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  kv_len,       // positions to attend: [0, kv_len)
    int32_t  max_seq_len,  // allocated cache rows (validation parity with the *_nhd_* pair)
    float    sm_scale,
    void*    stream)
{
  PEGAINFER_FFI_GUARD_BEGIN
    if (q == nullptr || output == nullptr || k_cache == nullptr || v_cache == nullptr ||
        num_qo_heads <= 0 || num_kv_heads <= 0 || head_dim != 128 ||
        num_qo_heads % num_kv_heads != 0 ||
        kv_len <= 0 || max_seq_len < kv_len) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    DecodeParamsT params(
        reinterpret_cast<DType*>(q),
        reinterpret_cast<DType*>(k_cache),
        reinterpret_cast<DType*>(v_cache),
        reinterpret_cast<DType*>(output),
        /*maybe_alibi_slopes=*/nullptr,
        /*seq_len(=kv_len)=*/static_cast<uint32_t>(kv_len),
        static_cast<uint32_t>(num_qo_heads),
        static_cast<uint32_t>(num_kv_heads),
        QKVLayout::kNHD,
        static_cast<uint32_t>(head_dim),
        /*window_left=*/-1,
        /*logits_soft_cap=*/0.0f,
        sm_scale,
        /*rope_scale=*/1.0f,
        /*rope_theta=*/1e6f);

    return static_cast<int>(
        SingleDecodeWithKVCacheDispatched<
            /*HEAD_DIM=*/128,
            PosEncodingMode::kNone,
            Variant,
            DecodeParamsT>(
            params,
            /*tmp=*/nullptr,
            reinterpret_cast<cudaStream_t>(stream)));
  PEGAINFER_FFI_GUARD_END(-1)
}

// ---------------------------------------------------------------------------
// Single-request prefill for HEAD_DIM=256 — wraps FlashInfer SinglePrefillWithKVCache.
//
// Identical to single_prefill_cuda but instantiated with HEAD_DIM_QK/VO=256.
// Reads Q from col-major [q_dim, seq_len] (HiddenStates layout).
// Reads K/V from contiguous HND cache: k[head, pos, dim].
// No RoPE inside (caller does QK norm + partial RoPE beforehand).
// Causal mask, no split-KV.
//
// Used by Qwen3.5-4B multi-token prefill.  Single-token decode still routes to
// the Triton AOT path (CUDA-Graph safe) until Phase 2d introduces paged decode.
// ---------------------------------------------------------------------------
int single_prefill_cuda_hd256(
    void*    q,
    void*    output,
    void*    k_cache,
    void*    v_cache,
    int32_t  num_qo_heads,
    int32_t  num_kv_heads,
    int32_t  seq_len,          // number of Q tokens (qo_len)
    int32_t  kv_len,           // total KV length (start_pos + seq_len)
    int32_t  max_seq_len,      // allocated cache rows (for HND stride)
    float    sm_scale,
    void*    stream)
{
  return single_prefill_launch</*HEAD_DIM=*/256, Variant>(
      q, output, k_cache, v_cache, num_qo_heads, num_kv_heads,
      /*head_dim=*/256, seq_len, kv_len, max_seq_len,
      sm_scale, stream);
}

// ---------------------------------------------------------------------------
// HEAD_DIM=256 paged entry points.  Qwen3.5-4B uses the full-attention pair;
// the windowed pair applies the sliding-window mask for Gemma 4's local layers.
// ---------------------------------------------------------------------------
int paged_attention_decode_cuda_hd256(
    void* q, void* output, void* kv_data,
    int64_t k_offset_elems, int64_t v_offset_elems,
    int32_t* page_indices, int32_t* page_indptr, int32_t* last_page_len_d,
    int32_t* request_indices, int32_t* kv_tile_indices, int32_t* kv_chunk_size_ptr,
    int32_t num_qo_heads, int32_t num_kv_heads, int32_t head_dim,
    int32_t page_size, int32_t batch_size, int64_t stride_page,
    float sm_scale, void* stream)
{
  return decode_launch</*HEAD_DIM=*/256, Variant>(
      q, output, kv_data, k_offset_elems, v_offset_elems,
      page_indices, page_indptr, last_page_len_d,
      request_indices, kv_tile_indices, kv_chunk_size_ptr,
      num_qo_heads, num_kv_heads, head_dim, page_size, batch_size,
      stride_page, sm_scale, /*window_left=*/-1, stream);
}

int paged_attention_decode_window_cuda_hd256(
    void* q, void* output, void* kv_data,
    int64_t k_offset_elems, int64_t v_offset_elems,
    int32_t* page_indices, int32_t* page_indptr, int32_t* last_page_len_d,
    int32_t* request_indices, int32_t* kv_tile_indices, int32_t* kv_chunk_size_ptr,
    int32_t num_qo_heads, int32_t num_kv_heads, int32_t head_dim,
    int32_t page_size, int32_t batch_size, int64_t stride_page,
    float sm_scale, int32_t window_left, void* stream)
{
  return decode_launch</*HEAD_DIM=*/256, WindowVariant>(
      q, output, kv_data, k_offset_elems, v_offset_elems,
      page_indices, page_indptr, last_page_len_d,
      request_indices, kv_tile_indices, kv_chunk_size_ptr,
      num_qo_heads, num_kv_heads, head_dim, page_size, batch_size,
      stride_page, sm_scale, window_left, stream);
}

// Split-KV decode at head_dim 512 — the Gemma global family's decode
// read. The non-partitioned grid is (pseudo-requests, kv heads) CTAs and
// starves the device; chunking the KV brings the grid to occupancy.
int paged_attention_decode_split_kv_cuda_hd512(
    void* q, void* output, void* kv_data,
    int64_t k_offset_elems, int64_t v_offset_elems,
    int32_t* page_indices, int32_t* page_indptr, int32_t* last_page_len_d,
    int32_t* request_indices, int32_t* kv_tile_indices, int32_t* kv_chunk_size_ptr,
    int32_t* o_indptr, uint8_t* block_valid_mask,
    void* tmp_v, float* tmp_s,
    int32_t num_qo_heads, int32_t num_kv_heads, int32_t head_dim,
    int32_t page_size, int32_t batch_size, int32_t padded_batch_size,
    int64_t stride_page, float sm_scale, void* stream)
{
  return decode_split_kv_launch</*HEAD_DIM=*/512, Variant>(
      q, output, kv_data, k_offset_elems, v_offset_elems,
      page_indices, page_indptr, last_page_len_d,
      request_indices, kv_tile_indices, kv_chunk_size_ptr,
      o_indptr, block_valid_mask, tmp_v, tmp_s,
      num_qo_heads, num_kv_heads, head_dim, page_size,
      batch_size, padded_batch_size, stride_page, sm_scale,
      /*window_left=*/-1, stream);
}

// ---------------------------------------------------------------------------
// Split-KV decode at head_dim 256, written for sm_80.
//
// The FlashInfer path above launches nblks(padded_batch_size, num_kv_heads) and
// never grows the grid with KV length, so a single-request decode runs four CTAs
// and a 16-request decode runs 64, each with 128 threads walking a four-position
// inner tile. On A100 that leaves the device idle: measured per layer-step, the
// FlashInfer kernel costs about 154 us at c16 and about 156 us at bs1, where
// vLLM's flash_fwd_splitkv costs 72 us and 26 us.
//
// This kernel splits the KV range across CTAs instead. One CTA per (split, kv
// head, request); each warp owns one query head of the GQA group and streams its
// share of the range one position at a time straight into registers, with no
// shared memory and no barrier. Partials are merged by a second kernel, one CTA
// per (request, query head).
//
// Measured standalone against the same paged layout: 33.6 us at bs1 / ctx 1024
// and 124.3 us at c16 / ctx 1024, against a 54 us memory floor at c16 (a
// load-only control reaches 1235 GB/s). Serving measurements at every bucket
// this kernel is admitted for beat the FlashInfer path — see the threshold in
// pegainfer-qwen35/src/decode_buffers.rs — and c16 gains less than bs1 does
// because at that size the kernel is closer to its own arithmetic limit. The
// remaining c16 headroom needs tensor cores.
// ---------------------------------------------------------------------------

constexpr int kSplitDecodeHeadDim = 256;
constexpr int kSplitDecodeVec = 8;   // bf16 per lane per head-dim slice (32 * 8 = 256)
constexpr int kSplitDecodeGqa = 4;   // Qwen3.5-4B: 16 query heads over 4 kv heads
constexpr int kSplitDecodeThreads = 32 * kSplitDecodeGqa;

__global__ void __launch_bounds__(kSplitDecodeThreads)
split_decode_partial_hd256_kernel(
    const __nv_bfloat16* __restrict__ q,        // [batch, num_qo_heads, head_dim]
    const __nv_bfloat16* __restrict__ k_pool,   // paged NHD, [page][page_size][kv_heads][head_dim]
    const __nv_bfloat16* __restrict__ v_pool,
    int64_t stride_page,
    int page_size,
    const int32_t* __restrict__ page_indices,
    const int32_t* __restrict__ page_indptr,
    const int32_t* __restrict__ kv_lens,
    float* __restrict__ partial_o,              // [splits, batch, num_qo_heads, head_dim]
    float* __restrict__ partial_m,              // [splits, batch, num_qo_heads]
    float* __restrict__ partial_l,              // [splits, batch, num_qo_heads]
    int num_qo_heads, int num_kv_heads, int num_splits, float sm_scale) {
  const int split = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int b = blockIdx.z;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int q_head = kv_head * kSplitDecodeGqa + warp;
  const int head_base = lane * kSplitDecodeVec;
  const int batch_stride = gridDim.z;

  float* o_out =
      partial_o + (((int64_t)split * batch_stride + b) * num_qo_heads + q_head) * kSplitDecodeHeadDim;
  float* m_out = partial_m + ((int64_t)split * batch_stride + b) * num_qo_heads + q_head;
  float* l_out = partial_l + ((int64_t)split * batch_stride + b) * num_qo_heads + q_head;

  // A split that holds no positions falls out of the loop below with the
  // neutral state already in its registers: o = 0, l = 0, m = -inf. Each warp
  // writes its own query head's partial, so the merge kernel sees a zero weight
  // (exp(-inf - m) == 0) for that split instead of a stale row. Writing that
  // state from one warp, or once per CTA, would silently leave the other heads
  // holding whatever the previous layer left in the buffer.
  const int kv_len = kv_lens[b];
  const int chunk = (kv_len + num_splits - 1) / num_splits;
  const int kv_start = split * chunk;
  const int kv_end = min(kv_start + chunk, kv_len);

  float qf[kSplitDecodeVec];
  {
    const __nv_bfloat16* qp =
        q + ((int64_t)b * num_qo_heads + q_head) * kSplitDecodeHeadDim + head_base;
    const uint4 qv = *reinterpret_cast<const uint4*>(qp);
    const __nv_bfloat16* qh = reinterpret_cast<const __nv_bfloat16*>(&qv);
#pragma unroll
    for (int i = 0; i < kSplitDecodeVec; ++i) {
      qf[i] = __bfloat162float(qh[i]);
    }
  }

  float m_i = -INFINITY;
  float l_i = 0.0f;
  float o_i[kSplitDecodeVec];
#pragma unroll
  for (int i = 0; i < kSplitDecodeVec; ++i) {
    o_i[i] = 0.0f;
  }

  const int page_base = page_indptr[b];

  // Every position's K and V are issued one position ahead, so a warp keeps two
  // iterations' loads in flight instead of one. The loop around a global load is
  // this kernel's whole cost: it runs at a 21% issue utilisation because each
  // iteration consumes its own load before issuing the next, while a load-only
  // control moves the same bytes from the same addresses 2.3x faster.
  uint4 k_cur{};
  uint4 v_cur{};
  int64_t off_cur = 0;
  if (kv_start < kv_end) {
    const int page = kv_start / page_size;
    const int slot = kv_start - page * page_size;
    off_cur = (int64_t)page_indices[page_base + page] * stride_page +
              ((int64_t)slot * num_kv_heads + kv_head) * kSplitDecodeHeadDim + head_base;
    k_cur = *reinterpret_cast<const uint4*>(&k_pool[off_cur]);
    v_cur = *reinterpret_cast<const uint4*>(&v_pool[off_cur]);
  }

  for (int j = kv_start; j < kv_end; ++j) {
    // j is uniform across the warp, so the guard is a uniform branch.
    uint4 k_next{};
    uint4 v_next{};
    int64_t off_next = off_cur;
    if (j + 1 < kv_end) {
      const int ntoken = j + 1;
      const int npage = ntoken / page_size;
      const int nslot = ntoken - npage * page_size;
      off_next = (int64_t)page_indices[page_base + npage] * stride_page +
                 ((int64_t)nslot * num_kv_heads + kv_head) * kSplitDecodeHeadDim + head_base;
      k_next = *reinterpret_cast<const uint4*>(&k_pool[off_next]);
      v_next = *reinterpret_cast<const uint4*>(&v_pool[off_next]);
    }

    const __nv_bfloat16* kh = reinterpret_cast<const __nv_bfloat16*>(&k_cur);
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < kSplitDecodeVec; ++i) {
      s += qf[i] * __bfloat162float(kh[i]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      s += __shfl_xor_sync(0xffffffffu, s, o);
    }
    s *= sm_scale;

    float p;
    if (s > m_i) {
      const float alpha = __expf(m_i - s);
#pragma unroll
      for (int i = 0; i < kSplitDecodeVec; ++i) {
        o_i[i] *= alpha;
      }
      l_i *= alpha;
      m_i = s;
      p = 1.0f;
    } else {
      p = __expf(s - m_i);
    }
    l_i += p;

    const __nv_bfloat16* vh = reinterpret_cast<const __nv_bfloat16*>(&v_cur);
#pragma unroll
    for (int i = 0; i < kSplitDecodeVec; ++i) {
      o_i[i] += p * __bfloat162float(vh[i]);
    }

    k_cur = k_next;
    v_cur = v_next;
    off_cur = off_next;
  }

#pragma unroll
  for (int i = 0; i < kSplitDecodeVec; ++i) {
    o_out[head_base + i] = o_i[i];
  }
  if (lane == 0) {
    *m_out = m_i;
    *l_out = l_i;
  }
}

// One CTA per (request, query head), one thread per head dimension.
__global__ void __launch_bounds__(kSplitDecodeHeadDim)
split_decode_combine_hd256_kernel(
    const float* __restrict__ partial_o,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    __nv_bfloat16* __restrict__ output,
    int num_qo_heads, int num_splits, int batch_size) {
  const int b = blockIdx.x;
  const int q_head = blockIdx.y;
  const int d = threadIdx.x;

  float m = -FLT_MAX;
  for (int s = 0; s < num_splits; ++s) {
    m = fmaxf(m, partial_m[((int64_t)s * batch_size + b) * num_qo_heads + q_head]);
  }
  float acc = 0.0f;
  float l = 0.0f;
  for (int s = 0; s < num_splits; ++s) {
    const int64_t row = ((int64_t)s * batch_size + b) * num_qo_heads + q_head;
    const float scale = __expf(partial_m[row] - m);
    acc += partial_o[row * kSplitDecodeHeadDim + d] * scale;
    l += partial_l[row] * scale;
  }
  output[((int64_t)b * num_qo_heads + q_head) * kSplitDecodeHeadDim + d] =
      __float2bfloat16(acc / l);
}

int paged_attention_decode_split_hd256_cuda(
    void* q, void* output, void* kv_data,
    int64_t k_offset_elems, int64_t v_offset_elems,
    int32_t* page_indices, int32_t* page_indptr,
    int32_t* kv_chunk_size_ptr,
    void* partial_o, void* partial_m, void* partial_l,
    int32_t num_qo_heads, int32_t num_kv_heads, int32_t head_dim,
    int32_t page_size, int32_t batch_size, int32_t num_splits,
    int64_t stride_page, float sm_scale, void* stream)
{
  PEGAINFER_FFI_GUARD_BEGIN
    if (num_splits < 1) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
    // The kernel is shaped for Qwen3.5-4B's GQA group: one warp per query head.
    if (head_dim != kSplitDecodeHeadDim ||
        num_qo_heads != num_kv_heads * kSplitDecodeGqa) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
    const __nv_bfloat16* k_base =
        reinterpret_cast<const __nv_bfloat16*>(kv_data) + k_offset_elems;
    const __nv_bfloat16* v_base =
        reinterpret_cast<const __nv_bfloat16*>(kv_data) + v_offset_elems;

    dim3 grid(num_splits, num_kv_heads, batch_size);
    split_decode_partial_hd256_kernel<<<grid, kSplitDecodeThreads, 0, cu_stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q), k_base, v_base, stride_page, page_size,
        page_indices, page_indptr, kv_chunk_size_ptr,
        reinterpret_cast<float*>(partial_o), reinterpret_cast<float*>(partial_m),
        reinterpret_cast<float*>(partial_l), num_qo_heads, num_kv_heads, num_splits, sm_scale);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
      return static_cast<int>(status);
    }

    dim3 combine_grid(batch_size, num_qo_heads);
    split_decode_combine_hd256_kernel<<<combine_grid, kSplitDecodeHeadDim, 0, cu_stream>>>(
        reinterpret_cast<const float*>(partial_o), reinterpret_cast<const float*>(partial_m),
        reinterpret_cast<const float*>(partial_l), reinterpret_cast<__nv_bfloat16*>(output),
        num_qo_heads, num_splits, batch_size);
    return static_cast<int>(cudaGetLastError());
  PEGAINFER_FFI_GUARD_END(-1)
}

int batch_prefill_paged_cuda_hd256(
    void* q, void* output, void* kv_data,
    int64_t k_offset_elems, int64_t v_offset_elems,
    int32_t* page_indices, int32_t* page_indptr, int32_t* last_page_len_d,
    int32_t* q_indptr, int32_t* request_indices, int32_t* qo_tile_indices,
    int32_t* kv_tile_indices, int32_t* kv_chunk_size_ptr, uint32_t* total_num_rows,
    int32_t num_qo_heads, int32_t num_kv_heads, int32_t head_dim,
    int32_t page_size, int32_t seq_len, int32_t batch_size,
    int32_t padded_batch_size, int64_t stride_page, float sm_scale,
    void* stream)
{
  return prefill_paged_launch</*HEAD_DIM=*/256, Variant>(
      q, output, kv_data, k_offset_elems, v_offset_elems,
      page_indices, page_indptr, last_page_len_d, q_indptr,
      request_indices, qo_tile_indices, kv_tile_indices, kv_chunk_size_ptr,
      total_num_rows, num_qo_heads, num_kv_heads, head_dim, page_size,
      seq_len, batch_size, padded_batch_size, stride_page, sm_scale,
      /*cta_tile_q_override=*/0, /*window_left=*/-1, stream);
}

int batch_prefill_paged_window_cuda_hd256(
    void* q, void* output, void* kv_data,
    int64_t k_offset_elems, int64_t v_offset_elems,
    int32_t* page_indices, int32_t* page_indptr, int32_t* last_page_len_d,
    int32_t* q_indptr, int32_t* request_indices, int32_t* qo_tile_indices,
    int32_t* kv_tile_indices, int32_t* kv_chunk_size_ptr, uint32_t* total_num_rows,
    int32_t num_qo_heads, int32_t num_kv_heads, int32_t head_dim,
    int32_t page_size, int32_t seq_len, int32_t batch_size,
    int32_t padded_batch_size, int64_t stride_page, float sm_scale,
    int32_t window_left, void* stream)
{
  return prefill_paged_launch</*HEAD_DIM=*/256, WindowVariant>(
      q, output, kv_data, k_offset_elems, v_offset_elems,
      page_indices, page_indptr, last_page_len_d, q_indptr,
      request_indices, qo_tile_indices, kv_tile_indices, kv_chunk_size_ptr,
      total_num_rows, num_qo_heads, num_kv_heads, head_dim, page_size,
      seq_len, batch_size, padded_batch_size, stride_page, sm_scale,
      /*cta_tile_q_override=*/0, window_left, stream);
}

} // extern "C"
