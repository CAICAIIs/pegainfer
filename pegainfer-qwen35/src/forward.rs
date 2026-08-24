//! Qwen3.5 forward pass: prefill, batch decode, CUDA-Graph decode, unified
//! prefill+decode, and recurrent (GDR/conv1d) attention.
//! Grouped here so the compute path has a single owner. Reaches crate-root
//! modules (`weights`/`config`/`decode_buffers`/etc.) via `crate::`.

pub(crate) mod batch_decode;
pub(crate) mod batch_decode_graph;
pub(crate) mod prefill;
pub(crate) mod recurrent;
pub(crate) mod unified_forward;
