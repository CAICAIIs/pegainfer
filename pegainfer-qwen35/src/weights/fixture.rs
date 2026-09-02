//! Config fixtures shared by the weights unit tests.

use super::*;

pub(crate) fn test_config() -> Config35 {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        r#"{
  "max_position_embeddings": 262144,
  "tie_word_embeddings": true,
  "text_config": {
    "hidden_size": 2560,
    "intermediate_size": 9216,
    "num_hidden_layers": 1,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "vocab_size": 248320,
    "rms_norm_eps": 1e-6,
    "layer_types": ["linear_attention"],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_value_head_dim": 128,
    "rope_parameters": { "rope_theta": 10000.0, "partial_rotary_factor": 0.25 },
    "eos_token_id": 151645
  }
}"#,
    )
    .unwrap();
    Config35::from_file(dir.path().to_str().unwrap()).expect("fixture validates")
}

pub(crate) fn test_geometry(rank: usize, world_size: usize) -> LocalGeometry {
    let config = test_config();
    let tp = TensorParallelConfig::try_from((rank, world_size)).unwrap();
    LocalGeometry::try_new(&config, tp, false).unwrap()
}
