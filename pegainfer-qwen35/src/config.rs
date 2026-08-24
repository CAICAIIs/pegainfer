use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use serde_json::Value;

mod model;
mod tokenizer;
mod tp;

pub(crate) use model::*;
pub(crate) use tokenizer::*;
pub(crate) use tp::*;

/// Identity check that `json` is a Qwen3.5 config; size and shape validation belong to the config loader.
pub(crate) fn probe_config_json(json: &Value) -> Result<()> {
    let model_type = json.get("model_type").and_then(Value::as_str).unwrap_or("");
    if model_type != "qwen3_5" {
        bail!("not a Qwen3.5 config: model_type={model_type}");
    }
    let architectures: Vec<&str> = json
        .get("architectures")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    ensure!(
        architectures.contains(&"Qwen3_5ForConditionalGeneration"),
        "Qwen3.5 architectures must contain Qwen3_5ForConditionalGeneration"
    );
    let text_model_type = json
        .get("text_config")
        .and_then(|tc| tc.get("model_type"))
        .and_then(Value::as_str)
        .unwrap_or("");
    ensure!(
        text_model_type == "qwen3_5_text",
        "Qwen3.5 text_config.model_type must be qwen3_5_text, got {text_model_type}"
    );
    Ok(())
}

#[cfg(test)]

mod tp_tests {
    use super::*;

    fn test_config() -> Config35 {
        Config35 {
            hidden_size: 2560,
            intermediate_size: 9216,
            num_hidden_layers: 32,
            vocab_size: 248_320,
            selection_vocab: 248_320,
            rms_norm_eps: 1e-6,
            eos_token_id: 151_645,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 256,
            linear_num_key_heads: 16,
            linear_key_head_dim: 128,
            linear_num_value_heads: 32,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            rope_theta: 10_000.0,
            rotary_dim: 64,
            max_position_embeddings: 262_144,
            tie_word_embeddings: true,
            layer_types: vec![LayerType::LinearAttention; 32],
        }
    }

    #[test]
    fn default_tensor_parallel_is_tp1() {
        let config = test_config();
        let tp = TensorParallelConfig::default();

        tp.validate_for(&config, true).unwrap();
        assert!(!tp.is_sharded());
        assert_eq!(tp.shard_range(config.full_attn_q_dim()), (0, 4096));
        assert_eq!(config.local_num_attention_heads(tp), 16);
        assert_eq!(config.local_num_key_value_heads(tp), 4);
        assert_eq!(config.local_intermediate_size(tp), 9216);
        assert_eq!(config.local_full_attn_q_dim(tp), 4096);
        assert_eq!(config.local_full_attn_kv_dim(tp), 1024);
        assert_eq!(config.local_full_attn_gated_q_dim(tp), 8192);
    }

    #[test]
    fn computes_tp2_dense_local_dimensions() {
        let config = test_config();
        let tp = TensorParallelConfig {
            rank: 1,
            world_size: 2,
        };

        tp.validate_for(&config, false).unwrap();
        assert!(tp.is_sharded());
        assert_eq!(tp.shard_range(config.full_attn_q_dim()), (2048, 2048));
        assert_eq!(config.local_num_attention_heads(tp), 8);
        assert_eq!(config.local_num_key_value_heads(tp), 2);
        assert_eq!(config.local_intermediate_size(tp), 4608);
        assert_eq!(config.local_full_attn_q_dim(tp), 2048);
        assert_eq!(config.local_full_attn_kv_dim(tp), 512);
        assert_eq!(config.local_full_attn_gated_q_dim(tp), 4096);
    }

    #[test]
    fn rejects_invalid_world_size_and_rank() {
        let config = test_config();

        let err = TensorParallelConfig {
            rank: 0,
            world_size: 0,
        }
        .validate_for(&config, false)
        .unwrap_err()
        .to_string();
        assert!(err.contains("world_size must be >= 1"));

        let err = TensorParallelConfig {
            rank: 2,
            world_size: 2,
        }
        .validate_for(&config, false)
        .unwrap_err()
        .to_string();
        assert!(err.contains("rank 2 must be < world_size 2"));
    }

    #[test]
    fn rejects_indivisible_dense_dimensions() {
        let tp = TensorParallelConfig {
            rank: 0,
            world_size: 3,
        };

        let mut config = test_config();
        let err = tp.validate_for(&config, false).unwrap_err().to_string();
        assert!(err.contains("num_attention_heads=16 not divisible"));

        config.num_attention_heads = 15;
        config.num_key_value_heads = 4;
        let err = tp.validate_for(&config, false).unwrap_err().to_string();
        assert!(err.contains("num_key_value_heads=4 not divisible"));

        config.num_key_value_heads = 3;
        config.intermediate_size = 9217;
        let err = tp.validate_for(&config, false).unwrap_err().to_string();
        assert!(err.contains("intermediate_size=9217 not divisible"));
    }

    #[test]
    fn rejects_tensor_parallel_cuda_graph() {
        let config = test_config();
        let tp = TensorParallelConfig {
            rank: 0,
            world_size: 2,
        };

        let err = tp.validate_for(&config, true).unwrap_err().to_string();
        assert!(err.contains("eager-only"));
    }

    #[test]
    fn tp_does_not_require_linear_attention_divisibility() {
        let mut config = test_config();
        config.linear_num_key_heads = 17;
        config.linear_num_value_heads = 31;
        let tp = TensorParallelConfig {
            rank: 1,
            world_size: 2,
        };

        tp.validate_for(&config, false).unwrap();
    }
}

/// Schema kept identical to the pinned vLLM frontend; unread fields exist for
/// payload type-checking.
#[cfg(test)]
mod tests {
    use super::Config35;

    #[test]
    fn guard_accepts_48_value_heads() {
        let dir = tempfile::tempdir().unwrap();
        let json = r#"{
  "max_position_embeddings": 4096,
  "tie_word_embeddings": true,
  "text_config": {
    "hidden_size": 512,
    "intermediate_size": 1024,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "vocab_size": 1000,
    "rms_norm_eps": 1e-6,
    "layer_types": ["linear_attention", "full_attention"],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_value_head_dim": 128,
    "rope_parameters": { "rope_theta": 10000.0, "partial_rotary_factor": 0.25 },
    "eos_token_id": 0
  }
}"#;
        std::fs::write(dir.path().join("config.json"), json).unwrap();
        Config35::from_file(dir.path().to_str().unwrap()).expect("48 value heads must load");
    }

    #[test]
    fn guard_rejects_wide_linear_conv_decode_kernel() {
        let dir = tempfile::tempdir().unwrap();
        let json = r#"{
  "max_position_embeddings": 4096,
  "tie_word_embeddings": true,
  "text_config": {
    "hidden_size": 512,
    "intermediate_size": 1024,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "vocab_size": 1000,
    "rms_norm_eps": 1e-6,
    "layer_types": ["linear_attention", "full_attention"],
    "linear_conv_kernel_dim": 5,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_value_head_dim": 128,
    "rope_parameters": { "rope_theta": 10000.0, "partial_rotary_factor": 0.25 },
    "eos_token_id": 0
  }
}"#;
        std::fs::write(dir.path().join("config.json"), json).unwrap();

        let err = Config35::from_file(dir.path().to_str().unwrap())
            .expect_err("wide conv decode kernels must be rejected");
        assert!(
            err.to_string().contains("linear conv decode kernels"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn effective_vocab_is_the_dense_decodable_width() {
        let dir = tempfile::tempdir().unwrap();
        let json = r#"{
  "model": { "vocab": { "a": 0, "b": 1, "c": 2 } },
  "added_tokens": [ { "id": 3, "content": "<x>" } ]
}"#;
        std::fs::write(dir.path().join("tokenizer.json"), json).unwrap();
        let cfg = r#"{ "added_tokens_decoder": { "4": { "content": "<z>" }, "5": { "content": "<w>" }, "x": { "content": "<bad-key>" } } }"#;
        std::fs::write(dir.path().join("tokenizer_config.json"), cfg).unwrap();
        assert_eq!(
            super::tokenizer_effective_vocab(dir.path().to_str().unwrap()).unwrap(),
            6
        );
    }

    #[test]
    fn effective_vocab_fails_on_a_sparse_id_space() {
        let dir = tempfile::tempdir().unwrap();
        let json = r#"{ "model": { "vocab": { "a": 0, "b": 1 } } }"#;
        std::fs::write(dir.path().join("tokenizer.json"), json).unwrap();
        let cfg = r#"{ "added_tokens_decoder": { "5": { "content": "<z>" } } }"#;
        std::fs::write(dir.path().join("tokenizer_config.json"), cfg).unwrap();
        assert!(super::tokenizer_effective_vocab(dir.path().to_str().unwrap()).is_err());
    }

    #[test]
    fn one_invalid_decoder_entry_drops_all_decoder_tokens() {
        let dir = tempfile::tempdir().unwrap();
        let json = r#"{ "model": { "vocab": { "a": 0, "b": 1 } } }"#;
        std::fs::write(dir.path().join("tokenizer.json"), json).unwrap();
        let cfg = r#"{ "added_tokens_decoder": { "2": { "content": "<z>" }, "3": { "content": "<w>", "special": "not-a-bool" } } }"#;
        std::fs::write(dir.path().join("tokenizer_config.json"), cfg).unwrap();
        assert_eq!(
            super::tokenizer_effective_vocab(dir.path().to_str().unwrap()).unwrap(),
            2
        );
    }
}
