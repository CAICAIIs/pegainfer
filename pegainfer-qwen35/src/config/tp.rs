use anyhow::Result;

use super::model::Config35;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TensorParallelConfig {
    pub(crate) rank: usize,
    pub(crate) world_size: usize,
}

impl Default for TensorParallelConfig {
    fn default() -> Self {
        Self {
            rank: 0,
            world_size: 1,
        }
    }
}
impl TensorParallelConfig {
    pub(crate) fn validate_for(self, config: &Config35, enable_cuda_graph: bool) -> Result<()> {
        if self.world_size == 0 {
            return Err(anyhow::anyhow!("tensor_parallel.world_size must be >= 1"));
        }
        if self.rank >= self.world_size {
            return Err(anyhow::anyhow!(
                "tensor_parallel.rank {} must be < world_size {}",
                self.rank,
                self.world_size
            ));
        }
        if self.is_sharded() && enable_cuda_graph {
            return Err(anyhow::anyhow!(
                "Qwen3.5 tensor parallelism is eager-only; disable CUDA Graph for tp world_size={}",
                self.world_size
            ));
        }
        if !config.num_attention_heads.is_multiple_of(self.world_size) {
            return Err(anyhow::anyhow!(
                "num_attention_heads={} not divisible by tp world_size={}",
                config.num_attention_heads,
                self.world_size
            ));
        }
        if !config.num_key_value_heads.is_multiple_of(self.world_size) {
            return Err(anyhow::anyhow!(
                "num_key_value_heads={} not divisible by tp world_size={}",
                config.num_key_value_heads,
                self.world_size
            ));
        }
        if !config.intermediate_size.is_multiple_of(self.world_size) {
            return Err(anyhow::anyhow!(
                "intermediate_size={} not divisible by tp world_size={}",
                config.intermediate_size,
                self.world_size
            ));
        }
        Ok(())
    }

    pub(crate) fn shard_range(self, total: usize) -> (usize, usize) {
        let shard_len = total / self.world_size;
        (self.rank * shard_len, shard_len)
    }

    pub(crate) fn is_sharded(self) -> bool {
        self.world_size > 1
    }
}
