use std::sync::Arc;

use vllm_text::backend::hf::{ResolvedModelFiles, TokenizerSource};
use vllm_text::tokenizer::{
    DynTokenizer, HuggingFaceTokenizer, TekkenTokenizer, TiktokenTokenizer,
};
use vllm_text::{Error, Result};

pub fn load_tokenizer(model_id: &str) -> Result<DynTokenizer> {
    let files = resolve_model_files(model_id)?;
    tokenizer_from_source(&files.tokenizer)
}

pub fn tokenizer_from_source(source: &TokenizerSource) -> Result<DynTokenizer> {
    match source {
        TokenizerSource::HuggingFace(path) => Ok(Arc::new(HuggingFaceTokenizer::new(path)?)),
        TokenizerSource::Tiktoken(path) => Ok(Arc::new(TiktokenTokenizer::new(path)?)),
        TokenizerSource::Tekken(path) => Ok(Arc::new(TekkenTokenizer::new(path)?)),
    }
}

fn resolve_model_files(model_id: &str) -> Result<ResolvedModelFiles> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| {
            Error::Tokenizer(format!(
                "failed to create tokenizer resolver runtime: {err}"
            ))
        })?;

    runtime.block_on(ResolvedModelFiles::new(model_id))
}
