use crate::types::TokenUsage;
use serde::{Deserialize, Serialize};

/// Prompt caching configuration for Anthropic API.
/// Marks system prompt sections with `cache_control` for reduced token costs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptCacheConfig {
    /// Whether caching is enabled.
    pub enabled: bool,
    /// Minimum token count to consider caching a section.
    pub min_tokens: u32,
}

impl Default for PromptCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_tokens: 1024,
        }
    }
}

/// Tracks cumulative cache hits/misses for cost estimation.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub total_cache_creation_tokens: u64,
    pub total_cache_read_tokens: u64,
}

impl CacheStats {
    pub fn record(&mut self, usage: &TokenUsage) {
        self.total_cache_creation_tokens += u64::from(usage.cache_creation_input_tokens);
        self.total_cache_read_tokens += u64::from(usage.cache_read_input_tokens);
    }

    /// Estimated savings: cache reads are 90% cheaper than full input tokens.
    #[must_use]
    pub fn estimated_savings_ratio(&self) -> f64 {
        let total = self.total_cache_creation_tokens + self.total_cache_read_tokens;
        if total == 0 {
            return 0.0;
        }
        (self.total_cache_read_tokens as f64 * 0.9) / total as f64
    }
}
