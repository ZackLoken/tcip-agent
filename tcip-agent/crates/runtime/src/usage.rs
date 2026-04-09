/// Tracks cumulative token usage and estimated cost across turns.
#[derive(Debug, Default)]
pub struct UsageTracker {
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub turn_count: u32,
}

impl UsageTracker {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, input_tokens: u32, output_tokens: u32) {
        self.total_input_tokens += u64::from(input_tokens);
        self.total_output_tokens += u64::from(output_tokens);
        self.turn_count += 1;
    }

    /// Estimate cost in USD based on Claude pricing.
    /// Sonnet: $3/M input, $15/M output
    /// Opus: $15/M input, $75/M output
    #[must_use]
    pub fn estimated_cost_usd(&self, model: &str) -> f64 {
        let (input_rate, output_rate) = if model.contains("opus") {
            (15.0 / 1_000_000.0, 75.0 / 1_000_000.0)
        } else {
            (3.0 / 1_000_000.0, 15.0 / 1_000_000.0)
        };

        self.total_input_tokens as f64 * input_rate
            + self.total_output_tokens as f64 * output_rate
    }
}
