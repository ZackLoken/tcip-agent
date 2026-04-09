use chrono::{DateTime, Utc};
use tracing::debug;

/// Priority levels for injected context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ContextPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// A single piece of context to inject into the system prompt.
#[derive(Debug, Clone)]
pub struct ContextEntry {
    /// Source identifier (e.g. "training_progress", "eval_metrics").
    pub source: String,
    /// Unique key for deduplication.
    pub id: String,
    /// The text content to inject.
    pub content: String,
    /// Priority determines injection order.
    pub priority: ContextPriority,
    /// When this entry was created/updated.
    pub timestamp: DateTime<Utc>,
    /// Whether this entry has been consumed (injected into a prompt).
    consumed: bool,
}

/// Collects context from various sources for injection into the system prompt.
///
/// Entries are deduplicated by `id` — registering with an existing id updates the entry.
/// On consume, entries are marked consumed and won't be re-injected unless updated.
pub struct ContextCollector {
    entries: Vec<ContextEntry>,
}

impl ContextCollector {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Register or update a context entry.
    ///
    /// If an entry with the same `id` already exists, it's updated (and unconsumed).
    pub fn register(
        &mut self,
        source: impl Into<String>,
        id: impl Into<String>,
        content: impl Into<String>,
        priority: ContextPriority,
    ) {
        let id = id.into();
        let source = source.into();
        let content = content.into();

        if let Some(existing) = self.entries.iter_mut().find(|e| e.id == id) {
            existing.content = content;
            existing.priority = priority;
            existing.timestamp = Utc::now();
            existing.consumed = false;
            debug!("updated context entry: {} ({})", id, source);
        } else {
            debug!("registered context entry: {} ({})", id, source);
            self.entries.push(ContextEntry {
                source,
                id,
                content,
                priority,
                timestamp: Utc::now(),
                consumed: false,
            });
        }
    }

    /// Consume all unconsumed entries, returning them sorted by priority (descending) then timestamp.
    ///
    /// Consumed entries won't be returned again unless updated via `register`.
    pub fn consume(&mut self) -> Vec<ContextEntry> {
        let mut result: Vec<ContextEntry> = self
            .entries
            .iter()
            .filter(|e| !e.consumed)
            .cloned()
            .collect();

        // Sort: highest priority first, then most recent first
        result.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then(b.timestamp.cmp(&a.timestamp))
        });

        // Mark as consumed
        for entry in &mut self.entries {
            entry.consumed = true;
        }

        result
    }

    /// Build a formatted string from consumed context entries.
    pub fn consume_formatted(&mut self) -> String {
        let entries = self.consume();
        if entries.is_empty() {
            return String::new();
        }

        entries
            .iter()
            .map(|e| format!("[{}] {}", e.source, e.content))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Get the number of unconsumed entries.
    pub fn pending_count(&self) -> usize {
        self.entries.iter().filter(|e| !e.consumed).count()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl Default for ContextCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_consume() {
        let mut collector = ContextCollector::new();
        collector.register("training", "epoch_5", "loss=0.42, mAP=0.67", ContextPriority::High);
        collector.register("dataset", "summary", "1200 images, 3 classes", ContextPriority::Normal);

        let entries = collector.consume();
        assert_eq!(entries.len(), 2);
        // High priority first
        assert_eq!(entries[0].source, "training");
        assert_eq!(entries[1].source, "dataset");
    }

    #[test]
    fn dedup_by_id() {
        let mut collector = ContextCollector::new();
        collector.register("training", "latest_metrics", "loss=1.0", ContextPriority::Normal);
        collector.register("training", "latest_metrics", "loss=0.5", ContextPriority::Normal);

        let entries = collector.consume();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].content, "loss=0.5");
    }

    #[test]
    fn consumed_entries_not_returned_twice() {
        let mut collector = ContextCollector::new();
        collector.register("src", "id1", "content", ContextPriority::Normal);

        assert_eq!(collector.consume().len(), 1);
        assert_eq!(collector.consume().len(), 0); // Already consumed
    }

    #[test]
    fn update_unconsumed_entry() {
        let mut collector = ContextCollector::new();
        collector.register("src", "id1", "old", ContextPriority::Normal);

        assert_eq!(collector.consume().len(), 1);

        // Update makes it unconsumed again
        collector.register("src", "id1", "new", ContextPriority::High);
        let entries = collector.consume();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].content, "new");
        assert_eq!(entries[0].priority, ContextPriority::High);
    }

    #[test]
    fn consume_formatted_output() {
        let mut collector = ContextCollector::new();
        collector.register("training", "e5", "loss=0.3", ContextPriority::High);
        collector.register("dataset", "ds", "500 images", ContextPriority::Normal);

        let formatted = collector.consume_formatted();
        assert!(formatted.contains("[training] loss=0.3"));
        assert!(formatted.contains("[dataset] 500 images"));
    }

    #[test]
    fn pending_count() {
        let mut collector = ContextCollector::new();
        assert_eq!(collector.pending_count(), 0);
        collector.register("src", "a", "content", ContextPriority::Normal);
        assert_eq!(collector.pending_count(), 1);
        collector.consume();
        assert_eq!(collector.pending_count(), 0);
    }
}
