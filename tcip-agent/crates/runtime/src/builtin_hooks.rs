use crate::hooks::{FailureHookResult, HookResult, PostHookResult, ToolHook};
use tracing::info;

/// Pre-hook: validates CUDA/GPU availability before training tools.
pub struct GpuEnvValidator;

impl ToolHook for GpuEnvValidator {
    fn pre_tool_use(&self, tool_name: &str, _input: &serde_json::Value) -> HookResult {
        if !is_training_tool(tool_name) {
            return HookResult::allow();
        }

        // Check for CUDA availability via environment
        if std::env::var("CUDA_VISIBLE_DEVICES").is_ok() || gpu_likely_available() {
            HookResult::allow()
        } else {
            let mut result = HookResult::allow();
            result.messages.push(
                "Warning: No GPU detected (CUDA_VISIBLE_DEVICES not set). \
                 Training may be very slow on CPU."
                    .to_string(),
            );
            result
        }
    }

    fn name(&self) -> &str {
        "gpu_env_validator"
    }
}

/// Post-hook: logs output path and timestamp for inference results.
pub struct ArtifactLogger;

impl ToolHook for ArtifactLogger {
    fn pre_tool_use(&self, _: &str, _: &serde_json::Value) -> HookResult {
        HookResult::allow()
    }

    fn post_tool_use(
        &self,
        tool_name: &str,
        _input: &serde_json::Value,
        output: &str,
    ) -> PostHookResult {
        if !is_inference_tool(tool_name) {
            return PostHookResult::default();
        }

        // Log artifact creation
        info!("artifact from {tool_name}: {}", truncate(output, 200));

        PostHookResult {
            feedback: vec![format!(
                "Inference artifact logged at {}",
                chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
            )],
        }
    }

    fn name(&self) -> &str {
        "artifact_logger"
    }
}

/// Pre-hook: validates that label and image file counts match before dataset loading.
pub struct DatasetIntegrityCheck;

impl ToolHook for DatasetIntegrityCheck {
    fn pre_tool_use(&self, tool_name: &str, input: &serde_json::Value) -> HookResult {
        if !is_dataset_tool(tool_name) {
            return HookResult::allow();
        }

        // Try to extract image and label paths from the input
        let images_dir = input
            .get("images_dir")
            .or_else(|| input.get("image_path"))
            .and_then(|v| v.as_str());
        let labels_dir = input
            .get("labels_dir")
            .or_else(|| input.get("label_path"))
            .and_then(|v| v.as_str());

        if let (Some(img_dir), Some(lbl_dir)) = (images_dir, labels_dir) {
            match validate_dataset_counts(img_dir, lbl_dir) {
                DatasetValidation::Ok { count } => {
                    let mut result = HookResult::allow();
                    result.messages.push(format!(
                        "Dataset integrity: {count} image-label pairs verified."
                    ));
                    result
                }
                DatasetValidation::Mismatch {
                    images,
                    labels,
                    missing_labels,
                } => {
                    let mut result = HookResult::allow();
                    result.messages.push(format!(
                        "Warning: Dataset mismatch — {images} images, {labels} labels. \
                         {missing_labels} images have no corresponding label file."
                    ));
                    result
                }
                DatasetValidation::Error(e) => {
                    let mut result = HookResult::allow();
                    result.messages.push(format!(
                        "Could not validate dataset integrity: {e}"
                    ));
                    result
                }
            }
        } else {
            HookResult::allow()
        }
    }

    fn post_tool_use_failure(
        &self,
        tool_name: &str,
        _input: &serde_json::Value,
        error: &str,
    ) -> FailureHookResult {
        if is_dataset_tool(tool_name)
            && (error.contains("no such file") || error.contains("not found"))
        {
            FailureHookResult {
                recovery_hint: Some(
                    "The dataset path does not exist. Verify the data directory structure \
                     matches the expected layout (data/images/, data/labels/detect/)."
                        .to_string(),
                ),
            }
        } else {
            FailureHookResult::default()
        }
    }

    fn name(&self) -> &str {
        "dataset_integrity_check"
    }
}

// --- Helper functions ---

fn is_training_tool(name: &str) -> bool {
    name.contains("train") || name.contains("launch_training") || name.contains("run_hpo")
}

fn is_inference_tool(name: &str) -> bool {
    name.contains("inference") || name.contains("predict")
}

fn is_dataset_tool(name: &str) -> bool {
    name.contains("dataset") || name.contains("load_data") || name.contains("load_labels")
}

fn gpu_likely_available() -> bool {
    // Simple heuristic: check if nvidia-smi exists
    #[cfg(target_os = "windows")]
    {
        std::path::Path::new(r"C:\Windows\System32\nvidia-smi.exe").exists()
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::path::Path::new("/usr/bin/nvidia-smi").exists()
    }
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        &s[..max]
    }
}

enum DatasetValidation {
    Ok { count: usize },
    Mismatch { images: usize, labels: usize, missing_labels: usize },
    Error(String),
}

fn validate_dataset_counts(images_dir: &str, labels_dir: &str) -> DatasetValidation {
    let img_path = std::path::Path::new(images_dir);
    let lbl_path = std::path::Path::new(labels_dir);

    if !img_path.is_dir() {
        return DatasetValidation::Error(format!("Images directory not found: {images_dir}"));
    }
    if !lbl_path.is_dir() {
        return DatasetValidation::Error(format!("Labels directory not found: {labels_dir}"));
    }

    let image_extensions = ["jpg", "jpeg", "png", "tif", "tiff", "bmp"];
    let images: Vec<_> = match std::fs::read_dir(img_path) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| image_extensions.contains(&ext.to_lowercase().as_str()))
            })
            .collect(),
        Err(e) => return DatasetValidation::Error(format!("Cannot read images dir: {e}")),
    };

    let labels: std::collections::HashSet<String> = match std::fs::read_dir(lbl_path) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let path = e.path();
                if path.extension().and_then(|ext| ext.to_str()) == Some("txt") {
                    path.file_stem()
                        .and_then(|s| s.to_str())
                        .map(String::from)
                } else {
                    None
                }
            })
            .collect(),
        Err(e) => return DatasetValidation::Error(format!("Cannot read labels dir: {e}")),
    };

    let image_count = images.len();
    let label_count = labels.len();
    let missing = images
        .iter()
        .filter(|img| {
            img.path()
                .file_stem()
                .and_then(|s| s.to_str())
                .is_some_and(|stem| !labels.contains(stem))
        })
        .count();

    if missing == 0 && image_count == label_count {
        DatasetValidation::Ok { count: image_count }
    } else {
        DatasetValidation::Mismatch {
            images: image_count,
            labels: label_count,
            missing_labels: missing,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn gpu_hook_allows_non_training_tools() {
        let hook = GpuEnvValidator;
        let result = hook.pre_tool_use("read_file", &json!({}));
        assert!(result.proceed);
        assert!(result.messages.is_empty());
    }

    #[test]
    fn artifact_logger_ignores_non_inference() {
        let hook = ArtifactLogger;
        let result = hook.post_tool_use("read_file", &json!({}), "contents");
        assert!(result.feedback.is_empty());
    }

    #[test]
    fn artifact_logger_logs_inference() {
        let hook = ArtifactLogger;
        let result = hook.post_tool_use("mcp__run_inference", &json!({}), "predictions saved");
        assert_eq!(result.feedback.len(), 1);
        assert!(result.feedback[0].contains("Inference artifact logged"));
    }

    #[test]
    fn dataset_hook_allows_non_dataset_tools() {
        let hook = DatasetIntegrityCheck;
        let result = hook.pre_tool_use("read_file", &json!({}));
        assert!(result.proceed);
    }

    #[test]
    fn dataset_hook_provides_recovery_hint_on_missing_path() {
        let hook = DatasetIntegrityCheck;
        let result = hook.post_tool_use_failure(
            "mcp__load_dataset",
            &json!({}),
            "no such file or directory",
        );
        assert!(result.recovery_hint.is_some());
    }
}
