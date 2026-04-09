#[cfg(test)]
mod tests {
    use serde_json::json;

    #[test]
    fn test_read_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "line1\nline2\nline3\nline4\n").unwrap();

        let input = json!({"path": file_path.to_str().unwrap()});
        let result = tcip_tools::file_ops::read_file(&input, dir.path()).unwrap();
        assert!(result.contains("line1"));
        assert!(result.contains("line4"));
    }

    #[test]
    fn test_read_file_partial() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "line1\nline2\nline3\nline4\n").unwrap();

        let input = json!({"path": file_path.to_str().unwrap(), "start_line": 2, "end_line": 3});
        let result = tcip_tools::file_ops::read_file(&input, dir.path()).unwrap();
        assert!(result.contains("line2"));
        assert!(result.contains("line3"));
        assert!(!result.contains("line1"));
    }

    #[test]
    fn test_write_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("output.txt");

        let input = json!({"path": file_path.to_str().unwrap(), "content": "hello world"});
        let result = tcip_tools::file_ops::write_file(&input, dir.path()).unwrap();
        assert!(result.contains("11 bytes"));
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "hello world");
    }

    #[test]
    fn test_edit_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("edit.txt");
        std::fs::write(&file_path, "hello world").unwrap();

        let input = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "world",
            "new_string": "TCIP"
        });
        tcip_tools::file_ops::edit_file(&input, dir.path()).unwrap();
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "hello TCIP");
    }

    #[test]
    fn test_edit_file_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("edit.txt");
        std::fs::write(&file_path, "hello world").unwrap();

        let input = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "nonexistent",
            "new_string": "replacement"
        });
        let result = tcip_tools::file_ops::edit_file(&input, dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_edit_file_ambiguous() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("edit.txt");
        std::fs::write(&file_path, "aaa bbb aaa").unwrap();

        let input = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "aaa",
            "new_string": "ccc"
        });
        let result = tcip_tools::file_ops::edit_file(&input, dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_write_creates_directories() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("sub").join("deep").join("file.txt");

        let input = json!({"path": file_path.to_str().unwrap(), "content": "nested"});
        tcip_tools::file_ops::write_file(&input, dir.path()).unwrap();
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "nested");
    }

    #[tokio::test]
    async fn test_bash_echo() {
        let dir = tempfile::tempdir().unwrap();
        let input = if cfg!(windows) {
            json!({"command": "echo hello"})
        } else {
            json!({"command": "echo hello"})
        };
        let result = tcip_tools::bash::run_bash(&input, dir.path()).await.unwrap();
        assert!(result.contains("hello"));
    }

    #[test]
    fn test_grep_search() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "foo bar baz\nhello world\n").unwrap();
        std::fs::write(dir.path().join("b.txt"), "no match here\n").unwrap();

        let input = json!({"pattern": "hello"});
        let result = tcip_tools::search::grep_search(&input, dir.path()).unwrap();
        assert!(result.contains("hello world"));
        assert!(!result.contains("no match"));
    }

    #[test]
    fn test_glob_search() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("test.rs"), "").unwrap();
        std::fs::write(dir.path().join("test.py"), "").unwrap();

        let input = json!({"pattern": "*.rs"});
        let result = tcip_tools::search::glob_search(&input, dir.path()).unwrap();
        assert!(result.contains("test.rs"));
        assert!(!result.contains("test.py"));
    }
}
