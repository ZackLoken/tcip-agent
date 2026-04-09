#[cfg(test)]
mod tests {
    use tcip_runtime::permission::{PermissionEnforcer, PermissionMode, PermissionResult};

    #[test]
    fn test_readonly_allows_reads() {
        let enforcer = PermissionEnforcer::new(PermissionMode::ReadOnly);
        assert!(matches!(enforcer.check("read_file"), PermissionResult::Allowed));
        assert!(matches!(enforcer.check("grep_search"), PermissionResult::Allowed));
        assert!(matches!(enforcer.check("glob_search"), PermissionResult::Allowed));
    }

    #[test]
    fn test_readonly_denies_writes() {
        let enforcer = PermissionEnforcer::new(PermissionMode::ReadOnly);
        assert!(matches!(enforcer.check("write_file"), PermissionResult::Denied { .. }));
        assert!(matches!(enforcer.check("edit_file"), PermissionResult::Denied { .. }));
    }

    #[test]
    fn test_workspace_write_allows_writes() {
        let enforcer = PermissionEnforcer::new(PermissionMode::WorkspaceWrite);
        assert!(matches!(enforcer.check("write_file"), PermissionResult::Allowed));
        assert!(matches!(enforcer.check("edit_file"), PermissionResult::Allowed));
        assert!(matches!(enforcer.check("read_file"), PermissionResult::Allowed));
    }

    #[test]
    fn test_workspace_write_needs_approval_for_training() {
        let enforcer = PermissionEnforcer::new(PermissionMode::WorkspaceWrite);
        assert!(matches!(
            enforcer.check("mcp__launch_training"),
            PermissionResult::NeedsApproval { .. }
        ));
        assert!(matches!(
            enforcer.check("mcp__run_hpo"),
            PermissionResult::NeedsApproval { .. }
        ));
        assert!(matches!(
            enforcer.check("bash"),
            PermissionResult::NeedsApproval { .. }
        ));
    }

    #[test]
    fn test_full_access_allows_everything() {
        let enforcer = PermissionEnforcer::new(PermissionMode::FullAccess);
        assert!(matches!(enforcer.check("read_file"), PermissionResult::Allowed));
        assert!(matches!(enforcer.check("write_file"), PermissionResult::Allowed));
        assert!(matches!(enforcer.check("bash"), PermissionResult::Allowed));
        assert!(matches!(enforcer.check("mcp__launch_training"), PermissionResult::Allowed));
        assert!(matches!(enforcer.check("mcp__run_hpo"), PermissionResult::Allowed));
    }

    #[test]
    fn test_readonly_mcp_tools_allowed_in_readonly() {
        let enforcer = PermissionEnforcer::new(PermissionMode::ReadOnly);
        assert!(matches!(enforcer.check("mcp__list_crops"), PermissionResult::Allowed));
        assert!(matches!(enforcer.check("mcp__get_crop_traits"), PermissionResult::Allowed));
        assert!(matches!(enforcer.check("mcp__get_trait_info"), PermissionResult::Allowed));
        assert!(matches!(enforcer.check("mcp__get_registry_summary"), PermissionResult::Allowed));
    }

    #[test]
    fn test_unknown_tools_default_permission() {
        let enforcer = PermissionEnforcer::new(PermissionMode::WorkspaceWrite);
        // Unknown MCP tools default to WorkspaceWrite
        assert!(matches!(enforcer.check("mcp__some_new_tool"), PermissionResult::Allowed));
        // Unknown native tools default to FullAccess → needs approval
        assert!(matches!(
            enforcer.check("unknown_tool"),
            PermissionResult::NeedsApproval { .. }
        ));
    }

    #[test]
    fn test_mode_change() {
        let mut enforcer = PermissionEnforcer::new(PermissionMode::ReadOnly);
        assert!(matches!(enforcer.check("write_file"), PermissionResult::Denied { .. }));

        enforcer.set_mode(PermissionMode::WorkspaceWrite);
        assert!(matches!(enforcer.check("write_file"), PermissionResult::Allowed));
    }
}
