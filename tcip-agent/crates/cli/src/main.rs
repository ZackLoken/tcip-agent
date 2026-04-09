mod jsonrpc_transport;
mod repl;

use tcip_api::{AnthropicClient, EchoClient};
use tcip_runtime::{
    checkpoint::StdinCheckpointResolver, config::RuntimeConfig, events::EventSink,
    ConversationRuntime, EventBus, EventEmitter, GuiEventSink, JsonLogSink, PermissionEnforcer,
    Session, SkillInjector,
};
use tcip_tools::ToolDispatcher;
use tracing::info;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    // Parse simple CLI args
    let args: Vec<String> = std::env::args().collect();
    let use_jsonrpc = args.iter().any(|a| a == "--jsonrpc");
    let workspace = if let Some(pos) = args.iter().position(|a| a == "--workspace") {
        std::path::PathBuf::from(args.get(pos + 1).expect("--workspace requires a path"))
    } else {
        std::env::current_dir()?
    };

    info!("workspace: {}", workspace.display());

    // Load config
    let config = RuntimeConfig::load(&workspace);
    info!("model: {}, permission: {}", config.model, config.permission_mode);

    // Try to create API client; fall back to offline echo client
    match AnthropicClient::new(None, None) {
        Ok(client) => {
            info!("API key found, running with Anthropic backend");
            run_with_client(client, config, workspace, use_jsonrpc).await
        }
        Err(e) => {
            info!("No API key ({e}), running in offline mode with EchoClient");
            if use_jsonrpc {
                // Notify the GUI that we're in offline mode (non-fatal)
                let notice = serde_json::json!({
                    "jsonrpc": "2.0",
                    "method": "assistant.text_done",
                    "params": {
                        "text": "⚠ Running in **offline mode** — no ANTHROPIC_API_KEY set. GUI and tools are functional, but the AI agent is disabled."
                    }
                });
                println!("{}", serde_json::to_string(&notice).unwrap_or_default());
            }
            run_with_client(EchoClient::new(), config, workspace, use_jsonrpc).await
        }
    }
}

async fn run_with_client<C: tcip_api::ApiClient>(
    api_client: C,
    config: RuntimeConfig,
    workspace: std::path::PathBuf,
    use_jsonrpc: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create tool dispatcher
    let mut tool_dispatcher = ToolDispatcher::new(workspace.clone());

    // Try to connect to MCP server if configured
    if let Some(ref mcp_command) = config.mcp_command {
        let args_refs: Vec<&str> = config.mcp_args.iter().map(String::as_str).collect();
        match tcip_tools::mcp_bridge::McpBridge::spawn(mcp_command, &args_refs) {
            Ok(bridge) => {
                tool_dispatcher.connect_mcp(bridge).await?;
                info!("MCP connected: {} tools", tool_dispatcher.mcp_tool_count());
            }
            Err(e) => {
                tracing::warn!("MCP connection failed (degraded mode): {e}");
            }
        }
    }

    // Create session
    let session_id = uuid::Uuid::new_v4().to_string();
    let mut session = Session::new(session_id);

    // Enable persistence
    let session_path = config.sessions_dir.join(format!("{}.jsonl", session.id()));
    if let Err(e) = session.enable_persistence(session_path) {
        tracing::warn!("session persistence disabled: {e}");
    }

    // Create permission enforcer
    let permission_enforcer = PermissionEnforcer::new(config.resolved_permission_mode());

    // Create skill injector
    let skill_injector = SkillInjector::new(config.skills_dir.clone());

    // Create event bus with appropriate sinks
    let mut sinks: Vec<Box<dyn EventSink>> = vec![
        Box::new(JsonLogSink::new(&workspace)),
    ];
    if use_jsonrpc {
        // In JSON-RPC mode, also push events to stdout for the GUI
        sinks.push(Box::new(GuiEventSink::new(Box::new(std::io::stdout()))));
    }
    let (event_bus, _bus_handle) = EventBus::new(sinks);
    let emitter = EventEmitter::new(event_bus.sender());

    // Create runtime
    let mut runtime = ConversationRuntime::new(
        session,
        api_client,
        tool_dispatcher,
        StdinCheckpointResolver,
        permission_enforcer,
        skill_injector,
        config.model.clone(),
        workspace,
    );
    runtime.set_event_emitter(emitter);

    // Run in appropriate mode
    if use_jsonrpc {
        jsonrpc_transport::run(runtime).await
    } else {
        repl::run(runtime).await
    }
}
