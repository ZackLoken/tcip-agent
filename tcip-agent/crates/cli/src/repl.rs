use std::io::{self, BufRead, Write};
use tcip_api::ApiClient;
use tcip_runtime::checkpoint::CheckpointResolver;
use tcip_runtime::skills::SubagentMode;
use tcip_runtime::ConversationRuntime;
use tcip_tools::ToolExecutor;

/// Run the interactive REPL loop.
pub async fn run<C, T, R>(
    mut runtime: ConversationRuntime<C, T, R>,
) -> Result<(), Box<dyn std::error::Error>>
where
    C: ApiClient,
    T: ToolExecutor,
    R: CheckpointResolver,
{
    let stdin = io::stdin();
    let mut reader = stdin.lock();

    println!("TCIP Agent v0.1.0");
    println!("Type /help for commands, /quit to exit.\n");

    loop {
        print!("you> ");
        io::stdout().flush()?;

        let mut input = String::new();
        if reader.read_line(&mut input)? == 0 {
            break; // EOF
        }

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Handle slash commands
        if let Some(cmd) = trimmed.strip_prefix('/') {
            match handle_command(cmd, &mut runtime) {
                CommandResult::Continue => continue,
                CommandResult::Quit => break,
            }
        }

        // Run conversation turn
        match runtime.run_turn(trimmed).await {
            Ok(summary) => {
                if !summary.assistant_text.is_empty() {
                    println!("\nagent> {}", summary.assistant_text);
                }
                for tc in &summary.tool_calls {
                    let status = if tc.is_error { "ERROR" } else { "OK" };
                    println!("  [tool: {} → {status}]", tc.name);
                }
                let cost = runtime.usage().estimated_cost_usd(runtime.model());
                println!(
                    "  [{} input + {} output tokens, ~${:.4}]\n",
                    summary.input_tokens, summary.output_tokens, cost
                );
            }
            Err(e) => {
                eprintln!("Error: {e}\n");
            }
        }
    }

    println!("Session saved. Goodbye.");
    Ok(())
}

enum CommandResult {
    Continue,
    Quit,
}

fn handle_command<C, T, R>(
    cmd: &str,
    runtime: &mut ConversationRuntime<C, T, R>,
) -> CommandResult
where
    C: ApiClient,
    T: ToolExecutor,
    R: CheckpointResolver,
{
    let parts: Vec<&str> = cmd.splitn(2, ' ').collect();
    match parts[0] {
        "quit" | "exit" | "q" => {
            return CommandResult::Quit;
        }
        "help" | "h" => {
            println!("Commands:");
            println!("  /help           Show this help");
            println!("  /quit           Exit the agent");
            println!("  /mode <name>    Switch subagent mode");
            println!("  /fork <name>    Fork session into a named branch");
            println!("  /interview      Start/check deep interview");
            println!("  /workflow       Show workflow status");
            println!("  /skills         List learned skills");
            println!("  /status         Show session status");
            println!("  /cost           Show usage & cost");
        }
        "mode" => {
            if let Some(mode_name) = parts.get(1) {
                match *mode_name {
                    "PipelineDesigner" | "pipeline" => {
                        runtime.switch_mode(SubagentMode::PipelineDesigner);
                        println!("Switched to PipelineDesigner mode.");
                    }
                    "CodeGenerator" | "code" => {
                        runtime.switch_mode(SubagentMode::CodeGenerator);
                        println!("Switched to CodeGenerator mode.");
                    }
                    "TrainingOrchestrator" | "training" => {
                        runtime.switch_mode(SubagentMode::TrainingOrchestrator);
                        println!("Switched to TrainingOrchestrator mode.");
                    }
                    "ResultsAnalyzer" | "results" => {
                        runtime.switch_mode(SubagentMode::ResultsAnalyzer);
                        println!("Switched to ResultsAnalyzer mode.");
                    }
                    _ => println!("Unknown mode: {mode_name}. Options: PipelineDesigner, CodeGenerator, TrainingOrchestrator, ResultsAnalyzer"),
                }
            } else {
                println!("Current mode: {}", runtime.mode());
            }
        }
        "status" => {
            let usage = runtime.usage();
            println!("Session: {}", runtime.session.id());
            if let Some(parent) = runtime.session.parent_session_id() {
                println!("  forked from: {parent}");
            }
            if let Some(branch) = runtime.session.branch_name() {
                println!("  branch: {branch}");
            }
            println!("Messages: {}", runtime.session.message_count());
            println!("Mode: {}", runtime.mode());
            println!("Turns: {}", usage.turn_count);
        }
        "fork" => {
            if let Some(branch_name) = parts.get(1) {
                let forked = runtime.session.fork(branch_name);
                println!(
                    "Forked session → {} (branch: {branch_name})",
                    forked.id()
                );
                println!("Note: forked session is detached. Continuing on the original session.");
                // In the future, the forked session could be stored / switched to.
            } else {
                println!("Usage: /fork <branch_name>");
            }
        }
        "cost" => {
            let usage = runtime.usage();
            println!(
                "Input tokens: {}, Output tokens: {}",
                usage.total_input_tokens, usage.total_output_tokens
            );
            println!(
                "Estimated cost: ${:.4}",
                usage.estimated_cost_usd("claude-sonnet-4-20250514")
            );
        }
        "interview" => {
            if let Some(interview) = runtime.interview() {
                println!("{}", interview.status_line());
            } else {
                runtime.start_interview();
                println!("Interview started. Ask a question or type your request.");
            }
        }
        "workflow" => {
            if let Some(wf) = runtime.workflow() {
                println!("{}", wf.status_summary());
            } else {
                println!("No active workflow. The agent will create one when needed.");
            }
        }
        "skills" => {
            if let Some(sub) = parts.get(1) {
                if sub.starts_with("forget ") || *sub == "forget" {
                    if let Some(name) = sub.strip_prefix("forget ").or_else(|| parts.get(2).copied()) {
                        if runtime.learner_mut().store.forget(name) {
                            println!("Forgot skill: {name}");
                        } else {
                            println!("Skill not found: {name}");
                        }
                    } else {
                        println!("Usage: /skills forget <name>");
                    }
                } else {
                    println!("Unknown subcommand. Options: /skills, /skills forget <name>");
                }
            } else {
                let skills = runtime.learner_mut().store.skills();
                if skills.is_empty() {
                    println!("No learned skills yet.");
                } else {
                    println!("Learned skills ({}):", skills.len());
                    for s in skills {
                        println!("  {} (confidence: {:.0}%, uses: {})", s.name, s.confidence * 100.0, s.use_count);
                    }
                }
            }
        }
        _ => println!("Unknown command: /{cmd}. Type /help for available commands."),
    }
    CommandResult::Continue
}
