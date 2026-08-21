# Security and data egress

This document is for an adopter deciding whether TCIP is safe for commercially sensitive breeding
data. Local-first storage answers where data rests. It does not answer what leaves the machine when
an agent drives the platform, which is what this document covers. Read the first channel first: it is
the largest and the least bounded.

## Channel 1: the in-app agent, to the model provider

The platform's central design is an agent driving the pipeline. The in-app terminal spawns a real
Claude Code process granted the workspace of breeding projects through `--add-dir`, behind a
permission fence that denies edits outside a narrow allowlist and denies a set of shell write verbs,
but places no restriction on reading. Every prompt, every file the agent reads, and every image it
views becomes model-provider context. Nothing in TCIP bounds, filters, or redacts what enters that
context.

Some of this leaves without the agent choosing to read anything. The session-start hook injects the
active project's name into the first model turn, so project identity reaches the provider on turn one
by construction.

This is the largest egress channel and the platform does not control it. It is not a defect in the
usual sense; it is the direct consequence of the platform's premise that the agent is the operator.
But an adopter must be told plainly: when you let the agent work, the breeding data it reads goes to
the model provider, the same as if you pasted that data into a chat. The provider is whoever the
operator authenticated Claude Code against; the platform holds no provider key of its own, so it
inherits the operator's authentication.

Two adjacent channels widen this. The fenced agent is allow-listed for `WebSearch` with no query
constraint, so an agent-composed query can carry a crop, site, cultivar, or trait name to a search
provider. And neither shell guard carries a network rule, so the same agent can `curl` or `wget`
breeding data to any host. The fence's per-domain `WebFetch` scoping to academic sources is real for
the `WebFetch` tool and does not constrain an agent that reaches the shell.

## Channel 2: the cross-family runner, to two more providers

`scripts/cross_family_ask.py` sends its prompt to Codex (OpenAI) and Antigravity (Google) on every
invocation, in read-only mode. It is developer tooling rather than a platform feature, and the
platform never invokes it on project data. But an adopter who runs cross-family work ships whatever
the prompt carries to two additional providers. Shipped code reaches three providers in total:
Anthropic by default, and OpenAI plus Google through this developer script.

## Channel 3: Ray usage statistics, phoning home by default

`run_hpo` starts a Ray cluster whose usage-statistics reporter is enabled by default and posts
periodically to `https://usage-stats.ray.io/`. The payload is machine and cluster metadata, not
project data. It is the platform's one genuine phone-home, and it is disabled with a single
environment variable, `RAY_USAGE_STATS_ENABLED=0`, which the platform does not set for you. An
adopter running hyperparameter optimization behind a firewall must either allow that host or set that
variable.

## Channel 4: inbound fetches a firewalled adopter must allow

The platform reaches out at install and first use: conda and PyPI at environment creation, the SAM-2
source from GitHub, and pretrained backbone weights from the timm and Hugging Face hosts on first
backbone build. SAM-2 checkpoint weights are not fetched automatically; the wrapper raises with a
manual-download instruction, which is adopter-initiated rather than an egress. An adopter behind a
restrictive firewall needs these hosts allow-listed, or the platform will fail at install and at
first model build.

## What does not phone home

An adopter needs the assurances as much as the warnings, and these were checked, not assumed:

- The platform's own code carries no telemetry vendor, analytics, crash reporter, or update check.
- The only HTTP client between platform components is the panel push from the MCP tools to the web
  backend, which defaults to loopback (`127.0.0.1`). Its host is configurable, so repointing it sends
  those panel events wherever it is pointed.
- The frontend contacts no external host; the GUI's annotation timing is loopback-only.
- TensorBoard and the Ray dashboard bind loopback; MCP is stdio with no socket.
- The web package holds no provider SDK, so the GUI runs no direct provider API loop of its own; all
  provider contact is through the operator's own Claude Code process.

## Agent transcripts sit outside TCIP retention

Your agent's session transcripts are written by Claude Code to `~/.claude/projects/`, outside every
TCIP retention or deletion control. Deleting a TCIP project deletes none of its transcripts.

## The trust boundary today, and where it stops holding

The GUI binds to loopback (`127.0.0.1`) by default and is frictionless with no authentication. A
connection is judged by the address it arrived on, not by the configured bind: one from this machine
is served, one through a network address is refused until the operator sets
`TCIP_WEB_ALLOW_INSECURE=1`, because an exposed GUI hands an unauthenticated network client
filesystem reads and writes and an interactive agent terminal, which is keyboard access to Claude
Code. A cross-site page cannot open a WebSocket and read GUI state (the Origin must be the request's
own origin), and DNS rebinding is blocked because the Host header must name this backend as reached
(its arrival address, its own hostname, or a name the operator advertised under the opt-in; there is
no wildcard). A reverse proxy in front of the backend is an exposed deployment: the backend cannot
see the proxied client, so it is served only under the opt-in with the proxy's name advertised.
Token authentication for an intentionally exposed GUI is a planned follow-on.

The whole no-authentication argument rests on the loopback bind and a single trusted machine. None of
the egress above is safe to treat as private-by-default once the platform moves off that machine, or
once the roadmap's centralized or cloud-backed storage lands: the local-first claim that carries the
trust argument stops being true the day data leaves the disk.

## In one paragraph

When you let the agent drive TCIP, the breeding data it reads is sent to your model provider, the same
as pasting it into a chat; the platform does not redact or bound this. Your agent's session
transcripts are written outside every TCIP retention control. Hyperparameter optimization phones home
to Ray by default unless you set `RAY_USAGE_STATS_ENABLED=0`. The platform itself carries no telemetry
and makes no other outbound connection beyond fetching dependencies and model weights at install and
first use.
