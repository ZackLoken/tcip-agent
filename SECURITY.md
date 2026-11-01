# Security and data egress

This document is for an adopter deciding whether TCIP is safe for commercially sensitive breeding
data. Local-first storage answers where data rests. It does not answer what leaves the machine when
an agent drives the platform, which is what this document covers. Read the first channel first: it
is the largest and the least bounded.

## Channel 1: the in-app agent, to the model provider

The platform's central design is an agent driving the pipeline. The in-app terminal spawns one of
three harnesses, Claude Code, Codex or Antigravity, with the workspace of breeding projects as its
working directory; the Claude Code row also adds this repository through `--add-dir`. Each row's
enforcement is the `confinement` its definition in `terminal.py`'s `PROVIDERS` states, recorded on
every launch. Every prompt, every file the agent reads, and every image it views becomes
model-provider context. Nothing in TCIP bounds, filters, or redacts what enters that context.

Some of this leaves without the agent choosing to read anything. The session-start hook injects the
active project's name into the first model turn of a new conversation (a resumed one is told
nothing), so project identity reaches the provider on that turn by construction.

This is the largest egress channel and the platform does not control it. When you let the agent
work, the breeding data it reads goes to the model provider, the same as if you pasted that data
into a chat. The provider is whoever the operator authenticated the launched harness against; the
platform holds no provider key of its own, so it inherits the operator's authentication.

Two adjacent channels widen this on the Claude Code row. Its agent is allow-listed for `WebSearch`
with no query constraint, so an agent-composed query can carry a crop, site, cultivar, or trait name
to a search provider. And neither of its shell guards carries a network rule, so the same agent can
`curl` or `wget` breeding data to any host. The fence's per-domain `WebFetch` scoping to academic
sources is real for the `WebFetch` tool and does not constrain an agent that reaches the shell.

## Channel 2: the cross-family runner, to every provider it names

`tools/cross_family_ask.py` sends its prompt to the provider of every harness it is run against;
its `BUILDERS` and their docstrings name the families and the mode each runs in, and its
`--families` default the selection. It is developer tooling rather than a platform feature, and
the platform never invokes it on project data. But an adopter who runs cross-family work ships
whatever the prompt carries to every provider it names, where the in-app terminal reaches the one
provider of the row it launched.

## Channel 3: Ray usage statistics

`run_hyperparameter_search` starts a local Ray cluster through `ray.init()` (`hpo._ray_session`).
Ray's usage-statistics page (`https://docs.ray.io/en/latest/cluster/usage-stats.html`) states that
collection "is not enabled when using local dev clusters started via `ray.init()` unless it's a
nightly wheel"; a cluster that does collect reports to `https://usage-stats.ray.io/` every hour.
The `hpo` extra installs Ray from PyPI (`ray[tune]`).

## Channel 4: inbound fetches a firewalled adopter must allow

The platform reaches out at install and first use: conda-forge, PyPI and PyTorch's wheel index
(`download.pytorch.org`) at environment creation, pretrained backbone weights from the timm and
Hugging Face hosts on first backbone build, and the two proposal engines' checkpoints (`sam` and
`grounding_dino`) from the Hugging Face host into the user-profile cache on each engine's first
call. An adopter behind a restrictive firewall needs these hosts allow-listed, or the platform will
fail at install, at first model build and at first proposal.

## What does not phone home

- The platform's own code carries no telemetry vendor, analytics, crash reporter, or update check,
  and connects outward only as Channels 3 and 4 state.
- Three HTTP clients run between platform components. The panel push from the MCP tools posts to
  the URL `web_client.backend_url` builds, the loopback address (`127.0.0.1`) at the port the
  backend recorded in the workspace's store. The session hook posts to the URL it is given, which
  every launch builds through that same function or leaves the hook to build so. The backend's
  TensorBoard proxy forwards to the loopback address (`routes/training.py`).
- The frontend contacts no external host; the GUI's annotation timing is loopback-only.
- TensorBoard binds loopback; MCP is stdio with no socket.
- The web package holds no provider SDK, so the GUI runs no direct provider API loop of its own; all
  provider contact is through the harness process the operator's terminal row launches.

## Agent transcripts sit outside TCIP retention

Your agent's session transcripts are written by its harness, outside every TCIP retention or
deletion control: Claude Code writes them to `~/.claude/projects/`, Codex under its `CODEX_HOME`
(its documentation's example is `~/.codex/history.jsonl`), and Antigravity where it chooses.
Deleting a TCIP project deletes none of its transcripts.

## The trust boundary today, and where it stops holding

The GUI binds the loopback address (`127.0.0.1`) and has no authentication. Its boundary,
`trust_boundary.py`, judges a connection by the address it arrived on: one through anything but the
loopback address or a UNIX socket is refused, always, since the GUI hands its client filesystem
reads and writes and an interactive agent terminal, which is keyboard access to the harness it
launches. The Host header must name the loopback host at the port the connection arrived on, so a
rebinding name is refused. Every WebSocket and every state-changing request (`POST`, `PUT`,
`PATCH`, `DELETE`) must carry either no Origin, as a non-browser client such as the MCP tools does,
or one naming the loopback host at any port, so a page another server on this machine serves
passes that check too. A route that requires a JSON body model refuses any other body before its
handler runs (`routes/_body_common.py`), and the backend answers no CORS preflight, which a browser
sends first for a cross-origin JSON request, so that page's browser cannot call such a route. The
TensorBoard proxy (`forward_to_board` in `routes/training.py`, for GET and POST) requires no body
model and forwards the body it receives, so its requests are admitted by the boundary's checks
alone: the arrival and Host checks, and for a POST the Origin check, which that page passes.

What the platform records about who acted is a declaration, not a verification. The MCP server
records which agent harness connected to it (the name and version the client declared in the MCP
handshake, such as `claude-code 2.1.238`), a session id it minted itself, and what the declaring
harness exported about itself to its MCP servers (Claude Code: its own session id and effective
effort; Codex exports nothing), on every audit line and statement record that process writes and as
headers on its pushes to the GUI; the model that ran is not recorded, since no harness forwards it;
the in-app terminal records which executable it launched and passes its own session id down as a
correlation any launcher could set. None of it authenticates a person: the person at the keyboard
stays a name the request declared, the breeder's confirmation routes keep taking that name from the
body, and nothing refuses on any of these values.

The whole no-authentication argument rests on the loopback bind and a single trusted machine. None
of the egress above is safe to treat as private-by-default once the platform moves off that machine,
or once the roadmap's centralized or cloud-backed storage lands: the local-first claim that carries
the trust argument stops being true the day data leaves the disk.

## Reporting a vulnerability

Report a vulnerability through GitHub's private vulnerability reporting for this repository (the
Security tab's Report a vulnerability button). If that is not enabled on this repository, open an
issue titled "security" with no details in the issue body; the maintainer will open a private
channel with you from there rather than have the details discussed in public.

Include what you would for any bug report plus what makes this one security-relevant: the affected
component or file, the egress channel or trust boundary involved if one of the above, reproduction
steps, and what you were able to observe or access as a result.

There is no bug bounty. Expect acknowledgement within 7 days of the report landing.
