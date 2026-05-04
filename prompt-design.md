# Designing an Ambient Pair-Programmer for Claude Code

## Executive Summary

You want a sidecar process that watches a live Claude Code (CC) session plus the developer's working directory and emits role-specific commentary (architect, debugger, security reviewer, teacher) without being asked. The good news: CC's session state is fully observable from disk, hooks give you push-style triggers with full context (`session_id`, `transcript_path`, `cwd`), and MCP recently gained channel-style server-pushed messages, so multiple ingress strategies are now viable. The hard problems are not infrastructure — they are (a) prompt design that yields *useful, non-obvious, often-silent* commentary, (b) repetition suppression so the advisor does not turn into a Greek chorus, and (c) calibration — knowing when to stay quiet.

The recommended architecture is a Python CLI sidecar with three ingress channels (CC hooks → event bus, JSONL tailer for completeness, filesystem watcher), a debounced event loop with a router that selects role(s), per-role LLM calls with strict structured output and an explicit `silent` decision, and a SQLite + small vector index memory store. An optional MCP server exposes "ask the advisor" as a pull-mode tool. Build the MVP in three days; treat MCP and TUI as v2.

---

## 1. Claude Code Session Ingress

The three ingress paths are not mutually exclusive. The most robust setup uses **hooks as the trigger** and the **JSONL transcript as the source of truth for context**, with MCP reserved for pull-mode "ask the advisor" interactions.

### 1.1 JSONL transcript files

**Location.** On macOS and Linux, CC writes one JSONL file per session at:

```
~/.claude/projects/<url-encoded-cwd>/<session-uuid>.jsonl
```

The directory name is the URL-encoded form of the project's absolute path (slashes become hyphens, e.g. `-Users-schacon-projects-testing`). There is also `~/.claude/history.jsonl` (a flat log of every prompt across all sessions, useful for global stats but not for reconstructing a single session) and `~/.claude/file-history/<session-id>/` snapshots used by `/rewind`.

**Line schema.** Each line is one JSON object linked by UUID. Common types you must handle:

| `type` | Meaning | Key fields |
|---|---|---|
| `user` | User prompt or tool result | `uuid`, `parentUuid`, `message.role`, `message.content` (string or array of blocks; `tool_result` blocks carry `tool_use_id`) |
| `assistant` | Model turn | `uuid`, `parentUuid`, `message.content[]` blocks (`text`, `thinking`, `tool_use`), `message.usage` (token counts), `message.model` |
| `system` | System notices / prompts | `uuid`, `content` |
| `summary` | Auto-generated rollup | `summary`, `leafUuid` |
| `file-history-snapshot` | Snapshot reference written on resume | `messageId`, `snapshot` |
| `compact_boundary` | Marker dropped before/after compaction | both inside the parent file and copied as the first record of the continuation file |

Every record carries `sessionId`, `timestamp`, `uuid`, `cwd`, `gitBranch`, and `parentUuid`. **Sub-agents (the `Task` tool) get their own JSONL files** with `isSidechain: true` records; they are linked to the parent via `parentToolUseId` / `agentId`. To reconstruct a full conversation programmatically you walk `parentUuid` backward from the last leaf, optionally crossing `compact_boundary` and inter-file resume boundaries (the new file's first non-prefix record's `parentUuid` points at the old file's tail).

**Operational gotchas (these are real, not theoretical).** Several are documented Anthropic issues:

- *Phantom `parentUuid` references.* CC sometimes writes records whose `parentUuid` references a UUID that was never written, breaking the resume chain (issue #22526). Treat any orphan as "stop here, render what's reachable" and log a diagnostic.
- *`file-history-snapshot.messageId` collisions.* On resume, snapshots reuse message UUIDs as their `messageId`, so naive UUID indexing jumps to a snapshot instead of a real message and orphans the tail (issue #36583). Index by `(uuid, type)` or skip snapshot records when building the chain.
- *Silent post-`cd` write failures.* After `cd` to another directory inside a session, assistant responses sometimes stop persisting (issue #22566). Don't rely on JSONL being a 100% complete record of what was visible.
- *`isSidechain` is rarely set on rewind branches.* You cannot trust it as the only filter (issue #24471).
- *Compaction creates a new root with `parentUuid: null`.* Don't assume one file = one chain; a long session may contain two or three disjoint trees joined via `compact_boundary`.
- *Append behavior is mostly append-only but not strictly so.* The CC team's own architecture analysis describes session storage as "mostly append-only," and earlier entries can be rewritten on resume. Don't compute hashes of "lines you have seen" — index by `uuid`.
- *Unescaped control characters in hook stdin.* Issue #53463 documents that CC sometimes sends raw `U+0000–U+001F` bytes inside JSON strings, breaking strict parsers like `jq`. Use a parser with lenient mode or strip control chars before parsing.
- *Multi-session concurrency.* Two CC instances may run in the same `cwd` at once; the project directory will then contain two active JSONLs. Distinguish by `session_id`.

**Tailing.** Files grow append-mostly, so the standard pattern works: open in binary, seek to end (or to a stored offset), poll/notify on size change, read the new bytes, split on `\n`, parse each line with try/except (a partial trailing line is normal — buffer it). Use `watchdog` (Python) with the OS-native observer (`FSEvents` on macOS, `inotify` on Linux). Polling fallback is fine here because a single file at known path is cheap to `stat`. **Do not** use file hashes to detect "rewrites" — CC's own format guarantees keep `uuid` stable, so dedupe by UUID instead.

```python
# transcript_tailer.py — minimal robust tailer
import json, time, os
from pathlib import Path

class JSONLTailer:
    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.seen_uuids: set[str] = set()
        self._buffer = b""

    def poll(self) -> list[dict]:
        if not self.path.exists():
            return []
        size = self.path.stat().st_size
        if size < self.offset:           # truncation / replacement
            self.offset = 0
            self._buffer = b""
        if size == self.offset:
            return []
        with self.path.open("rb") as f:
            f.seek(self.offset)
            chunk = f.read()
            self.offset = f.tell()
        self._buffer += chunk
        *complete, self._buffer = self._buffer.split(b"\n")
        out = []
        for line in complete:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue                  # stale partial line; will be re-read next time if needed
            uid = rec.get("uuid")
            if uid and uid in self.seen_uuids:
                continue
            if uid:
                self.seen_uuids.add(uid)
            out.append(rec)
        return out
```

### 1.2 Hooks

CC's hooks system is the cleanest *trigger* mechanism. Hooks fire at specific lifecycle events; CC writes a JSON envelope to your hook's stdin (or POSTs it to an HTTP endpoint, or calls an MCP tool, or runs a sub-prompt). The envelope always includes `session_id`, `transcript_path`, `cwd`, and `hook_event_name`, plus event-specific fields.

The events relevant to an ambient advisor:

- `UserPromptSubmit` — fires before CC processes the user prompt. You can return `additionalContext` via stdout to inject text into CC's context. Useful for "the advisor noticed X" injections, but be wary of the echo-chamber risk (§8).
- `PreToolUse` / `PostToolUse` — every tool call. `PostToolUse` includes `tool_input` and `tool_response`. This is where most "what just happened" signal lives.
- `Stop` — main agent finished its reply (not a user interrupt). Best place to fire a "review the just-completed turn" advisor pass.
- `SubagentStop` — a `Task` sub-agent finished.
- `Notification` — permission-request idle, etc. Useful as a prod for "user has been deciding for a while."
- `SessionStart` / `SessionEnd` — boot-up and teardown. Use for memory load/persist.
- `PreCompact` — fires before context compaction. Last chance to snapshot transcript before history is summarized.

**Capabilities.** Command hooks run arbitrary shell, get JSON on stdin, communicate decisions through exit codes and a stdout JSON envelope. Exit code 2 blocks the action (PreToolUse) or forces continuation (Stop). Hooks can return `{"hookSpecificOutput": {"additionalContext": "..."}}` (UserPromptSubmit, PreToolUse v2.1.9+) to inject context, `{"continue": false}` to halt, or `{"suppressOutput": true}` to hide. The `async: true` flag (Jan 2026) lets a hook fire-and-forget without blocking CC. There are also HTTP hooks (POST), prompt hooks (single-turn LLM evaluation), MCP-tool hooks, and agent hooks (subagent verification).

**Latency.** Hooks block the agent loop unless `async: true`. Default timeout is 60 s (configurable). Multiple hooks for the same event run in parallel. For an ambient advisor you almost always want `async: true` because the advisor's commentary is non-blocking — it's a side channel, not a gate.

**Security.** Hooks run with the user's full privileges. Don't store secrets in hook scripts; use `allowedEnvVars`. The unescaped-control-character bug means strict JSON parsers can fail silently and skip your security checks — wrap every hook in defensive parsing.

**Triggering external commentary.** A hook is the simplest way to wake the advisor. The pattern:

```jsonc
// .claude/settings.json
{
  "hooks": {
    "Stop": [
      { "matcher": "", "hooks": [
        { "type": "command", "command": "advisor notify --event stop", "async": true }
      ] }
    ],
    "PostToolUse": [
      { "matcher": "Edit|Write|MultiEdit", "hooks": [
        { "type": "command", "command": "advisor notify --event post_tool_use", "async": true }
      ] }
    ],
    "SessionStart": [
      { "matcher": "", "hooks": [
        { "type": "command", "command": "advisor session-start", "async": true }
      ] }
    ]
  }
}
```

The hook just enqueues an event into your sidecar's bus (Unix-domain socket, named pipe, or HTTP). All the heavy work — reading the transcript, building the prompt, calling the LLM — happens in the sidecar.

### 1.3 MCP

MCP is the most architecturally elegant ingress and the worst fit for "ambient." The protocol is fundamentally request/response: the client (CC) calls tools on the server. An ambient advisor is the inverse — the server wants to push to the user.

There are three relevant primitives, each with caveats:

- **Tools** (pull). Standard. CC calls `advisor.review_recent_turn` when CC decides it needs review. Works today. The interaction is initiated by the model, not the human, so the model must be coached (via system prompt or `CLAUDE.md`) to invoke it.
- **Resources with subscribe** (pull-with-notify). CC reads `advisor://commentary/latest`; the server can notify on change. Good for status-line-style surfaces but the model only sees the resource if it requests it.
- **Sampling** (server-initiated LLM call). The server asks the client to run an LLM completion on the server's behalf, billed to the user's subscription. Conceptually perfect for "advisor wants to use the user's Claude subscription rather than its own API key." **As of late 2025, Claude Code does NOT implement MCP sampling as a client** (issue #1785 is open). Cursor and a few other clients do. Don't depend on this in CC.
- **Channels (`claude/channel` capability).** Newer addition documented in CC's MCP docs: an MCP server can push messages directly into the session so Claude can react to external events (CI results, alerts). Enabled by `--channels` at startup. This is the closest thing to a true push primitive. It's how a future-proof "ambient" MCP server would surface commentary into the session, but the surface area is still limited (the message becomes context for the model, not a user-visible UI element).

**Bottom line on MCP for this use case.** Use MCP for *pull* — a "review my last action" tool the user or model can invoke explicitly. Don't try to replace the hook+sidecar push path with MCP today.

### 1.4 Comparison

| Property | JSONL tail | Hooks | MCP tools | MCP channels |
|---|---|---|---|---|
| Robust to CC version bumps | Medium (schema is undocumented) | High (documented) | High | Medium (newer API) |
| Latency to event | 50ms–1s | <50 ms | Pull only | <1 s push |
| Completeness of context | Highest (sees thinking, tool I/O, sub-agents) | High (event payload + transcript path) | Limited to what tool params expose | Limited |
| Can inject back into session | No | Yes (`additionalContext` in UserPromptSubmit/PreToolUse, `decision: block` reasons) | Yes (tool response) | Yes (channel message) |
| Engineering cost | Low | Low | Medium | Medium |
| Survives session resume | Yes | Yes | Yes | Yes |
| Sees compaction-discarded history | Yes (via JSONL on disk, with caveats) | Yes (via `transcript_path`) | No | No |

**Recommended combination for the ambient watcher:** **hooks** as the wake-up trigger + **JSONL tail** as the context source + **MCP tool** (optional v2) for explicit pull. Hooks alone don't give you the deep transcript history you need; JSONL tail alone doesn't tell you when CC has just finished.

---

## 2. CWD File-Change Ingress

### 2.1 Library choice

For a Python sidecar, `watchdog` is the right default. It dispatches to `inotify` on Linux, `FSEvents` on macOS, `ReadDirectoryChangesW` on Windows, and falls back to polling. Tradeoffs vs. alternatives:

- **`watchdog` (Python)** — mature, OS-native, supports `PatternMatchingEventHandler`. Caveat: kqueue (BSD/older macOS path) consumes a file descriptor per file; not great in deeply nested trees.
- **Node `chokidar`** — best-in-class for Electron/IDE-grade watching, glob support, atomic-write handling baked in. Use only if your sidecar is Node.
- **Rust `notify`** — lowest latency, lowest resource use, best for background daemons. Overkill for a sidecar; consider if you intend to ship as a static binary.
- **OS-native directly (`inotify`/`FSEvents`)** — only worth it if you need single-digit ms latency. Watchdog wraps these well.

For a CC ambient advisor at developer pace, watchdog with the default observer is more than enough.

### 2.2 Editor save patterns

Modern editors do *not* simply write the file. Common patterns you must accommodate:

- **Atomic rename** (vim, IntelliJ): write to `.foo.swp` or `foo~`, then `rename` over the original. Watchdog reports a `Created` for the new file and `Moved` for the rename, often without a `Modified`. Subscribe to all four event types and de-dupe on path.
- **Swap files / lock files** (`.foo.swp`, `.foo.swo`, `~$foo.docx`, `.#foo`): noise. Filter by name pattern.
- **VS Code save with hot exit / auto-save** can produce a flurry of writes per keystroke; debounce hard.
- **`.git/` internal churn**: `.git/index.lock`, `.git/HEAD`, `.git/objects/...` change constantly during commits, rebases, fetches. **Always exclude `.git/`**.
- **`node_modules/`, `.venv/`, `.tox/`, `target/`, `build/`, `dist/`, `__pycache__/`, `.next/`, `.nuxt/`, `.gradle/`, `.idea/`, `.vscode/`**: exclude.
- **OS noise**: `.DS_Store`, `Thumbs.db`, `desktop.ini`.

### 2.3 Filtering: gitignore + global denylist

The right default is "respect the project's `.gitignore` plus a built-in denylist of the directories above plus a configurable allowlist of file extensions." A clean implementation uses `pathspec` (Python) to parse `.gitignore` semantics correctly (it handles `!` un-ignores, double-`**` etc.):

```python
import pathspec
spec = pathspec.PathSpec.from_lines(
    "gitwildmatch", open(".gitignore").read().splitlines() + EXTRA_IGNORES)
def is_relevant(path: str) -> bool:
    return not spec.match_file(path)
```

Always also exclude obvious secrets (`.env*`, `*.pem`, `*.key`, `id_rsa*`, `*.p12`, things matching `**/credentials*`).

### 2.4 Debouncing

Editor saves and tools like `prettier --write` can produce many events in <100 ms. Debounce per-path with a quiet window (500 ms–2 s for our use case; we are not running tests, so we can afford to wait):

```python
class DebouncedDispatcher:
    def __init__(self, quiet_ms=1500, sink=lambda batch: None):
        self.quiet = quiet_ms / 1000
        self.sink = sink
        self.pending: dict[str, float] = {}     # path -> last_event_time
        self.timer = None

    def on_event(self, path):
        self.pending[path] = time.monotonic()
        if self.timer is None or not self.timer.is_alive():
            self.timer = threading.Timer(self.quiet, self._flush)
            self.timer.start()

    def _flush(self):
        now = time.monotonic()
        ready = [p for p, t in self.pending.items() if now - t >= self.quiet]
        for p in ready: del self.pending[p]
        if ready: self.sink(ready)
        if self.pending:    # reschedule for the rest
            delay = self.quiet - (now - min(self.pending.values()))
            self.timer = threading.Timer(max(delay, 0.05), self._flush)
            self.timer.start()
```

### 2.5 Capturing semantic deltas

"Knowing a file changed" is far less useful than "knowing *what* changed." Two practical options:

1. **Git as oracle.** If the project is a git repo, run `git diff --no-color HEAD -- <path>` (or `git diff` against the index). Free, semantic, ignores binary noise, and the developer probably already has the right `.gitignore`. For files newly created since HEAD, `git diff --no-index /dev/null <path>` works. Cache the diffs keyed by `(path, content_hash)`.
2. **Shadow copies.** Maintain a hidden mirror under `~/.cache/advisor/<project>/` and run `difflib.unified_diff` on changes. Use this only when git is unavailable.

`git diff` is strongly preferred — it gives you the exact mental model the user has. For new files, prefer line-numbered content over a "whole file diff" so the LLM gets file:line precision.

### 2.6 Edge: huge bursts

`npm install`, `pip install`, `cargo build`, `git checkout` can change tens of thousands of files in seconds. Detect this with a count threshold (e.g. >50 changed files in one debounce window) and degrade gracefully: skip the LLM call, just log "bulk change in `node_modules/`, ignored."

---

## 3. Existing Tools and Prior Art

The ambient pair-programmer concept sits at the intersection of three tool categories. None of them does exactly what you're describing, but each contributes patterns.

### 3.1 Coding agents and their watch modes

- **Aider's `--watch`** is the most direct prior art for "passively observe and act." Aider polls files for an `# AI` or `# AI!` magic comment; when it appears, Aider does the change. It is event-driven on file save but *triggered by a comment*, not by your activity in another tool. Open source (Apache 2.0). Single-LLM-call architecture; uses git as the source of truth for state.
- **Cursor's background agents** (Cursor 2.0, Feb 2026) run on cloud VMs in git worktrees, parallel to your foreground work. They are autonomous task executors, not commentators — they make PRs, not suggestions. Closed source.
- **Cline** (open-source, Apache 2.0, VS Code extension) has Plan & Act mode and step-by-step approval. Not ambient — fully synchronous and user-initiated.
- **Continue.dev** (open-source) is a chat/autocomplete extension with custom slash commands; ambient features limited to autocomplete.
- **Zed's agent panel** is conversational; no ambient critic mode.
- **Claude Code itself** has sub-agents and Stop hooks, the very primitives you'll use.
- **Goose** (Block, MCP-based, ~41k GitHub stars) — open-source, general-purpose; not coding-specific but easy to extend.

### 3.2 PR-time code review bots

- **CodeRabbit** — PR-scoped, lower-noise, ~82% bug-detection rate in benchmarks. Multi-agent under the hood (different agent personas).
- **Greptile** — whole-repo indexing first, then review; lower hit rate (~44%) but higher cross-file awareness; v3 (Sep 2025) is "complete rewrite."
- **Graphite Reviewer**, **Sweep**, **Sourcegraph Cody** — all PR-time, none ambient.
- **Sentry AI auto-fix** — runtime-error-driven, not source-driven.

These are *role-based pipelines* internally. Greptile's marketing explicitly describes "parallel agents [that] review changes." That validates the multi-role architecture you want, but operates at PR cadence (minutes), not keystroke cadence (seconds).

### 3.3 Claude Code transcript readers and observability

- **`claude-trace`** (badlogic/lemmy) — injects a `fetch()` interceptor into CC and logs every Anthropic API request to a JSONL in `.claude-trace/`. This is *more* complete than the on-disk JSONL because it captures system prompts and raw streamed responses. Open source, MIT.
- **`ccusage`** — CLI that reads `~/.claude/projects/**/*.jsonl` and reports token usage and cost. Plain JSONL parser; useful as reference code for the schema.
- **`claude-code-log`** (daaain) and **`simonw/claude-code-transcripts`** — JSONL → HTML viewers. Reference implementations of correct schema handling.
- **`claude-code-hooks-multi-agent-observability`** (disler) — real-time hook event tracking → web dashboard. The closest existing thing to "watch the session"; differs from your goal in that it shows what happened, not what *should* happen.
- **`claude-code-otel`** (ColeMurray) and **`claude_telemetry`** (TechNickAI's "claudia" wrapper) — OpenTelemetry-based; export to Logfire, Sentry, Honeycomb, Datadog. Use these for telemetry and cost analytics, not for advisor state.
- **`Claude-Code-Usage-Monitor`** (Maciek-roboblog) — live TUI dashboard reading the JSONL.

### 3.4 LLM-as-critic / LLM-as-reviewer literature

Highlights you can borrow techniques from:

- **Self-Refine** (Madaan et al. 2023) and **Reflexion** (Shinn et al. 2023) — same model criticises its own output and revises. Useful pattern but in the original setup the critic and the actor share weights, leading to consistency bias. The Pn-S literature ("Probing Negative Samples") and follow-up work shows naive self-critique often degrades performance unless the critic has an external signal (test runner, search results).
- **CRITIC** (Gou et al. 2023) — LLM uses tools to verify claims before critiquing. Good template: in our context the "tools" are `git diff`, `grep`, file-read.
- **Constitutional AI** (Bai et al. 2022) — explicit principles guide the critic. Useful when you want each role to have a stable set of values.
- **Multi-agent debate** (Du et al. 2023; surveys 2024–2025) — multiple agents argue, a judge picks. Good for adversarial review; expensive and noisy for ambient use.
- **MAGICORE / Process Reward Models** — finding from Lanham (2026 essay): "spend your budget on the critic signal, not the loop"; a learned PRM beats LLM-self-critique even when the loop is identical. For our purposes, the practical translation: rely on cheap external signals (test failures, lint output, type-checker output, existing diff structure) rather than asking another LLM to critique the first.
- **"Three explorers + one critic"** (Claude Code Ultra plan and MoA literature, Together AI). Validates the pattern of *separating* critic agents from generators rather than sharing a context window.
- Kang et al.'s work on detecting prompt knowledge gaps in developer-LLM conversations identifies four recurrent gap classes (Missing Context, Missing Specifications, Multiple Context, Unclear Instructions) — directly useful as taxonomy for the "teacher" role.

### 3.5 Ambient-agent prior art

- **LangGraph's email assistant** (LangChain's reference ambient agent) introduced the **notify / question / review** taxonomy of human-in-the-loop interactions. This is the right mental model for your advisor:
  - *Notify* — "FYI, you didn't add a test for the new error path in `auth.py:42`." No action needed.
  - *Question* — "You're about to delete `migrations/0034_*.py`. Are you sure no environment runs it yet?"
  - *Review* — never used by an advisor (it doesn't *take* actions).
- **Drasi (Microsoft)** — change-driven query architecture for ambient agents: continuous queries fire only on meaningful change patterns. The lesson: build your trigger as *change detection*, not polling, and keep most events from reaching the LLM.

**None** of the surveyed tools combines: live LLM-transcript ingestion + filesystem watching + multi-role commentary + silent-by-default. This is a real gap.

---

## 4. Multi-Role Architecture

### 4.1 Useful roles

The four you listed are the right starting set. Other roles worth considering, in rough priority for an MVP:

| Role | Primary signal | Common output | When relevant |
|---|---|---|---|
| **Architect** | Diff structure, abstractions, coupling | "This module now has 5 responsibilities; consider splitting" | New file added; major refactor; >2 files in one diff |
| **Debugger** | Error paths, missing branches | "What happens if `user_id` is `None` here?" | Logic added; new conditional; exception thrown in tool output |
| **Security reviewer** | Secrets, injection, auth, deps | "`subprocess.run(cmd, shell=True)` with user input on `:42`" | Touching auth, network, IO, eval, subprocess, SQL |
| **Teacher** | Concepts the user is touching | "You're using `asyncio.gather`; note exceptions are wrapped" | New API, unfamiliar idiom, repeated similar mistake |
| **Tester / QA** | Missing tests, fragile mocks | "No test covers the `403` path you just added" | Production code change without matching test |
| **Performance** | Big-O, allocations, queries | "`for u in users: db.get(u.id)` is N+1" | Loops over data; DB calls in loops; large data structures |
| **DevOps / SRE** | Deploy and ops surfaces | "This config change isn't reflected in `helm/values.yaml`" | Touching Dockerfile, k8s manifests, CI |
| **Dependency manager** | Supply chain, version drift | "Adding `left-pad@1.0.0` — last published 2014" | `package.json`, `requirements.txt`, `Cargo.toml` change |
| **Docs writer** | Public API drift | "`def login(...)` signature changed; docstring stale" | Public function signature changes |
| **Refactorer** | Duplication, dead code | "This is the third copy of this regex" | Repetition appears |
| **Product / UX** | User-facing impact | "This error message uses 'invalid token' for two distinct cases" | Strings/UI text |
| **Rubber duck / Socratic** | User's stated intent vs. actions | "You said you wanted X; this commit does Y. Intentional?" | Always available; very low fire rate |
| **Data / ML** | Training, eval, leakage | "Same DataFrame is used for fit and score — leakage?" | Touching training/eval pipelines |

### 4.2 Separate calls vs. router-with-single-call

This is the most important architectural decision, and the answer is **router → 1–N parallel role calls**, not "one call that picks a role" and not "all roles always."

- **One call with internal role selection.** Fastest, cheapest, but the model is biased toward the most "obvious" role and rarely produces deep critique in a non-architect role; consistency bias compresses everything to the same flavour.
- **All roles always.** The "Greek chorus" problem. Token cost scales linearly; UX gets buried.
- **Router-then-parallel (recommended).** A small, fast model (Haiku class) answers a single question: *which 0, 1, or 2 roles, if any, are likely to have something specific and non-obvious to say about this event?* Then you call only those roles, in parallel, with a structured-output schema that lets each role emit `{"silent": true}` if it has nothing to add after seeing the actual context.

The router is where you stop the chorus. Fired on every debounced event, it should return `[]` for the vast majority of events.

### 4.3 Router prompt

```
You are a triage system. Given the recent activity in a Claude Code coding 
session and the recent file diffs, decide which of the following advisor roles, 
if any, are likely to have a *specific, non-obvious, actionable* observation 
to make.

ROLES:
- architect: abstractions, coupling, module boundaries
- debugger: missing error handling, edge cases, untested branches
- security: injection, secrets, auth, untrusted input
- teacher: concept the user is actively touching that they may misunderstand
- tester: missing tests for new logic
- performance: obvious algorithmic / I/O issues
- devops: deploy/config/CI surface
- (...)

DEFAULT IS EMPTY. Return [] unless you can name a SPECIFIC reason for each 
role you select. Maximum 2 roles. No vague hedging.

Output format:
{"roles": ["debugger"], "rationale": "function added with bare except: pass on line 47"}
or
{"roles": [], "rationale": "trivial typo fix"}
```

Run the router on Haiku/Flash; cost is tiny. If `roles == []`, the loop ends here — no expensive call, no commentary.

### 4.4 Avoiding the Greek chorus

Three layered defenses:

1. **Router emits ≤2 roles.** Hard cap.
2. **Each role can self-mute** by returning a `silent` decision if its output would be generic.
3. **Cross-role dedupe.** When two roles return overlapping content (e.g. security and debugger both flag the same `eval`), the formatter picks the higher-priority one (security > debugger > everything) and drops the duplicate.

---

## 5. Prompt Design (Primary Focus)

This is the single highest-leverage area in the system. A passive-observer LLM is in a uniquely difficult prompt regime: it has not been asked anything, the user is busy, and most of what it could say is noise. The prompts must default to silence.

### 5.1 What goes wrong by default

If you give a vanilla LLM "here is a recent CC transcript and a diff, comment on it," you get:

- Restating what just happened ("It looks like you added a function to validate user input...")
- Generic platitudes ("Consider adding error handling to make the code more robust")
- Hedged, low-confidence suggestions ("You *might* want to *consider* whether *perhaps* a test would *potentially* be useful")
- Cargo-cult style commentary ("Add docstrings, use type hints, consider PEP 8")
- Confident hallucinations of file paths, function names, or code that wasn't in the diff

The prompts have to actively counter each of these.

### 5.2 System prompt structure

Per-role system prompts share a common spine. Skeleton:

```text
You are the {{ROLE}} on a developer's pair-programming team. You are NOT 
the developer's assistant; you are a peer reviewer who watches, mostly 
silent, and speaks only when you have something concrete and useful to say.

# Your concern
{{ROLE_SPECIFIC_CONCERN}}    # e.g. "Detect security risks in the recent changes."

# What you will see
1. A summary of the developer's current task (inferred from the conversation).
2. The most recent ~30 turns of the Claude Code session, including tool calls.
3. The git diff against HEAD for files touched in the last few minutes.
4. A list of past observations you (and your peers) have already made in this 
   session, including which the developer accepted, dismissed, or ignored.

# Your job
Find the ONE most useful, specific, non-obvious thing to say — or say nothing.

# What "useful" means here
- Anchored to a concrete file:line. Never say "consider adding error 
  handling" without naming where.
- Non-obvious. If a senior engineer would say "yeah, obviously" — don't say it.
- Actionable. The developer should be able to do something with it in <2 minutes.
- Not already said. Read the past observations. If you're repeating, stay silent.
- Not contradicting an explicit user instruction. If the user told Claude 
  "skip tests for now," do NOT nag them about missing tests.

# Hard rules
- DO NOT summarize what the developer/Claude just did.
- DO NOT comment on style, formatting, naming unless egregious AND 
  this is a style-related role.
- DO NOT say "you might want to" or "consider" or "perhaps" or "it would 
  be a good idea to". Make claims and recommendations.
- DO NOT mention concerns you cannot point at a specific line for.
- DO NOT invent file paths, line numbers, or symbols. If you would need 
  to guess, stay silent.

# Output (strict JSON)
Either:
{"silent": true, "reason": "<one short sentence why nothing useful to say>"}
or:
{
  "silent": false,
  "severity": "info" | "suggestion" | "warning",
  "anchor": {"file": "<path relative to cwd>", "line": <int>},
  "claim": "<one sentence stating the observation, no hedging>",
  "evidence": "<one sentence pointing at the exact code or transcript line>",
  "suggestion": "<one sentence with a concrete next action>",
  "confidence": 0.0..1.0
}

# Calibration
Default to silent. Aim for at most 1 non-silent output per ~10 calls. 
If you find yourself reaching, set silent: true.
```

### 5.3 The "notice what's missing" prompt

The single most valuable prompt direction for this product is *negative-space reasoning*. The LLM is naturally good at commenting on what's there; you have to actively prompt it for absences. A reusable block:

```text
# Negative-space pass
Before deciding, ask yourself:
- What test should exist for this change but doesn't?
- What error path is left unhandled?
- What edge case (null, empty, very large, concurrent, malformed) is 
  not addressed?
- What invariant is now violated that wasn't before?
- What documentation or callsite is now stale because of this change?
- What was in the user's stated goal that the change does NOT fulfill?

For each, either name a specific instance (file:line) or move on. Do not 
emit "considered adding tests" as commentary — emit the specific test 
that's missing, or stay silent.
```

The literature on this is sparse but converging: Kang et al.'s prompt-knowledge-gap taxonomy directly maps onto the categories above; "red-teaming" prompts in alignment work show that explicit "list what would break this" framing dramatically increases gap detection. Ehsani et al.'s knowledge-gap taxonomy (Missing Context, Missing Specifications, Multiple Context, Unclear Instructions) is a useful checklist for the **teacher** role.

### 5.4 Forcing specificity

The strongest single technique is *output format constraints that mechanically reject vagueness*:

- Required `anchor.file` and `anchor.line` fields in JSON — the model cannot complete the schema without picking a line.
- Required `evidence` field that quotes the actual code or transcript line — if the model can't quote it, it can't have observed it.
- Banned-word list in the system prompt: "consider", "perhaps", "might want to", "it could be", "you may", "I would suggest", "best practice".
- Few-shot examples where the model emits `{"silent": true}` when the only thing it could say is generic.

You can additionally validate at parse time: if `claim` contains banned hedging phrases, reject and re-prompt once with `"Your last claim was hedged. Restate without softening or set silent: true."` This is cheap because the second pass is short.

### 5.5 Calibrated silence

Getting the LLM to *actually* output `silent: true` requires more than telling it to. Effective techniques:

1. **Explicit budget.** "Aim for at most 1 non-silent output per ~10 calls" sets a quota the model honors surprisingly well.
2. **Few-shots that are ~80% silent.** If your examples show silence as the most common outcome, the model anchors there.
3. **Structured "would the developer change behavior?" gate.** Add: "Before you set `silent: false`, ask: would a smart, busy developer do something *different* in the next 5 minutes because of this comment? If no, stay silent."
4. **Confidence threshold.** Have the role emit `confidence`, then drop anything below 0.7 in post-processing. Don't trust the model to gate itself on confidence alone — the literature on LLM confidence calibration (e.g. Causal Evidence that Language Models Use Confidence to Drive Behavior) shows models *can* abstain when prompted to but require explicit threshold framing.
5. **"Soft sycophancy" guardrails.** The "Silicon Mirror" anti-sycophancy work documents a class of failure mode where models hedge-then-validate rather than disagree. If you see "[hedge], but [actual point]", that is the sycophancy pattern — explicit instruction to drop the hedge prefix helps.
6. **Reward sparsity in evaluation.** When you build an eval set, score "appropriately silent" as positive. Otherwise the model trends toward "say something, anything."

### 5.6 Stale context

The user has moved on. The transcript window from 5 minutes ago may be irrelevant to what's in `cwd` right now. Counter:

- Always pass a *fresh* `git status` and the most recent N file mtimes, plus the **last user message timestamp**.
- In the system prompt: *"If the developer's most recent action contradicts the focus of an earlier observation, treat the earlier observation as obsolete."*
- In memory retrieval (§6), prefer recency over semantic similarity for the working buffer.

### 5.7 Few-shot examples per role

For each role, two or three few-shots — at least one ending in `silent: true`. Example for the **debugger**:

```text
EXAMPLE 1 (silent — too generic to say)
Recent change: Added a docstring to `parse_config()`.
Output: {"silent": true, "reason": "documentation-only change"}

EXAMPLE 2 (non-silent — specific, anchored)
Recent change: 
  + def get_user(user_id: str) -> User:
  +     row = db.fetch_one(f"SELECT * FROM users WHERE id = '{user_id}'")
  +     return User(**row)
Output: {
  "silent": false,
  "severity": "warning",
  "anchor": {"file": "users.py", "line": 14},
  "claim": "SQL injection: user_id is interpolated directly into the query.",
  "evidence": "f\"SELECT * FROM users WHERE id = '{user_id}'\"",
  "suggestion": "Use db.fetch_one(\"SELECT * FROM users WHERE id = ?\", (user_id,)).",
  "confidence": 0.95
}

EXAMPLE 3 (silent — already said)
Past observations include: "SQL injection at users.py:14" (DISMISSED by user)
Recent change: minor refactor to users.py:14 still has the issue.
Output: {"silent": true, "reason": "already raised and dismissed"}
```

### 5.8 Output format: structured JSON wins

For everything except the final user-facing rendering, use strict JSON with a schema enforced at parse-time. Reasons:

- Mechanically prevents vague output.
- Trivially deduplicates across roles by hashing `(file, line, claim_keywords)`.
- Lets the formatter style the output for the surface (TUI vs. status line vs. notification).
- Makes evals easy.

Use Anthropic's tool-use mode or `response_format: json_schema` for guaranteed validity.

### 5.9 Citation/reference format

Inside `claim` and `suggestion`, refer to files as `path/to/file.py:42` (Goneril/Grep convention; clickable in many terminals via OSC-8 hyperlinks). Refer to symbols as `module.Class.method`. Refer to transcript turns by relative position only ("two turns ago"), never by UUID — UUIDs leak into commentary readability.

### 5.10 Anti-sycophancy and anti-hallucination for passive observation

Two compounding risks in this regime:

1. **Sycophancy** — even without a user prompt, models trained with RLHF will hedge to seem helpful. Counter with: explicit "you are not the user's helper, you are a peer reviewer" framing; banned-word list; confidence gate.
2. **Hallucinated anchors** — the model invents file paths or line numbers because the schema requires them. Counter with: a post-LLM verifier that rejects any `anchor` whose `file` does not exist or whose `line` is beyond EOF; on rejection, retry once with `"Your anchor was invalid; either find a real one or set silent: true."`

A simple verifier loop catches the vast majority of hallucinations:

```python
def verify(out, cwd):
    if out.get("silent"): return out
    a = out["anchor"]
    p = Path(cwd) / a["file"]
    if not p.exists(): return None
    n_lines = sum(1 for _ in p.open())
    if not (1 <= a["line"] <= n_lines): return None
    if "evidence" in out:
        # the evidence string should appear in the file or in the recent diff
        if out["evidence"] not in p.read_text() and out["evidence"] not in recent_diff:
            return None
    return out
```

### 5.11 Borrowed agentic techniques

Translating prompting techniques from active agents to a passive critic:

- **ReAct (Reason+Act).** Useful but the "Act" surface for a critic is read-only: read the file, grep for callsites, run the type checker. Give the critic role a small read-only toolset (`read_file`, `grep`, `git_log_for_file`, `run_tests --dry-run`); CRITIC paper shows tool-grounded critique substantially outperforms tool-free.
- **Plan-and-solve.** For multi-step critiques: first plan (what would I check?), then execute. Useful for deep architecture reviews; overkill for single-event ambient commentary.
- **Self-critique.** *Not* recommended in this regime. Lanham (2026) and the broader Reflexion-vs-PRM literature show LLM-on-LLM critique without an external signal often degrades quality. Where you want self-critique, prefer a different role with a different system prompt as the "second pass" — that's effectively multi-agent debate.
- **Reflexion.** Useful as memory: when the developer dismisses commentary, store the dismissal as a "this was wrong/unhelpful" signal the role consults next time.

---

## 6. Memory

A passive advisor needs three layers of memory; the pragmatic implementation is small and uses one boring database.

### 6.1 Layers

- **Working buffer (per invocation).** The last ~30 transcript turns plus the recent file diffs plus the past 5 commentary outputs. Held in process memory, rebuilt each invocation.
- **Session state.** Per-CC-session: list of files touched, current inferred task, observations made (with status: new/seen/accepted/dismissed/ignored), last router decision per event hash. Persisted to SQLite. Keyed by `session_id` from the hook envelope.
- **Project state.** Per-cwd: architectural notes the user has confirmed ("we use FastAPI, not Flask"), conventions ("we use ruff with these rules"), past advice that landed, past advice that was rejected with reason, the developer's expressed preferences. Persisted to SQLite plus a small embedding index (per-cwd `~/.cache/advisor/<project-hash>/`).
- **User state (cross-project).** Long-term style preferences ("dislikes nags about tests in spike branches"), language proficiency hints, blocked roles, etc. Tiny JSON file in `~/.config/advisor/`.

### 6.2 Storage choice

**SQLite is correct here.** Reasons:

- Zero-config, single file, trivially backed up.
- Full-text search comes for free with FTS5.
- JSON1 extension lets you store role outputs as JSON columns and still query.
- Vector search via `sqlite-vec` extension is enough for project-scale (<10k items) — no separate vector DB needed.
- Survives crashes.

Schema sketch:

```sql
CREATE TABLE observation (
  id INTEGER PRIMARY KEY,
  ts INTEGER NOT NULL,
  session_id TEXT NOT NULL,
  project_hash TEXT NOT NULL,
  role TEXT NOT NULL,
  anchor_file TEXT, anchor_line INTEGER,
  claim TEXT NOT NULL,
  body_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'new',  -- new|seen|accepted|dismissed|ignored
  embedding BLOB                         -- via sqlite-vec
);
CREATE INDEX ix_obs_session ON observation(session_id);
CREATE INDEX ix_obs_project_status ON observation(project_hash, status);
CREATE VIRTUAL TABLE observation_fts USING fts5(claim, body_json,
  content='observation', content_rowid='id');
```

Vector store, OWL/RDF, or even a graph DB are over-engineered for this; revisit only if you need to model deep architectural facts (v3+).

### 6.3 Retrieval strategy

For each new event, the role-call context includes:

1. The last 5 observations on this file (recency, by `mtime`/`ts`).
2. The 3 highest-similarity past observations across the project for this role and status ∈ {accepted, dismissed} — *especially dismissed*. The dismissed ones serve as anti-examples ("don't say this again").
3. Any user-confirmed project conventions tagged for this role.

This is RAG-flavored but small; the embedding model can be local (e.g. `nomic-embed-text` via Ollama) to avoid cost and latency.

### 6.4 Compaction and eviction

The transcript window can be 30 turns in the system prompt and unlimited on disk. For working-buffer compaction, use a tiered approach:

- Last 10 turns verbatim.
- Turns 11–30 summarized in 1–2 lines each by a Haiku-class call (cached on the first observation that needed it).
- Older turns → "task summary so far" (one paragraph, regenerated every N turns).

This mirrors how CC itself does compaction (the architecture analysis mentions a 5-layer compaction pipeline) and is cheap enough to run inline.

### 6.5 Repeat suppression

Crucial for not annoying the user. Before emitting an observation:

```python
def should_emit(new_obs, project_hash, conn):
    # Exact-anchor match within 24h: skip
    cutoff = time.time() - 86400
    same = conn.execute("""
        SELECT id, status FROM observation
        WHERE project_hash=? AND anchor_file=? AND anchor_line=? AND ts > ?
    """, (project_hash, new_obs.anchor.file, new_obs.anchor.line, cutoff)).fetchall()
    if same:
        return False
    # Semantic match: skip if a near-duplicate exists in last 7 days
    sim = vector_search(conn, new_obs.embedding, role=new_obs.role,
                        days=7, threshold=0.85)
    return not sim
```

The user's `dismissed` reactions feed back into a *role-level cooldown*: if the user has dismissed three security warnings in a row in this session, mute the security role for the rest of the session.

---

## 7. Triggers and Scheduling

### 7.1 Comparison

| Strategy | Pro | Con |
|---|---|---|
| Pure interval (every N s) | Simple | Wasteful; cost adds up; lags behind events |
| Pure event-driven | Lowest latency | Spam during bursts |
| Debounced event-driven | Best of both | Needs careful debounce window |
| Heartbeat + event hybrid | Catches "stuck" patterns | More state |
| MCP pull | Cheap; only fires on demand | Requires CC to invoke; defeats "ambient" |
| Hook pull (Stop / PostToolUse) | CC tells you exactly when to look | Misses purely file-side activity |

### 7.2 Recommended: hooks + debounce + heartbeat

- **Hook events** (`Stop`, `PostToolUse(Write|Edit|MultiEdit)`, `UserPromptSubmit`) push events into the bus.
- **Filesystem events** also push events.
- A **debouncer** waits 1.5 s of quiet before firing the router.
- A **heartbeat** every 5 minutes checks: "has the user been working on the same file with no progress for >10 min?" and fires a special "stuck" event the router can dispatch to the rubber-duck role.
- A **rate limiter** caps total LLM cost: at most one router call per 30 s, at most three role calls per minute, soft-cap of $X/day.

Pseudocode:

```python
class Trigger:
    def __init__(self, debounce=1.5, heartbeat=300):
        self.events = asyncio.Queue()
        self.last_fire = 0
        self.heartbeat = heartbeat

    async def run(self, dispatch):
        loop = asyncio.get_event_loop()
        last_event_ts = loop.time()
        while True:
            try:
                ev = await asyncio.wait_for(self.events.get(),
                                            timeout=self.heartbeat)
                last_event_ts = loop.time()
                # debounce — drain anything else within the window
                deadline = loop.time() + 1.5
                batch = [ev]
                while loop.time() < deadline:
                    try:
                        batch.append(await asyncio.wait_for(
                            self.events.get(), timeout=deadline - loop.time()))
                    except asyncio.TimeoutError:
                        break
                await dispatch(batch, kind="event")
            except asyncio.TimeoutError:
                # heartbeat
                await dispatch([{"kind": "heartbeat"}], kind="heartbeat")
```

### 7.3 Cost considerations

A typical day of active CC use produces dozens to a few hundred tool-use events. With router-then-parallel architecture and ~80% silence:

- Router: 100 calls × ~500 tokens × Haiku = pennies.
- Roles: 20 calls × ~3 K tokens × Sonnet = under a dollar.
- Embeddings: free (local).

For Pro/Max subscribers: route the router to Haiku via API key (cheap) and the roles to Sonnet via the user's existing Claude Max subscription if you can — but **CC does not currently support MCP sampling**, so you can't easily piggyback on Max from within CC's MCP server. The pragmatic answer is API key with a per-day spend cap.

---

## 8. Edge Cases and Failure Modes

| Failure mode | Mitigation |
|---|---|
| User mid-typing; commentary lags reality | Debounce 1.5 s; always include latest user message timestamp; older context marked "stale" |
| User explicitly told CC to do X; advisor disagrees | Detect imperative user instructions in the recent transcript; system prompt: "do NOT contradict an explicit user instruction." Allow user override via `# advisor: skip-for-now` magic comment, à la Aider's `# AI` |
| Sensitive files (`.env`, secrets) | Hard denylist before files are sent to the LLM; `.env*`, `*.pem`, `*.key`, `*credentials*`, `*.p12`, anything matching project's `.gitignore` AND containing `=` patterns |
| Very large diffs (10 K files) | Threshold: if >50 changed paths in a window, skip LLM; emit a single "bulk change ignored" log line |
| Multiple concurrent CC sessions in different cwds | Key all state by `(session_id, project_hash)`; sidecar shows roster; per-session rate limits |
| Advisor commentary leaks into CC context | If you use `UserPromptSubmit`'s `additionalContext` to inject, mark it explicitly as advisor output and never let CC's responses re-enter the advisor's view as "user content" — strip them in the tailer |
| Privacy / on-prem | Two-tier mode: `--local` runs everything through Ollama / local model; `--cloud` uses Anthropic API. Default to local for the router (it's small) and let the user opt in to cloud for roles |
| Cost runaway | Token budget per day, per session, per role; hard stop with a notification; show running cost in status line |
| Conflicting advice across roles | Cross-role dedupe by anchor; severity ordering (security > debugger > architect > teacher > tester > performance > devops > ...) |
| Race: advisor reads transcript while CC writes it | The tailer's tolerant parser drops malformed trailing lines and re-reads them next tick; UUID dedupe prevents duplicates |
| User ignoring advice | Per-anchor cooldown (24 h); per-role session cooldown after 3 dismissals; per-claim global suppression on dismiss |
| Time-stale advice | Always include `as_of: <timestamp>` and `git_head: <sha>` in the observation; in formatter, drop anything where `git diff` of the anchored file has changed since |
| Hallucinated paths/lines | Verifier (§5.10) — reject and retry once or stay silent |
| CC version drift | Snapshot the JSONL schema fields you depend on; defensive parsing; pin to `cwd`, `transcript_path`, `session_id`, `tool_name`, `tool_input`, `tool_response` (these are the most stable) |

---

## 9. Concrete Architecture Recommendation

### 9.1 Components

```
┌─────────────────────┐     ┌─────────────────────┐
│  Claude Code        │     │   Filesystem        │
│  (Stop, PostToolUse,│     │   (cwd, watchdog)   │
│   UserPromptSubmit) │     │                     │
└──────────┬──────────┘     └──────────┬──────────┘
           │ async hooks                │ debounced
           │  (HTTP POST)                │   events
           ▼                             ▼
       ┌───────────────────────────────────────┐
       │         Advisor sidecar (Python)      │
       │                                       │
       │   ┌────────────┐    ┌────────────┐    │
       │   │ Hook        │    │ FS watcher │    │
       │   │ ingester    │    │ (watchdog) │    │
       │   └─────┬──────┘    └─────┬──────┘    │
       │         └──────┬──────────┘            │
       │                ▼                       │
       │        ┌───────────────┐               │
       │        │ Event bus      │               │
       │        │ (asyncio.Queue)│               │
       │        └───────┬───────┘               │
       │                ▼                       │
       │    ┌──────────────────────┐            │
       │    │ Debouncer + heartbeat│            │
       │    └────────┬─────────────┘            │
       │             ▼                          │
       │    ┌──────────────────────┐            │
       │    │ Context builder      │            │
       │    │  - tail JSONL         │            │
       │    │  - git diff           │            │
       │    │  - working buffer    │            │
       │    │  - memory retrieval  │            │
       │    └────────┬─────────────┘            │
       │             ▼                          │
       │    ┌──────────────────────┐            │
       │    │ Router (Haiku)       │            │
       │    │  → list of roles     │            │
       │    └────────┬─────────────┘            │
       │             ▼                          │
       │    ┌──────────────────────┐            │
       │    │ Role workers (parallel)           │
       │    │  - architect, debugger, ...       │
       │    │  - JSON schema enforced           │
       │    │  - hallucination verifier         │
       │    └────────┬─────────────┘            │
       │             ▼                          │
       │    ┌──────────────────────┐            │
       │    │ Suppressor (memory)  │            │
       │    │  - dedupe / cooldown │            │
       │    │  - severity ordering │            │
       │    └────────┬─────────────┘            │
       │             ▼                          │
       │    ┌──────────────────────┐            │
       │    │ Renderer / output    │            │
       │    │  - TUI / notif / file│            │
       │    │  - optional MCP push │            │
       │    └────────┬─────────────┘            │
       │             ▼                          │
       │    ┌──────────────────────┐            │
       │    │ SQLite + sqlite-vec  │            │
       │    │ (memory)             │            │
       │    └──────────────────────┘            │
       └───────────────────────────────────────┘
```

### 9.2 Tech stack

- **Language**: Python 3.12. Matches the user's stack and has the right libraries.
- **Filesystem watching**: `watchdog`.
- **Event loop**: `asyncio` + `httpx` for the LLM client.
- **LLM**: Anthropic Claude (`anthropic` SDK) by default, with a pluggable provider interface so you can swap to OpenAI, Gemini, or local models. Use Haiku for the router, Sonnet for the roles.
- **Persistence**: SQLite + `sqlite-vec` extension.
- **Embeddings**: local via Ollama (`nomic-embed-text`) by default; swap to API on demand.
- **Hook ingress**: HTTP hook → FastAPI on a localhost-only port, OR a tiny stdin-reading script that POSTs to the daemon.
- **TUI** (optional): `textual` for the dashboard.
- **Packaging**: `uv` for fast dependency installs; ship as a single CLI (`advisor`).
- **Configuration**: `~/.config/advisor/config.toml` with role enable/disable, model choices, cost cap, redaction rules.

### 9.3 Surface: CLI sidecar + optional MCP

- **Phase 1** — **CLI sidecar**. Single command `advisor watch [--cwd PATH]` runs the daemon. Output goes to a panel in your terminal multiplexer or a `tail -f`-able file at `.advisor/log.jsonl` and as desktop notifications via `terminal-notifier`/`notify-send`. CC integrates via hook scripts in `.claude/settings.json` that POST events to `http://127.0.0.1:port/event`.
- **Phase 2** — **TUI**. A `textual` panel showing the live transcript on the left, observations on the right, with keyboard shortcuts to dismiss/accept/expand. Dismissals feed back into memory.
- **Phase 3** — **MCP server**. Expose `advisor.review_recent_turn`, `advisor.explain_concept(symbol)`, `advisor.check_security_of(path)` as MCP tools so CC can pull when it wants. If CC ever ships sampling support, switch role calls to use it for cost.

### 9.4 Phased implementation plan

**Day 1–3 — MVP (CLI, no MCP, two roles).**
- Hook scripts that `curl localhost:9876/event`.
- Sidecar reads JSONL with the tailer above, runs git diff on touched files, calls one combined "debugger + security" prompt with the silent-default schema, writes JSON observations to `.advisor/log.jsonl` and terminal notifications.
- SQLite for dedupe by anchor.

**Week 2 — v1 (multi-role, router, memory).**
- Add router (Haiku), four roles (architect, debugger, security, teacher).
- Memory: session state, repeat suppression, role cooldowns.
- Hallucination verifier.
- Cost cap and rate limiter.

**Month 2 — v2 (TUI, project memory, more roles).**
- Textual TUI; user feedback (accept/dismiss) feeds memory.
- Project memory: confirmed conventions, dismissed-with-reason corpus, semantic dedupe via `sqlite-vec`.
- Tester, performance, dependency-manager, devops roles.
- `# advisor: skip` and `# advisor: explain` magic comments in code.

**Month 3 — v2.5 (MCP server + redaction polish).**
- MCP server exposing pull-mode tools.
- Refined redaction (secrets detection beyond filename match — entropy heuristics on diff content).
- Optional `claude/channel` push if the CC MCP support stabilizes.

**v3 — deeper signals.**
- Plug in the LSP for the project (via `multilspy` or the language's own LSP) and use diagnostics as a *first-class* signal alongside diffs — much cheaper than asking the LLM to do type checking.
- Run the project's test suite when files change and feed failures into the debugger role.
- Index the codebase with embeddings for cross-file architecture commentary (Greptile-style, but per-machine).
- Optional: a "debate" mode where the architect role and the refactorer role disagree explicitly; show the user both sides.

### 9.5 Open questions and tradeoffs without a clean winner

A few decisions are genuinely open:

- **Inject commentary back into CC?** Tempting (use `UserPromptSubmit`'s `additionalContext`). Risk: feedback loops, polluted context, model getting confused about who's saying what. The conservative answer is *no by default*; provide a `/advisor` slash command (custom CC command) the user can invoke to fold the latest observations into the next turn.
- **Self-critique before emitting?** A second-pass critic that reads the role's draft and only allows it through if it's specific and non-obvious. Adds latency and cost; in practice the structured output + verifier + memory-suppression pipeline gets you most of the way. Treat as v3.
- **Local model for security role.** Some users will not want to send code to a cloud API for a tool they didn't ask for. Worth offering a fully-local mode (Llama-3.x-Instruct or DeepSeek-Coder via Ollama) even at quality cost.
- **How aggressively to surface.** Some users will want a bell on every observation; some will want a status-bar dot only. The ratio of `info`/`suggestion`/`warning` severities maps roughly to "log-only / status-bar / notification"; let users configure.
- **Eval methodology.** The hardest part of this product to validate is "is the *silence* good?" Standard LLM evals score correctness on outputs; you also need to score appropriate non-outputs. Build an eval set with both "should have spoken" and "should have stayed silent" cases; track precision and recall of *non-silence*.

---

## Appendix A — Sample JSONL records

```jsonc
// User turn
{
  "type": "user", "uuid": "8a52...", "parentUuid": null,
  "sessionId": "eb5b0174-...", "cwd": "/Users/x/proj", "gitBranch": "main",
  "timestamp": "2026-05-01T10:00:00Z",
  "message": {"role": "user", "content": "Add auth to /api/users"}
}
// Assistant tool call
{
  "type": "assistant", "uuid": "e431...", "parentUuid": "8a52...",
  "message": {
    "role": "assistant", "model": "claude-sonnet-4.5",
    "content": [
      {"type": "thinking", "thinking": "I should read the existing routes..."},
      {"type": "tool_use", "id": "tu_01", "name": "Read",
       "input": {"file_path": "/Users/x/proj/api.py"}}
    ],
    "usage": {"input_tokens": 1234, "output_tokens": 89}
  }
}
// Tool result
{
  "type": "user", "uuid": "f021...", "parentUuid": "e431...",
  "message": {"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "tu_01",
     "content": "from fastapi import FastAPI\n..."}
  ]}
}
// Compact boundary
{ "type": "compact_boundary", "uuid": "...", "parentUuid": "...",
  "headUuid": "...", "anchorUuid": "...", "tailUuid": "..." }
// File-history snapshot (resume-time)
{ "type": "file-history-snapshot", "messageId": "8a52...",  // collides — see issue 36583
  "snapshot": {"files": [...]} }
```

## Appendix B — Hook envelope on stdin (PostToolUse, Edit)

```json
{
  "session_id": "eb5b0174-0555-4601-804e-672d68069c89",
  "transcript_path": "/Users/x/.claude/projects/-Users-x-proj/eb5b0174-....jsonl",
  "cwd": "/Users/x/proj",
  "permission_mode": "default",
  "hook_event_name": "PostToolUse",
  "tool_name": "Edit",
  "tool_input": {"file_path": "/Users/x/proj/api.py",
                 "old_string": "...", "new_string": "..."},
  "tool_response": {"success": true, "filePath": "/Users/x/proj/api.py"}
}
```

## Appendix C — Minimal role call

```python
import json, anthropic
client = anthropic.Anthropic()
SYSTEM = open("prompts/security.md").read()

def role_call(context: dict) -> dict | None:
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=600,
        system=SYSTEM,
        messages=[{"role": "user", "content": json.dumps(context)}],
        tools=[{
            "name": "emit_observation",
            "description": "Emit one observation, or stay silent.",
            "input_schema": {
                "type": "object",
                "oneOf": [
                  {"properties": {"silent": {"const": True},
                                  "reason": {"type": "string"}},
                   "required": ["silent"]},
                  {"properties": {
                    "silent": {"const": False},
                    "severity": {"enum": ["info","suggestion","warning"]},
                    "anchor": {"type": "object",
                               "properties": {"file":{"type":"string"},
                                              "line":{"type":"integer"}},
                               "required":["file","line"]},
                    "claim": {"type":"string"},
                    "evidence": {"type":"string"},
                    "suggestion": {"type":"string"},
                    "confidence": {"type":"number","minimum":0,"maximum":1}},
                   "required":["silent","severity","anchor","claim",
                               "evidence","suggestion","confidence"]}
                ]
            }
        }],
        tool_choice={"type": "tool", "name": "emit_observation"})
    out = next(b.input for b in resp.content if b.type == "tool_use")
    if out.get("silent"): return None
    if out["confidence"] < 0.7: return None
    return verify(out, context["cwd"])
```

This is the whole product in miniature: structured output, silent-by-default, threshold gate, verifier. The remaining engineering is plumbing — hooks, watchers, debouncing, memory, formatting — and the remaining art is in the system prompts.