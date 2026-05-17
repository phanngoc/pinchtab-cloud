"""Claude CLI subprocess provider — alternative to the Anthropic SDK.

Wraps the local `claude` CLI binary (authenticated via `claude setup-token`)
so the agent runner can drive a task using the operator's Claude subscription
instead of a paid API key. Returns objects shaped like the Anthropic Python
SDK so existing call sites work unchanged.

TOS scope (read carefully before deploying):
   This provider is for SINGLE-USER, OPERATOR-ONLY use. Anthropic's Claude
   Code terms prohibit using your subscription to provide service to third
   parties. The middleware in tasks.py enforces an OPERATOR_EMAIL allowlist
   when this provider is selected; any other authenticated user is blocked
   with 403 before the runner spawns.

Architecture notes:
   - No user-agent or header spoofing. We `exec` the official `claude` binary
     and feed it a prompt over stdin. The CLI handles its own OAuth.
   - The CLI does not accept a custom tools schema. We prompt Claude to
     respond in a fixed XML format ('<thought>...<tool>...<args>JSON</args>')
     and parse server-side back into mock SDK content blocks. Tool use is
     therefore text-pattern-driven, less robust than the native API.
   - No vision. Screenshots from pinchtab are dropped on the input path; the
     agent receives only the a11y text snapshot. Acceptable for most tasks
     against well-structured pages; degrades on visual-only UIs.
   - One subprocess per messages.create() call. ~3-5s cold-start overhead
     per turn vs the SDK.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import shutil
from pathlib import Path
from typing import Any

log = logging.getLogger("llm_cli")


def _empty_mcp_config_path() -> str:
    """Ensure a stable on-disk empty MCP config exists; return its path.

    The agent runner doesn't use any MCP tools — it only needs claude as a
    raw LLM responder. We pass `--strict-mcp-config --mcp-config <this>` so
    claude skips loading the user's 7 default MCP servers (deepwiki, serena,
    magic, context7, sequential-thinking, morphllm-fast-apply, playwright).
    This trims cold-start by 1-3s AND avoids hanging when any MCP server
    misbehaves on startup."""
    cfg_path = Path.home() / ".claude" / "pinchtab-empty-mcp.json"
    if not cfg_path.exists():
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text('{"mcpServers": {}}\n')
    return str(cfg_path)


_EMPTY_MCP_CONFIG = _empty_mcp_config_path()


# ---- Mock content blocks matching the Anthropic SDK shape ----


class _Block:
    """Mimics anthropic.types.TextBlock / ToolUseBlock surface area."""

    def __init__(self, type: str, **fields):
        self.type = type
        for k, v in fields.items():
            setattr(self, k, v)

    def model_dump(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type}
        for attr in ("id", "name", "input", "text"):
            if hasattr(self, attr):
                d[attr] = getattr(self, attr)
        return d


class _Response:
    """Mimics anthropic.types.Message surface area."""

    def __init__(self, content: list[_Block]):
        self.content = content

    def model_dump(self) -> dict[str, Any]:
        return {"content": [b.model_dump() for b in self.content]}


class _Usage:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_input_tokens = 0


# ---- Prompt builder / response parser ----


_OUTPUT_FORMAT_INSTRUCTIONS = """
# Output Format (CRITICAL)

You MUST respond with EXACTLY this structure on a SINGLE turn:

<thought>your brief reasoning here, max 2 sentences</thought>
<tool>tool_name</tool>
<args>{"arg_name": "value", ...}</args>

Rules:
- Exactly one tool call per response.
- args MUST be valid JSON.
- Use only tools listed in "Available Tools" below.
- If task is finished, use task_complete.
- If you cannot proceed safely (captcha, OTP, login needed), use halt_for_human.
- Do not output anything outside these three tags. No preamble, no code fences.
"""


_TAG_THOUGHT = re.compile(r"<thought>(.*?)</thought>", re.DOTALL)
_TAG_TOOL = re.compile(r"<tool>(.*?)</tool>", re.DOTALL)
_TAG_ARGS = re.compile(r"<args>(.*?)</args>", re.DOTALL)


def _flatten_message_content(content: Any) -> str:
    """Render Anthropic-style content blocks into plain text for the CLI prompt."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    out: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        bt = block.get("type")
        if bt == "text":
            out.append(block.get("text", ""))
        elif bt == "image":
            # No vision in CLI mode.
            out.append("[screenshot attached — not rendered in text mode]")
        elif bt == "tool_use":
            args_json = json.dumps(block.get("input", {}), ensure_ascii=False)
            out.append(f"<tool>{block.get('name','')}</tool>\n<args>{args_json}</args>")
        elif bt == "tool_result":
            res = block.get("content", "")
            if isinstance(res, list):
                # Tool result might be nested blocks; flatten.
                res = "\n".join(
                    b.get("text", "") if isinstance(b, dict) else str(b) for b in res
                )
            out.append(f"<tool_result>{str(res)[:800]}</tool_result>")
    return "\n".join(out)


def build_cli_prompt(
    system: list[dict] | str,
    tools: list[dict],
    messages: list[dict],
) -> str:
    """Render an Anthropic Messages-shape conversation into a single text
    prompt for `claude --print` to consume on stdin."""
    parts: list[str] = []

    # System
    parts.append("# System Instructions")
    if isinstance(system, str):
        parts.append(system)
    else:
        for s in system:
            if isinstance(s, dict) and "text" in s:
                parts.append(s["text"])
    parts.append("")

    # Tools
    parts.append("# Available Tools")
    for t in tools:
        name = t.get("name", "")
        desc = t.get("description", "")
        props = t.get("input_schema", {}).get("properties", {})
        required = t.get("input_schema", {}).get("required", [])
        args_doc = ", ".join(
            f"{k}{'?' if k not in required else ''}: {v.get('type','string')}"
            for k, v in props.items()
        )
        parts.append(f"- **{name}**({args_doc}): {desc}")
    parts.append("")
    parts.append(_OUTPUT_FORMAT_INSTRUCTIONS)
    parts.append("")

    # Conversation
    parts.append("# Conversation So Far")
    for msg in messages:
        role = msg.get("role", "user")
        content = _flatten_message_content(msg.get("content", ""))
        parts.append(f"## {role.title()}")
        parts.append(content)
        parts.append("")

    parts.append("## Assistant (your turn — output exactly the three tags)")
    return "\n".join(parts)


def parse_cli_response(text: str) -> _Response:
    """Extract the structured tool call from claude CLI's text response."""
    blocks: list[_Block] = []

    thought_m = _TAG_THOUGHT.search(text)
    if thought_m:
        blocks.append(_Block("text", text=thought_m.group(1).strip()))

    tool_m = _TAG_TOOL.search(text)
    args_m = _TAG_ARGS.search(text)
    if tool_m and args_m:
        tool_name = tool_m.group(1).strip()
        args_raw = args_m.group(1).strip()
        # Tolerate fenced JSON, leading/trailing whitespace, single quotes.
        args_raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", args_raw, flags=re.MULTILINE)
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            try:
                args = json.loads(args_raw.replace("'", '"'))
            except json.JSONDecodeError:
                log.warning("could not parse tool args JSON: %r", args_raw[:200])
                args = {}
        block_id = f"toolu_cli_{secrets.token_hex(6)}"
        blocks.append(
            _Block("tool_use", id=block_id, name=tool_name, input=args)
        )

    if not blocks:
        # No structured output — wrap the whole reply as text. The runner will
        # error-out on "no tool call returned" which is the right signal.
        blocks.append(_Block("text", text=text.strip()[:2000]))

    return _Response(blocks)


# ---- The provider ----


class _MessagesAPI:
    def __init__(self, parent: "ClaudeCLIProvider"):
        self._parent = parent

    async def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: Any,
        tools: list[dict],
        messages: list[dict],
    ) -> _Response:
        return await self._parent._create(
            model=model, max_tokens=max_tokens, system=system, tools=tools, messages=messages
        )


class ClaudeCLIProvider:
    """Drop-in replacement for AsyncAnthropic when LLM_PROVIDER=claude_cli.

    Single-user dev tool. Don't deploy with multiple authenticated users.
    """

    def __init__(
        self,
        claude_bin: str | None = None,
        timeout_seconds: float = 180.0,
        model_arg: str | None = "sonnet",
    ):
        bin_path = claude_bin or shutil.which("claude")
        if not bin_path:
            raise RuntimeError(
                "claude binary not in PATH. Install Claude Code and run "
                "`claude setup-token` (or `claude login`) first."
            )
        self.bin = bin_path
        self.timeout = timeout_seconds
        self.model_arg = model_arg
        self.messages = _MessagesAPI(self)
        # Session state (stateful per task).
        #
        # First _create() call generates a UUID and runs `--session-id <uuid>`
        # with the full prompt. Subsequent calls run `--resume <uuid>` with
        # ONLY the new user messages since the last call. claude CLI maintains
        # its own conversation history on disk; we avoid re-feeding 30+ turns
        # of context on every subprocess call.
        #
        # Token impact, observed: 30-step run before = ~210KB cumulative
        # prompt across all subprocess invocations. With session mode:
        # turn 1 = 6KB (full), turns 2-30 = ~1KB each (just new snap +
        # tool_result). Total ~36KB. ~6x less work for claude per turn.
        self.session_id: str | None = None
        self._messages_sent_count = 0

    async def _create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: Any,
        tools: list[dict],
        messages: list[dict],
    ) -> _Response:
        import uuid as _uuid

        # First call vs resume: shape command + prompt accordingly.
        if self.session_id is None:
            # First turn: full prompt (system + tools + initial conversation).
            self.session_id = str(_uuid.uuid4())
            prompt = build_cli_prompt(system, tools, messages)
            mode_args = ["--session-id", self.session_id]
            self._messages_sent_count = len(messages)
            log.info(
                "claude CLI new session %s (initial prompt: %d bytes)",
                self.session_id[:8], len(prompt),
            )
        else:
            # Resume: send ONLY new user messages since last call. Assistant
            # responses are already in claude's session from prior --resume
            # turns; sending them again wastes tokens and confuses the model.
            new_msgs = messages[self._messages_sent_count :]
            new_user_msgs = [m for m in new_msgs if m.get("role") == "user"]
            if not new_user_msgs:
                # Defensive: nothing new to send. Shouldn't normally happen
                # since the runner appends a user message every turn.
                new_user_msgs = [{"role": "user", "content": "continue"}]
            prompt = "\n\n".join(
                _flatten_message_content(m.get("content", ""))
                for m in new_user_msgs
            )
            mode_args = ["--resume", self.session_id]
            self._messages_sent_count = len(messages)
            log.info(
                "claude CLI resume %s (delta: %d bytes, %d new user msgs)",
                self.session_id[:8], len(prompt), len(new_user_msgs),
            )

        cmd = [
            self.bin,
            "--print",
            *mode_args,
            # Disable Claude Code's built-in tools (Bash/Read/Edit/etc.) so
            # the CLI behaves as a pure LLM responder for our prompts.
            "--disallowedTools", "*",
            # Speed: don't burn reasoning budget on our per-step task.
            "--effort", "low",
            # Strip Claude Code's per-machine system prompt sections.
            "--exclude-dynamic-system-prompt-sections",
            # Skip MCP server loading — runner uses zero MCP tools, and
            # any one of the user's 7 default MCP servers hanging on
            # startup would hang our subprocess too. Saves cold-start +
            # eliminates a hang class.
            "--strict-mcp-config", "--mcp-config", _EMPTY_MCP_CONFIG,
        ]
        if self.model_arg:
            cmd += ["--model", self.model_arg]

        # NOTE: do NOT pass --no-session-persistence; we WANT the session
        # written to disk so --resume can pick it up on the next turn.

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(prompt.encode("utf-8")),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            # Surface anything claude wrote to stderr before we killed it —
            # otherwise timeouts give zero forensic signal in the agent log.
            stderr_tail = ""
            try:
                if proc.stderr is not None:
                    stderr_bytes = await asyncio.wait_for(proc.stderr.read(), timeout=1.0)
                    stderr_tail = stderr_bytes.decode("utf-8", errors="replace")[-300:]
            except (asyncio.TimeoutError, Exception):
                pass
            msg = f"claude CLI timed out after {self.timeout}s"
            if stderr_tail.strip():
                msg += f" (stderr tail: {stderr_tail.strip()})"
            raise RuntimeError(msg)

        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"claude CLI exit {proc.returncode}: {err}")

        text = stdout.decode("utf-8", errors="replace")
        if not text.strip():
            err = stderr.decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"claude CLI returned empty stdout. stderr: {err}")
        return parse_cli_response(text)


def is_operator(user_email: str) -> bool:
    """Allowlist gate. Only the OPERATOR_EMAIL holder may use this provider.
    Resolves the operator email from app settings first (which reads the .env
    file via pydantic-settings) then falls back to os.environ for cases
    where the runner is invoked outside the FastAPI app context."""
    op = ""
    try:
        from backend.config import get_settings

        op = (get_settings().operator_email or "").strip().lower()
    except Exception:
        pass
    if not op:
        op = os.environ.get("OPERATOR_EMAIL", "").strip().lower()
    if not op:
        return False
    return user_email.strip().lower() == op
