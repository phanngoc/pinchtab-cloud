"""Generic pinchtab + Claude agent loop.

Usage:
  1. Start the pinchtab server in another terminal:
       pinchtab
  2. Open the target site in that browser, log in by hand if needed.
  3. Copy config.example.yaml to config.yaml and edit task + start_url.
  4. export ANTHROPIC_API_KEY=...
  5. python agent.py config.yaml

The agent stops automatically when:
  - it calls task_complete or halt_for_human
  - the page snapshot contains a safety.halt_on_text pattern (e.g. captcha)
  - an element it tries to click matches safety.confirm_before_click_text
  - max_steps is reached

This scaffold is generic. Point it at any web app you own or have permission
to automate. Do not use it to bypass anti-bot controls on third-party sites.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from anthropic import Anthropic

import tools as pt


TOOLS_SCHEMA = [
    {
        "name": "click",
        "description": "Click an element identified by its `refXXX` ID from the latest snapshot.",
        "input_schema": {
            "type": "object",
            "properties": {"ref": {"type": "string"}},
            "required": ["ref"],
        },
    },
    {
        "name": "type_text",
        "description": "Type text into an input/textarea identified by its ref.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ref": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["ref", "text"],
        },
    },
    {
        "name": "select_option",
        "description": "Select an option in a <select> dropdown by its value.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ref": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["ref", "value"],
        },
    },
    {
        "name": "press_key",
        "description": "Press a keyboard key (Enter, Tab, Escape, ArrowDown, etc.).",
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
    {
        "name": "scroll",
        "description": "Scroll the page by a pixel amount (e.g. '500', '-300') or to a ref.",
        "input_schema": {
            "type": "object",
            "properties": {"amount": {"type": "string"}},
            "required": ["amount"],
        },
    },
    {
        "name": "wait",
        "description": "Wait N seconds for the page to settle, network to finish, or animations to complete.",
        "input_schema": {
            "type": "object",
            "properties": {"seconds": {"type": "number"}},
            "required": ["seconds"],
        },
    },
    {
        "name": "halt_for_human",
        "description": (
            "Stop and return control to the human. Use this when you see a "
            "captcha, OTP/2FA prompt, payment confirmation, or any "
            "irreversible / sensitive action that requires human judgement."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    },
    {
        "name": "task_complete",
        "description": "Signal the task is done. Include a short summary of what was accomplished.",
        "input_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
        "cache_control": {"type": "ephemeral"},
    },
]


SYSTEM_PROMPT_BASE = """You are a browser automation agent driving a real Chrome session through the pinchtab CLI.

On each turn you receive:
- A snapshot of interactive elements (each line is `refID:role "text"`)
- A screenshot of the current viewport
- The original task description

You must respond by calling exactly one tool. Briefly explain your reasoning in plain text before the tool call.

CRITICAL RULES:
1. NEVER attempt to bypass captchas, OTPs, anti-bot challenges, or any security control. If you see one, call `halt_for_human` with a clear description.
2. Before clicking any button that submits a form, completes a purchase, sends a message, deletes data, or makes any other irreversible change, call `halt_for_human` instead of clicking. Let the human decide.
3. Use ref IDs exactly as they appear in the snapshot. Never invent a ref.
4. If the page is still loading or the desired element isn't visible, use `wait` or `scroll`. Do not guess.
5. Stay strictly within the task. Ignore unrelated popups, ads, and side navigation unless dismissing them is required to make progress.
6. When the task is fully done, call `task_complete`.
"""


@dataclass
class AgentConfig:
    task_description: str
    start_url: str
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 2048
    max_steps: int = 30
    step_delay: float = 1.5
    snap_max_tokens: int = 4000
    screenshot_quality: int = 60
    halt_on_text: list[str] = field(default_factory=list)
    confirm_before_click_text: list[str] = field(default_factory=list)
    log_dir: str = "logs"


def load_config(path: str) -> AgentConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return AgentConfig(
        task_description=data["task"]["description"].strip(),
        start_url=data["task"]["start_url"],
        model=data.get("claude", {}).get("model", "claude-sonnet-4-6"),
        max_tokens=data.get("claude", {}).get("max_tokens", 2048),
        max_steps=data.get("agent", {}).get("max_steps", 30),
        step_delay=data.get("agent", {}).get("step_delay_seconds", 1.5),
        snap_max_tokens=data.get("agent", {}).get("snap_max_tokens", 4000),
        screenshot_quality=data.get("agent", {}).get("screenshot_quality", 60),
        halt_on_text=data.get("safety", {}).get("halt_on_text", []),
        confirm_before_click_text=data.get("safety", {}).get("confirm_before_click_text", []),
        log_dir=data.get("logging", {}).get("dir", "logs"),
    )


def detect_halt_pattern(snap_text: str, patterns: list[str]) -> str | None:
    lower = snap_text.lower()
    for p in patterns:
        if p.lower() in lower:
            return p
    return None


def find_element_line(snap_text: str, ref: str) -> str:
    match = re.search(rf"^{re.escape(ref)}:.*$", snap_text, re.MULTILINE)
    return match.group(0) if match else ref


def detect_image_media_type(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def build_user_turn(snap_text: str, screenshot_path: Path, step: int) -> dict:
    raw = screenshot_path.read_bytes()
    media_type = detect_image_media_type(raw)
    img_b64 = base64.standard_b64encode(raw).decode()
    return {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": img_b64,
                },
            },
            {
                "type": "text",
                "text": (
                    f"## Step {step}\n\n"
                    f"### Interactive elements\n```\n{snap_text}\n```\n\n"
                    f"Decide the next action."
                ),
            },
        ],
    }


def execute_tool(
    name: str,
    params: dict,
    cfg: AgentConfig,
    snap_text: str,
) -> tuple[str, bool]:
    """Run one tool call. Returns (result_text, should_stop)."""

    if name == "click":
        ref = params["ref"]
        elem_line = find_element_line(snap_text, ref)
        for pattern in cfg.confirm_before_click_text:
            if pattern.lower() in elem_line.lower():
                return (
                    f"HALT: would click '{elem_line}' but it matches "
                    f"confirm pattern '{pattern}'. Human confirm required.",
                    True,
                )

    try:
        if name == "click":
            out = pt.click(params["ref"])
        elif name == "type_text":
            out = pt.type_text(params["ref"], params["text"])
        elif name == "select_option":
            out = pt.select_option(params["ref"], params["value"])
        elif name == "press_key":
            out = pt.press(params["key"])
        elif name == "scroll":
            out = pt.scroll(str(params["amount"]))
        elif name == "wait":
            seconds = float(params["seconds"])
            seconds = min(seconds, 15.0)
            time.sleep(seconds)
            out = f"waited {seconds}s"
        elif name == "halt_for_human":
            return (f"HALT: {params['reason']}", True)
        elif name == "task_complete":
            return (f"DONE: {params['summary']}", True)
        else:
            return (f"ERROR: unknown tool '{name}'", False)
        return (out.strip() or "ok", False)
    except pt.PinchtabError as e:
        return (f"ERROR: {e}", False)


def run(cfg: AgentConfig) -> None:
    log_dir = Path(cfg.log_dir) / time.strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] logging to {log_dir}")

    try:
        h = pt.health()
        print(f"[setup] pinchtab health: {h.strip()[:120]}")
    except pt.PinchtabError as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[setup] navigating to {cfg.start_url}")
    pt.nav(cfg.start_url)
    time.sleep(2.5)

    client = Anthropic()
    system = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT_BASE + f"\n## Task\n{cfg.task_description}",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    messages: list[dict] = []

    for step in range(1, cfg.max_steps + 1):
        print(f"\n── step {step} ──")

        snap_text = pt.snap(
            interactive=True, compact=True, max_tokens=cfg.snap_max_tokens
        )
        ss_path = log_dir / f"step-{step:03d}.png"
        pt.screenshot(str(ss_path), quality=cfg.screenshot_quality)

        halted = detect_halt_pattern(snap_text, cfg.halt_on_text)
        if halted:
            print(f"[halt] safety pattern matched in page: '{halted}'")
            print("       resolve manually, then re-run to continue.")
            break

        messages.append(build_user_turn(snap_text, ss_path, step))

        response = client.messages.create(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            system=system,
            tools=TOOLS_SCHEMA,
            messages=messages,
        )

        (log_dir / f"step-{step:03d}.json").write_text(
            json.dumps(
                {"snap": snap_text, "response": response.model_dump()},
                indent=2,
                default=str,
            )
        )

        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"[think] {block.text.strip()}")

        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            print("[error] model returned no tool call — stopping")
            break

        tool_results = []
        should_stop = False
        for tu in tool_uses:
            print(f"[do]    {tu.name}({json.dumps(tu.input)})")
            result, stop = execute_tool(tu.name, tu.input, cfg, snap_text)
            print(f"        → {result[:200]}")
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result,
                }
            )
            should_stop = should_stop or stop

        messages.append({"role": "user", "content": tool_results})

        if response.usage:
            u = response.usage
            cached = getattr(u, "cache_read_input_tokens", 0) or 0
            print(
                f"[usage] in={u.input_tokens} out={u.output_tokens} cached_read={cached}"
            )

        if should_stop:
            break

        time.sleep(cfg.step_delay)
    else:
        print(f"\n[end] reached max_steps ({cfg.max_steps})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("config", help="Path to YAML config")
    args = p.parse_args()

    cfg = load_config(args.config)
    print(f"task : {cfg.task_description}")
    print(f"url  : {cfg.start_url}")
    print(f"model: {cfg.model}")
    run(cfg)


if __name__ == "__main__":
    main()
