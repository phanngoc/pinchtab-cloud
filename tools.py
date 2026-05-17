"""Thin Python wrapper over the pinchtab CLI.

All functions return raw stdout (string). The agent treats this opaquely and
passes it back to Claude as tool results, so we don't need to parse JSON here.
"""
from __future__ import annotations

import shutil
import subprocess


class PinchtabError(RuntimeError):
    pass


def _run(args: list[str], timeout: float = 30.0) -> str:
    if shutil.which("pinchtab") is None:
        raise PinchtabError(
            "pinchtab CLI not found in PATH. Install it and start the server "
            "in another terminal: `pinchtab` (default port 9867)."
        )
    result = subprocess.run(
        ["pinchtab", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise PinchtabError(
            f"pinchtab {' '.join(args)} → exit {result.returncode}\n"
            f"stderr: {result.stderr.strip()}"
        )
    return result.stdout


def health() -> str:
    return _run(["health"], timeout=5)


def nav(url: str) -> str:
    return _run(["nav", url], timeout=60)


def snap(interactive: bool = True, compact: bool = True, max_tokens: int | None = None) -> str:
    args = ["snap"]
    if interactive:
        args.append("-i")
    if compact:
        args.append("-c")
    if max_tokens:
        args += ["--max-tokens", str(max_tokens)]
    return _run(args)


def screenshot(out_path: str, quality: int = 60) -> str:
    return _run(["ss", "-o", out_path, "-q", str(quality)])


def click(ref: str) -> str:
    return _run(["click", ref])


def type_text(ref: str, text: str) -> str:
    return _run(["type", ref, text])


def fill(ref: str, text: str) -> str:
    return _run(["fill", ref, text])


def press(key: str) -> str:
    return _run(["press", key])


def select_option(ref: str, value: str) -> str:
    return _run(["select", ref, value])


def scroll(amount: str) -> str:
    return _run(["scroll", amount])


def page_text() -> str:
    return _run(["text"])
