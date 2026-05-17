"""Tests for the claude CLI provider — prompt builder and response parser.

The subprocess itself isn't run in tests (would require a live claude binary
+ subscription). The risky logic — prompt assembly, response parsing — is
covered here.
"""
import json

import pytest

from backend.llm_cli import (
    _Block,
    build_cli_prompt,
    is_operator,
    parse_cli_response,
)


# ---- prompt builder ----


def test_prompt_includes_system_text():
    p = build_cli_prompt(
        system=[{"text": "You are a browser agent."}],
        tools=[],
        messages=[],
    )
    assert "You are a browser agent." in p
    assert "# System Instructions" in p


def test_prompt_lists_tools_with_args():
    p = build_cli_prompt(
        system=[],
        tools=[
            {
                "name": "click",
                "description": "Click an element by ref.",
                "input_schema": {
                    "type": "object",
                    "properties": {"ref": {"type": "string"}},
                    "required": ["ref"],
                },
            }
        ],
        messages=[],
    )
    assert "**click**" in p
    assert "ref: string" in p
    assert "Click an element by ref." in p


def test_prompt_renders_user_messages():
    p = build_cli_prompt(
        system=[],
        tools=[],
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "current snapshot here"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will scroll."},
                    {"type": "tool_use", "name": "scroll", "input": {"amount": "500"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": "scrolled successfully"}
                ],
            },
        ],
    )
    assert "current snapshot here" in p
    assert "I will scroll." in p
    assert "<tool>scroll</tool>" in p
    assert '"amount": "500"' in p or '"amount":"500"' in p
    assert "scrolled successfully" in p


def test_prompt_image_blocks_become_placeholder():
    p = build_cli_prompt(
        system=[],
        tools=[],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "data": "ABCD"}},
                    {"type": "text", "text": "what is on the page?"},
                ],
            }
        ],
    )
    assert "[screenshot attached" in p
    assert "what is on the page?" in p
    # The raw base64 must NOT leak into the prompt (waste of tokens).
    assert "ABCD" not in p


def test_prompt_has_output_format_section():
    p = build_cli_prompt(system=[], tools=[], messages=[])
    assert "<thought>" in p
    assert "<tool>" in p
    assert "<args>" in p
    assert "Exactly one tool call per response" in p


# ---- response parser ----


def test_parse_well_formed_response():
    text = """
<thought>I need to scroll to load more posts.</thought>
<tool>scroll</tool>
<args>{"amount": "800"}</args>
""".strip()
    r = parse_cli_response(text)
    assert len(r.content) == 2
    assert r.content[0].type == "text"
    assert "scroll to load more" in r.content[0].text
    assert r.content[1].type == "tool_use"
    assert r.content[1].name == "scroll"
    assert r.content[1].input == {"amount": "800"}
    assert r.content[1].id.startswith("toolu_cli_")


def test_parse_tolerates_fenced_json():
    text = """
<thought>Done.</thought>
<tool>task_complete</tool>
<args>
```json
{"summary": "found 3 headlines"}
```
</args>
""".strip()
    r = parse_cli_response(text)
    tool_uses = [b for b in r.content if b.type == "tool_use"]
    assert len(tool_uses) == 1
    assert tool_uses[0].name == "task_complete"
    assert tool_uses[0].input == {"summary": "found 3 headlines"}


def test_parse_tolerates_single_quotes_in_args():
    text = """
<thought>x</thought>
<tool>click</tool>
<args>{'ref': 'e7'}</args>
""".strip()
    r = parse_cli_response(text)
    tu = [b for b in r.content if b.type == "tool_use"][0]
    assert tu.input == {"ref": "e7"}


def test_parse_no_tags_falls_back_to_text_block():
    """Malformed model output should produce a text block (so the runner sees
    'no tool_use' and ends the loop with a clear error, not a crash)."""
    text = "I'm not sure what to do."
    r = parse_cli_response(text)
    assert len(r.content) == 1
    assert r.content[0].type == "text"
    assert "not sure what to do" in r.content[0].text


def test_parse_unparseable_args_produces_empty_input():
    text = """
<thought>x</thought>
<tool>scroll</tool>
<args>this is not json at all { broken</args>
""".strip()
    r = parse_cli_response(text)
    tool_uses = [b for b in r.content if b.type == "tool_use"]
    assert len(tool_uses) == 1
    assert tool_uses[0].input == {}


def test_response_model_dump_shape():
    text = "<thought>t</thought><tool>x</tool><args>{}</args>"
    r = parse_cli_response(text)
    dumped = r.model_dump()
    assert "content" in dumped
    assert dumped["content"][0]["type"] == "text"
    assert dumped["content"][1]["type"] == "tool_use"
    assert dumped["content"][1]["name"] == "x"


# ---- operator gate ----


def test_is_operator_requires_env(monkeypatch):
    monkeypatch.delenv("OPERATOR_EMAIL", raising=False)
    # Also clear from pydantic settings cache so the empty env var wins.
    from backend.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setattr(get_settings(), "operator_email", "")
    assert is_operator("anyone@example.com") is False


def test_is_operator_matches_case_insensitive(monkeypatch):
    monkeypatch.setenv("OPERATOR_EMAIL", "Phan.Ngoc@Example.com")
    # Clear cache so settings re-read the env var.
    from backend.config import get_settings
    get_settings.cache_clear()
    assert is_operator("phan.ngoc@example.com") is True
    assert is_operator("PHAN.NGOC@EXAMPLE.COM") is True
    assert is_operator("someone-else@example.com") is False
    get_settings.cache_clear()
