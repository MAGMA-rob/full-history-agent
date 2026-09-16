from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence

from openai_harmony import Role

from .tools import (
    ToolCatalog,
    actions_to_commander_format,
    parse_tool_call,
)


def parse_completion(
    encoding: Any,
    completion_ids: Sequence[int],
    terminal_token_ids: Sequence[int],
    catalog: ToolCatalog,
) -> Dict[str, Any]:
    trimmed = list(completion_ids)
    terminal_ids = set(terminal_token_ids)
    while trimmed and trimmed[-1] in terminal_ids:
        trimmed.pop()

    try:
        entries = encoding.parse_messages_from_completion_tokens(
            trimmed,
            Role.ASSISTANT,
            strict=False,
        )
        return messages_to_response(entries, catalog)
    except Exception:
        normalized = normalize_completion_text(encoding.decode(trimmed))
        try:
            normalized_ids = encoding.encode(normalized, allowed_special="all")
            entries = encoding.parse_messages_from_completion_tokens(
                normalized_ids,
                Role.ASSISTANT,
                strict=False,
            )
            return messages_to_response(entries, catalog)
        except Exception as error:
            return parse_completion_fallback(encoding.decode(trimmed), catalog, error)


def messages_to_response(entries: Sequence[Any], catalog: ToolCatalog) -> Dict[str, Any]:
    analysis: List[str] = []
    final_parts: List[str] = []
    commentary_parts: List[str] = []
    actions: List[Dict[str, Any]] = []
    clarification: str | None = None

    for entry in entries:
        entry_dict = entry.to_dict()
        channel = entry_dict.get("channel")
        recipient = entry_dict.get("recipient")
        text = content_to_text(entry_dict.get("content", "")).strip()
        if channel == "analysis":
            if text:
                analysis.append(text)
            continue
        if recipient:
            kind, value = parse_tool_call(recipient, text, catalog)
            if kind == "clarification":
                clarification = value
            else:
                actions.extend(value)
            continue
        if channel == "final":
            if text:
                final_parts.append(text)
        elif channel == "commentary" and text:
            commentary_parts.append(text)

    if actions:
        return response(
            say=(
                "\n".join(final_parts).strip()
                or clarification
                or "\n".join(commentary_parts).strip()
            ),
            action=actions_to_commander_format(actions),
            kind="tool_call",
            analysis=analysis,
        )
    if clarification is not None:
        return response(
            say="\n".join(final_parts).strip() or clarification,
            action={},
            kind="clarification",
            analysis=analysis,
        )
    if final_parts:
        return response(
            say="\n".join(final_parts).strip(),
            action={},
            kind="final",
            analysis=[],
        )
    if commentary_parts:
        raise ValueError("Commentary without a function recipient is not a valid response")
    raise ValueError("Harmony completion contains no final response or function call")


def response(
    *,
    say: str,
    action: Any,
    kind: str,
    analysis: Sequence[str],
    valid: bool = True,
) -> Dict[str, Any]:
    return {
        "think": "",
        "say": say,
        "action": action,
        "_gpt_oss_kind": kind,
        "_gpt_oss_analysis": list(analysis),
        "_gpt_oss_valid": valid,
    }


def parse_completion_fallback(
    text: str,
    catalog: ToolCatalog,
    error: Exception,
) -> Dict[str, Any]:
    final_text = extract_channel_text(text, "final")
    calls = extract_raw_tool_calls(text)
    try:
        actions: List[Dict[str, Any]] = []
        clarification: str | None = None
        for recipient, arguments in calls:
            kind, value = parse_tool_call(recipient, arguments, catalog)
            if kind == "clarification":
                clarification = value
            else:
                actions.extend(value)

        analysis = [extract_channel_text(text, "analysis")]
        analysis = [item for item in analysis if item]
        if actions:
            return response(
                say=final_text or clarification or "",
                action=actions_to_commander_format(actions),
                kind="tool_call",
                analysis=analysis,
            )
        if clarification is not None:
            return response(
                say=final_text or clarification,
                action={},
                kind="clarification",
                analysis=analysis,
            )
        if final_text:
            return response(say=final_text, action={}, kind="final", analysis=[])
    except ValueError:
        pass

    return {
        "think": "",
        "say": "",
        "action": text.strip() or str(error),
        "_gpt_oss_kind": "invalid",
        "_gpt_oss_analysis": [],
        "_gpt_oss_valid": False,
    }


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(content_to_text(item) for item in content)
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        if "content" in content:
            return content_to_text(content["content"])
        return json.dumps(content, ensure_ascii=True)
    if content is None:
        return ""
    return str(content)


def extract_channel_text(text: str, channel: str) -> str:
    pattern = re.compile(
        rf"<\|channel\|>{re.escape(channel)}.*?<\|message\|>(.*?)(?=<\|end\|>|<\|return\|>|<\|call\|>|<\|start\|>|$)",
        re.DOTALL,
    )
    return "\n".join(match.group(1).strip() for match in pattern.finditer(text)).strip()


def extract_raw_tool_calls(text: str) -> List[tuple[str, str]]:
    pattern = re.compile(
        r"to=([A-Za-z0-9_.-]+).*?<\|message\|>(.*?)(?=<\|call\|>|<\|return\|>|<\|end\|>|<\|start\|>|$)",
        re.DOTALL,
    )
    return [(match.group(1), match.group(2).strip()) for match in pattern.finditer(text)]


def normalize_completion_text(text: str) -> str:
    text = re.sub(
        r"(<\|start\|>assistant)(<\|channel\|>[A-Za-z0-9_-]+)\s+to=([A-Za-z0-9_.-]+)",
        r"\1 to=\3\2",
        text,
    )
    return re.sub(r"(?:<\|constrain\|>\s*)+json", "json", text)
