from __future__ import annotations

VALID_ACTIONS = [
    "move_forward",
    "turn_left",
    "turn_right",
    "pick_up",
    "use",
    "look",
]


def parse_agent_response(raw: str) -> tuple[str, str]:
    """Parse LLM output into (reasoning, action).

    Accepts plain action names or JSON with reasoning/action fields.
    """
    text = raw.strip()

    if text.startswith("{"):
        import json

        try:
            payload = json.loads(text)
            reasoning = str(payload.get("reasoning", "")).strip()
            action = str(payload.get("action", "")).strip().lower()
            if action in VALID_ACTIONS:
                return reasoning, action
        except json.JSONDecodeError:
            pass

    lowered = text.lower()
    for action in VALID_ACTIONS:
        if action in lowered:
            reasoning = text.replace(action, "").strip(" `\"'\n-:")
            return reasoning, action

    raise ValueError(
        f"Could not parse action from response. Expected one of: {VALID_ACTIONS}"
    )
