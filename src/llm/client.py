from __future__ import annotations

import json
import os
from typing import Any

from src.harness.actions import parse_agent_response


class ScriptedPolicy:
    """Deterministic policy for offline demos and tests."""

    def __init__(self, actions: list[str]):
        self.actions = actions
        self.index = 0

    def choose_action(self, observation: dict, system_prompt: str) -> tuple[str, str]:
        if self.index >= len(self.actions):
            action = "look"
            reasoning = "No scripted steps left; observing."
        else:
            action = self.actions[self.index]
            reasoning = f"Following scripted step {self.index + 1}: {action}."
        self.index += 1
        return reasoning, action


class LLMPolicy:
    """Calls an OpenAI-compatible or Anthropic chat API to choose actions."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
        self.model = model or os.getenv("LLM_MODEL", self._default_model())
        self.temperature = temperature

    def _default_model(self) -> str:
        if self.provider == "anthropic":
            return "claude-3-5-haiku-20241022"
        return "gpt-4o-mini"

    def choose_action(self, observation: dict, system_prompt: str) -> tuple[str, str]:
        user_message = json.dumps(observation, indent=2)
        raw = self._complete(system_prompt, user_message)
        return parse_agent_response(raw)

    def _complete(self, system_prompt: str, user_message: str) -> str:
        if self.provider == "anthropic":
            return self._anthropic(system_prompt, user_message)
        return self._openai(system_prompt, user_message)

    def _openai(self, system_prompt: str, user_message: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content or ""

    def _anthropic(self, system_prompt: str, user_message: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = client.messages.create(
            model=self.model,
            max_tokens=300,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        text_blocks = [block.text for block in response.content if block.type == "text"]
        return "\n".join(text_blocks)


def key_and_door_script() -> list[str]:
    """A hand-crafted path that solves the default map without an LLM."""
    return [
        "move_forward",
        "move_forward",
        "move_forward",
        "pick_up",
        "turn_left",
        "move_forward",
        "move_forward",
        "turn_right",
        "move_forward",
        "move_forward",
        "move_forward",
        "turn_right",
        "move_forward",
        "move_forward",
        "turn_left",
        "move_forward",
        "move_forward",
        "move_forward",
        "turn_right",
        "move_forward",
        "move_forward",
        "turn_left",
        "move_forward",
        "move_forward",
        "move_forward",
        "move_forward",
        "use",
    ]
