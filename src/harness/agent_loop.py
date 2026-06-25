from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Protocol

from src.harness.actions import parse_agent_response
from src.harness.observations import DEFAULT_GOAL, build_observation, format_observation_for_llm
from src.world.grid_world import GridWorld, StepResult


class Policy(Protocol):
    def choose_action(self, observation: dict, system_prompt: str) -> tuple[str, str]:
        """Return (reasoning, action)."""


@dataclass
class RunLog:
    goal: str
    steps: list[dict] = field(default_factory=list)
    completed: bool = False

    def record(self, step: int, reasoning: str, action: str, result: StepResult, world: GridWorld) -> None:
        self.steps.append(
            {
                "step": step,
                "reasoning": reasoning,
                "action": action,
                "success": result.success,
                "feedback": result.message,
                "position": {"x": world.agent_x, "y": world.agent_y},
                "facing": world.facing.label,
                "inventory": list(world.inventory),
            }
        )
        if result.done and result.reward > 0:
            self.completed = True

    def to_json(self) -> str:
        return json.dumps(
            {
                "goal": self.goal,
                "completed": self.completed,
                "total_steps": len(self.steps),
                "steps": self.steps,
            },
            indent=2,
        )


SYSTEM_PROMPT = """You are an autonomous agent inside a 2D grid world.

You receive JSON observations describing:
- your position, facing direction, and inventory
- ego-centric sensors (front scan, left/right/behind cells)
- task status and remaining step budget

Choose ONE action per turn from valid_actions. Respond ONLY with JSON:
{
  "reasoning": "brief explanation of what you observe and why this action helps",
  "action": "move_forward"
}

Rules:
- You cannot walk through walls (#) or the red cube (R).
- pick_up only works when a collectible object is directly in front of you.
- use unlocks the exit door (E) when you have the golden key and stand in front of it.
- look does not change the world; it simply re-reads the observation.
- Prefer efficient paths; you have a limited step budget.
"""


@dataclass
class AgentHarness:
    world: GridWorld
    policy: Policy
    goal: str = DEFAULT_GOAL
    on_step: Callable[[int, str, str, StepResult, GridWorld], None] | None = None

    def run(self, max_steps: int | None = None) -> RunLog:
        log = RunLog(goal=self.goal)
        limit = max_steps or self.world.config.max_steps

        while self.world.steps < limit and not self.world.goal_completed:
            observation = build_observation(self.world, goal=self.goal)
            obs_text = format_observation_for_llm(observation)

            reasoning, action = self.policy.choose_action(observation, SYSTEM_PROMPT + "\n\n" + obs_text)
            result = self.world.apply_action(action)

            log.record(self.world.steps, reasoning, action, result, self.world)

            if self.on_step:
                self.on_step(self.world.steps, reasoning, action, result, self.world)

            if result.done:
                break

        return log
