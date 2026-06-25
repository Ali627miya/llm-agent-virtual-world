from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.harness.agent_loop import AgentHarness
from src.llm.client import LLMPolicy, ScriptedPolicy, key_and_door_script
from src.world.grid_world import GridWorld


def print_step(step: int, reasoning: str, action: str, result, world: GridWorld) -> None:
    print(f"\n--- Step {step} ---")
    print(f"Reasoning: {reasoning}")
    print(f"Action: {action}")
    print(f"Result: {result.message}")
    print(world.render_ascii())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an LLM agent in a 2D grid world.")
    parser.add_argument(
        "--mode",
        choices=["llm", "scripted"],
        default="scripted",
        help="Use a real LLM policy or the built-in scripted demo.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default=None,
        help="LLM provider when --mode=llm.",
    )
    parser.add_argument("--model", default=None, help="Override default model name.")
    parser.add_argument("--max-steps", type=int, default=40, help="Step budget.")
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Optional path to write JSON run log.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress step-by-step output.")
    args = parser.parse_args()

    world = GridWorld()
    world.config.max_steps = args.max_steps

    if args.mode == "llm":
        policy = LLMPolicy(provider=args.provider, model=args.model)
    else:
        policy = ScriptedPolicy(key_and_door_script())

    harness = AgentHarness(
        world=world,
        policy=policy,
        on_step=None if args.quiet else print_step,
    )

    print("Initial world:")
    print(world.render_ascii())

    log = harness.run()

    print("\n=== Run Summary ===")
    print(f"Completed goal: {log.completed or world.goal_completed}")
    print(f"Steps taken: {len(log.steps)}")
    print(f"Final inventory: {world.inventory}")

    if args.log:
        args.log.write_text(log.to_json())
        print(f"Wrote log to {args.log}")

    return 0 if world.goal_completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
