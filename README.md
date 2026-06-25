# LLM Agent in a Virtual World

A small harness that places an LLM-driven agent inside a 2D grid world. The agent receives structured JSON observations, chooses discrete actions, and works toward a goal: **find the golden key and unlock the exit door**.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Offline demo (no API key required)
PYTHONPATH=. python src/main.py

# Run with an LLM
export OPENAI_API_KEY=sk-...
PYTHONPATH=. python src/main.py --mode llm --provider openai

# Save a JSON run log
PYTHONPATH=. python src/main.py --log examples/sample_run.json
```

Run tests:

```bash
PYTHONPATH=. pytest tests/ -v
```

## Example output

The scripted demo completes the puzzle in 12 steps. See [`examples/sample_run.json`](examples/sample_run.json) for structured step-by-step logs (reasoning, action, feedback, position).

Initial map (`>` is the agent facing east):

```
# # # # # # # #
# > . . . K . #
# . . . . . . #
# . # # # # . #
# . . . . . . #
# . # # . . . #
# . . . . . E #
# # # # # # # #
```

Legend: `#` wall, `.` floor, `K` key, `R` red cube (blocks movement), `E` locked exit, `X` unlocked exit.

## Architecture

```
┌─────────────┐     observation JSON      ┌──────────────┐
│  GridWorld  │ ─────────────────────────▶│ AgentHarness │
│  (physics)  │◀───────────────────────── │  (loop)      │
└─────────────┘     action string         └──────┬───────┘
                                                 │
                                    ┌────────────┴────────────┐
                                    ▼                         ▼
                              LLMPolicy                 ScriptedPolicy
                           (OpenAI / Anthropic)         (offline demo)
```

| Layer | Role |
|-------|------|
| `src/world/` | Grid environment, collision rules, goal checks |
| `src/harness/` | Observation builder, action parsing, agent loop |
| `src/llm/` | LLM and scripted policies |
| `src/main.py` | CLI entry point |

## Design choices

**World.** A compact 8×8 grid with a few internal walls keeps the state space small enough for an LLM to reason about, while still requiring multi-step planning (navigate around walls, pick up key, backtrack toward exit).

**Observations.** Rather than dumping the full map, the agent gets **ego-centric sensors**: a front raycast (up to 3 cells), plus left/right/behind cell descriptions. This mirrors how real embodied agents receive local perception and forces the model to build spatial memory over steps. Task status (`has_key`, `exit_unlocked`) and step budget are included so the model can track progress.

**Action space.** Six discrete verbs (`move_forward`, `turn_left`, `turn_right`, `pick_up`, `use`, `look`) map cleanly to LLM outputs and keep parsing reliable. The harness expects JSON `{ "reasoning": "...", "action": "..." }` but also tolerates plain action names as a fallback.

**LLM integration.** Provider-agnostic policy interface with OpenAI (JSON mode) and Anthropic backends. A scripted policy verifies the harness end-to-end without burning API credits.

**What worked.** Structured JSON observations + a fixed action vocabulary produced consistent parsing. Ego-centric sensing reduced token noise compared to full-map dumps.

**What I'd improve next.** Add partial map memory in observations, a second task (e.g. move the red cube), and retry logic when the LLM returns an invalid action.

## CLI options

```
--mode {scripted,llm}   Policy to use (default: scripted)
--provider {openai,anthropic}
--model MODEL           Override default model
--max-steps N           Step budget (default: 40)
--log PATH              Write JSON run log
--quiet                 Suppress step-by-step console output
```

## License

MIT
