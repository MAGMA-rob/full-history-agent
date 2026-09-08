# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

FIX_FAILURE_PROMPT = """\
You are a privileged teacher.

A previous trajectory analysis identified one decision as the earliest causal decision responsible for preventing the completion of the Stage Goal.

Your task is to produce a corrected version of THIS decision using the privileged diagnosis provided below.

To do so, you must strictly follow the failure diagnosis and propose a counterfactual decision to this step.

DO NOT:
- Redesign the full plan
- Add extra steps
- Justify using hidden objectives
- Modify previous steps
- Mention "stage objective". You must place yourself at the agent, using inputs to make the decision.

If you need to select a tool call, you can only select ONE per robot.

────────────
CORRECTION PRINCIPLE
────────────

Treat the diagnosis as correct.

Do NOT question which decision should be corrected. Your only task is to generate a better decision for this specific step following the analysis and what tools you can use.

Your correction should:
- remain consistent with the current observations
- preserve all previous decisions
- maximize future progress toward the Stage Goal
- exploit valid recovery opportunities when they exist
- avoid irreversible mistakes

Your correction must modify ONLY the selected decision. Assume every previous decision is fixed.

────────────
Task
────────────

Stage Objective:
{stage_goal}

Constraints:
{constraint_list}

Task Attributes:
{task_attributes}

────────────
Tools
────────────

{tools_api}

────────────
Trajectory
────────────

{trajectory}

────────────
Decision to correct
────────────

{bad_answer}

Why it was rejected?
{coach_diagnose}

────────────
REQUIRED OUTPUT FORMAT (STRICT)
────────────

Return exactly one JSON array and nothing else.
Do not use Markdown or wrap the array in a code fence.
The first output character must be [ and the last output character must be ].
Use this format:

[{{"robot": "<robot-name>", "name": "<tool-name>", "arguments": {{"<argument>": "<value>"}}}}]
"""
