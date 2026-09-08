# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

FIX_SUBOPTI_PROMPT = """\
You are a privileged teacher correcting one locally suboptimal decision in a
trajectory that successfully completed its Stage Goal.

The diagnosis already selected the earliest decision that can be replaced to
preserve success while avoiding later work. Treat that diagnosis as correct.

Produce only the replacement decision. All previous decisions remain fixed and
all following decisions will be replanned after this answer is executed.

Do not copy object names, positions, or assumptions from unrelated examples.
Do not mention hidden stage objectives in the answer. You may issue at most one
tool call per robot, but several robots may act in the same decision.

STAGE GOAL:
{stage_goal}

PERMANENT CONSTRAINTS:
{constraint_list}

TASK ATTRIBUTES:
{task_attributes}

AVAILABLE TOOLS:
{tools_api}

TRAJECTORY THROUGH THE SELECTED DECISION:
{trajectory}

SELECTED ORIGINAL DECISION:
{bad_answer}

WHY THIS DECISION IS SUBOPTIMAL:
{diagnosis}

Return exactly one JSON object and nothing else:

{{
  "say": "<message to the user, or an empty string>",
  "action": {{
    "<robot-name>": {{
      "name": "<tool-name>",
      "arguments": {{"<argument>": "<value>"}}
    }}
  }}
}}

The action may be empty only when a user-facing message is the correct next
decision. Do not use Markdown or a JSON code fence.
"""
