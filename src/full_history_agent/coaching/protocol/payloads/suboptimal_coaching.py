# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import json
from typing import Any, ClassVar, Dict, List, Optional

from magma_core.protocol.payload.base_payload import BasePayload
from magma_core.protocol.payload.coaching_common import format_stage_trajectory

from ..prompts.sub_optimal.fix_non_optimal import FIX_SUBOPTI_PROMPT


class SuboptimalFixPayload(BasePayload):
    prompt_template: ClassVar[str] = FIX_SUBOPTI_PROMPT
    debug_log: ClassVar[bool] = True

    def __init__(
        self,
        stage_goal: str,
        constraint_list: List[str],
        task_attributes: Dict[str, Any],
        tools_api: List[Dict[str, Any]],
        trajectory: List[Dict[str, Any]],
        bad_answer: Dict[str, Any],
        diagnosis: Dict[str, Any],
        id: int,
        max_tokens: int = 5000,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(id, max_tokens, model)
        self.stage_goal = stage_goal
        self.constraint_list = list(constraint_list)
        self.task_attributes = task_attributes
        self.tools_api = tools_api
        self.trajectory_steps = trajectory
        self.trajectory = format_stage_trajectory(trajectory)
        self.bad_answer = bad_answer
        self.diagnosis = diagnosis

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_goal": self.stage_goal,
            "constraint_list": json.dumps(self.constraint_list, ensure_ascii=False),
            "task_attributes": json.dumps(
                self.task_attributes,
                ensure_ascii=False,
                default=str,
            ),
            "tools_api": json.dumps(self.tools_api, ensure_ascii=False, default=str),
            "trajectory": self.trajectory,
            "bad_answer": json.dumps(self.bad_answer, ensure_ascii=False, default=str),
            "diagnosis": json.dumps(self.diagnosis, ensure_ascii=False),
        }

    def to_human_dict(self) -> Dict[str, Any]:
        return {
            "stage_goal": self.stage_goal,
            "constraint_list": self.constraint_list,
            "task_attributes": self.task_attributes,
            "tools_api": self.tools_api,
            "trajectory": self.trajectory,
            "trajectory_steps": self.trajectory_steps,
            "bad_answer": self.bad_answer,
            "diagnosis": self.diagnosis,
        }
