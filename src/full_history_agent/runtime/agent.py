from __future__ import annotations

from copy import deepcopy
import json

from magma_core.protocol.agent import (
    AgentDecision, AgentError, AgentOutput, AgentRequest, ToolCall,
)
from full_history_agent.config import Settings
from .clients.commander.messages import BatchedMessageCommander
from .clients.loading import detect_format


class Runtime:
    def __init__(self, settings: Settings) -> None:
        model_settings = settings.model
        model_format = detect_format(model_settings)
        model_settings = model_settings.model_copy(update={"format": model_format})
        if model_format == "qwen":
            from .clients.commander.qwen_model import QwenCommander
            self.commander = QwenCommander(model_settings)
        elif model_format == "gpt-oss":
            from .clients.commander.gpt_model import OSSCommander
            self.commander = OSSCommander(model_settings)
        else:
            from .clients.commander.magma_model import MagmaCommander
            self.commander = MagmaCommander(model_settings)
        self.commander.set_prompt_log_dir(settings.prompt_log_dir)

    def validate_request(self, request: AgentRequest) -> None:
        for entry in request.inputs:
            history = entry.memory.get("history", [])
            if not isinstance(history, list) or any(not isinstance(item, dict) for item in history):
                raise ValueError("memory.history must be a list of objects")
            rules = entry.memory.get("memory_list", [])
            if not isinstance(rules, list) or any(not isinstance(item, str) for item in rules):
                raise ValueError("memory.memory_list must be a list of strings")
            if type(entry.extra_keys.get("inference_mode", False)) is not bool:
                raise ValueError("extra_keys.inference_mode must be a boolean")
            if not isinstance(entry.memory.get("summary", ""), str):
                raise ValueError("memory.summary must be a string")

    def process(self, request: AgentRequest) -> list[AgentOutput]:
        results: dict[tuple[int, int], AgentOutput] = {}
        for inference_mode, coached in ((False, False), (True, False), (False, True), (True, True)):
            entries = [entry.model_copy(deep=True) for entry in request.inputs
                       if entry.extra_keys.get("inference_mode", False) == inference_mode
                       and ("coaching" in entry.extra_keys) == coached]
            if not entries:
                continue
            candidates = [(entry, index) for entry in entries for index in range(entry.num_outputs)]
            if not candidates:
                continue
            inputs = [entry for entry, _ in candidates]
            message = BatchedMessageCommander(
                memory=[deepcopy(entry.memory) for entry in inputs],
                attributes=[deepcopy(entry.attributes) for entry in inputs],
                history=[deepcopy(entry.memory.get("history", [])) for entry in inputs],
                function=[deepcopy(entry.tools) for entry in inputs],
                instruction=[entry.instruction.content for entry in inputs],
                instruction_role=[
                    "USER" if entry.instruction.type == "user" else "SYSTEM"
                    for entry in inputs
                ],
            )
            self.commander.exchanges.clear()
            if coached:
                prompts = self.commander._format_batch(message)
                answers = []
                exchanges = []
                for entry, prompt in zip(inputs, prompts):
                    correction = entry.extra_keys["coaching"]
                    answer = None
                    try:
                        if not isinstance(correction, dict):
                            raise ValueError("coaching must be an object")
                        if correction.get("kind") == "replace_say":
                            from magma_core.protocol.agent_coaching import ReplaceSay
                            decision = AgentDecision(say=ReplaceSay.model_validate(correction).text)
                        elif correction.get("kind") == "replace_decision":
                            decision = AgentDecision.model_validate(correction["decision"])
                        else:
                            raise ValueError("Unknown HR coaching correction")
                        answer = {"say": decision.say, "action": [
                            {call.target_robot_name: {"name": call.name, "arguments": call.arguments}}
                            for call in decision.tool_calls
                        ]}
                    except (TypeError, ValueError, KeyError):
                        pass
                    answers.append(answer)
                    exchanges.append({"prompt": prompt, "raw_output": json.dumps(answer if answer is not None else correction),
                                      "valid": answer is not None})
            else:
                answers = self.commander.process_batched_entry(message, inference_mode)
                exchanges = self.commander.exchanges
            if len(answers) != len(candidates) or len(exchanges) != len(candidates):
                raise RuntimeError("Model batch size or recorded exchanges do not match candidates")
            validities = []
            for position, ((entry, index), answer, exchange) in enumerate(zip(candidates, answers, exchanges)):
                memory = deepcopy(entry.memory)
                steps = [{
                    "id": "step-0", "component": "commander", "origin": "coaching" if coached else "model",
                    "full_prompt": exchange["prompt"], "output_raw": exchange["raw_output"],
                    "input_elements": deepcopy(self.commander.input_elements[position]),
                }]
                try:
                    if isinstance(answer, str):
                        answer = json.loads(answer)
                    if not isinstance(answer, dict) or not ({"say", "action"} & answer.keys()):
                        raise ValueError("Model output must contain say or action")
                    action = answer.get("action", {})
                    if isinstance(action, str):
                        action = json.loads(action)
                    actions = action if isinstance(action, list) else [action]
                    calls = []
                    for item in actions:
                        if item == {}:
                            continue
                        if not isinstance(item, dict):
                            raise ValueError("Actions must be JSON objects")
                        if "name" in item:
                            calls.append(ToolCall(
                                name=item["name"], arguments=item.get("arguments", {}),
                                target_robot_name=item.get("target_robot_name", item.get("target_robot")),
                            ))
                        else:
                            for robot, call in item.items():
                                if not isinstance(call, dict):
                                    raise ValueError("Robot action must be an object")
                                calls.append(ToolCall(
                                    name=call.get("name"), arguments=call.get("arguments", {}),
                                    target_robot_name=robot,
                                ))
                    decision = AgentDecision(say=answer.get("say", ""), tool_calls=calls)
                    if exchange.get("valid") is False:
                        raise ValueError("Model parser rejected its raw response")
                except (TypeError, ValueError) as error:
                    validities.append(False)
                    results[(entry.id, index)] = AgentOutput(
                        request_id=request.request_id, source_id=entry.id, candidate_index=index,
                        status="error", memory=memory, internal_steps=steps,
                        error=AgentError(code="invalid_output", message=str(error), component="commander"),
                    )
                    continue
                validities.append(True)
                action = {call.target_robot_name: {"name": call.name, "arguments": call.arguments}
                          for call in decision.tool_calls}
                if len(action) != len(decision.tool_calls):
                    action = [{call.target_robot_name: {"name": call.name, "arguments": call.arguments}}
                              for call in decision.tool_calls]
                memory.setdefault("history", []).extend([
                    {"author": "USER" if entry.instruction.type == "user" else "SYSTEM",
                     "content": entry.instruction.content},
                    {"author": "MODEL", "content": json.dumps(
                        {"say": decision.say, "action": action}, ensure_ascii=False)},
                ])
                results[(entry.id, index)] = AgentOutput(
                    request_id=request.request_id, source_id=entry.id, candidate_index=index,
                    status="completed", memory=memory, internal_steps=steps, output=decision,
                )
            if not coached:
                self.commander.update_prompt_log_validity(validities)
        return [results[(entry.id, index)] for entry in request.inputs for index in range(entry.num_outputs)]
