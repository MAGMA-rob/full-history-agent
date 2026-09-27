from __future__ import annotations

from copy import deepcopy
import json

from magma_core.protocol.agent import (
    AgentDecision, AgentError, AgentOutput, AgentRequest, ToolCall,
)
from full_history_agent.config import Settings
from .clients.harmony import HarmonyCommander
from .clients.magma import MagmaCommander
from .clients.llama import LlamaCommander
from .clients.llama.parsing import invalid_response
from .clients.messages import BatchedMessageCommander
from .clients.loading import detect_format
from .clients.qwen import QwenCommander


COMMANDER_TYPES = {
    "magma": MagmaCommander,
    "qwen": QwenCommander,
    "harmony": HarmonyCommander,
    "llama": LlamaCommander,
}


class Runtime:
    def __init__(self, settings: Settings) -> None:
        model_settings = settings.model
        model_format = detect_format(model_settings)
        if model_settings.history_max_messages is not None and model_format != "harmony":
            raise ValueError("history_max_messages is currently supported only for harmony")
        model_settings = model_settings.model_copy(update={"format": model_format})
        self.commander = COMMANDER_TYPES[model_format](model_settings)
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
            is_llama = isinstance(self.commander, LlamaCommander)
            if coached:
                prompts = self.commander._format_batch(message)
                answers = []
                exchanges = []
                for position, (entry, prompt) in enumerate(zip(inputs, prompts)):
                    correction = entry.extra_keys["coaching"]
                    answer = None
                    raw_output = json.dumps(correction)
                    diagnostic = None
                    try:
                        if is_llama and "error" in self.commander.input_elements[position]:
                            raise ValueError(self.commander.input_elements[position]["error"])
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
                        answer, raw_output = self.commander.format_coached_response(answer)
                        if is_llama and not answer.get("_llama_valid"):
                            diagnostic = answer.get("_llama_error")
                    except Exception as error:
                        if not is_llama and not isinstance(error, (TypeError, ValueError, KeyError)):
                            raise
                        answer = None
                        diagnostic = str(error)
                    answers.append(answer)
                    exchanges.append({"prompt": prompt, "raw_output": raw_output,
                                      "valid": answer is not None and diagnostic is None, "error": diagnostic})
            else:
                try:
                    answers = self.commander.process_batched_entry(message, inference_mode)
                    exchanges = self.commander.exchanges
                except Exception as error:
                    if not is_llama:
                        raise
                    answers = [invalid_response(error) for _ in candidates]
                    exchanges = [{"prompt": "", "raw_output": "", "valid": False, "error": str(error)}
                                 for _ in candidates]
                    self.commander.input_elements = [{} for _ in candidates]
            if len(answers) != len(candidates) or len(exchanges) != len(candidates):
                if not is_llama:
                    raise RuntimeError("Model batch size or recorded exchanges do not match candidates")
                diagnostic = "Model batch size or recorded exchanges do not match candidates"
                answers = [invalid_response(diagnostic) for _ in candidates]
                exchanges = [{"prompt": "", "raw_output": "", "valid": False, "error": diagnostic}
                             for _ in candidates]
                self.commander.input_elements = [{} for _ in candidates]
            validities = []
            for position, ((entry, index), answer, exchange) in enumerate(zip(candidates, answers, exchanges)):
                memory = deepcopy(entry.memory)
                steps = [{
                    "id": "step-0", "component": "commander", "origin": "coaching" if coached else "model",
                    "full_prompt": exchange["prompt"], "output_raw": exchange["raw_output"],
                    "input_elements": deepcopy(self.commander.input_elements[position]),
                }]
                if exchange.get("error"):
                    steps[0]["error"] = exchange["error"]
                if exchange.get("logging_error"):
                    steps[0]["logging_error"] = exchange["logging_error"]
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
                        raise ValueError(exchange.get("error") or "Model parser rejected its raw response")
                except (TypeError, ValueError) as error:
                    diagnostic = exchange.get("error") or str(error)
                    if is_llama:
                        steps[0]["error"] = diagnostic
                    validities.append(False)
                    results[(entry.id, index)] = AgentOutput(
                        request_id=request.request_id, source_id=entry.id, candidate_index=index,
                        status="error", memory=memory, internal_steps=steps,
                        error=AgentError(code="invalid_output", message=diagnostic, component="commander"),
                    )
                    continue
                try:
                    self.commander.update_memory_after_response(memory, answer)
                    action = {call.target_robot_name: {"name": call.name, "arguments": call.arguments}
                              for call in decision.tool_calls}
                    if is_llama or len(action) != len(decision.tool_calls):
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
                except Exception as error:
                    if not is_llama:
                        raise
                    validities.append(False)
                    steps[0]["error"] = str(error)
                    results[(entry.id, index)] = AgentOutput(
                        request_id=request.request_id, source_id=entry.id, candidate_index=index,
                        status="error", memory=deepcopy(entry.memory), internal_steps=steps,
                        error=AgentError(code="invalid_output", message=str(error) or "Llama memory update failed", component="commander"),
                    )
                    continue
                validities.append(True)
            if not coached:
                self.commander.update_prompt_log_validity(validities)
        return [results[(entry.id, index)] for entry in request.inputs for index in range(entry.num_outputs)]
