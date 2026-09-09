"""Specialized HR correction. Uses external coaching backends only."""
import json
from copy import deepcopy
from magma_core.workers.base import PayloadWorker
from magma_core.protocol.agent import AgentDecision, ToolCall
from magma_core.protocol.agent_coaching import CoachingProposal, SpecializedCoachingRequest, SpecializedCoachingResponse
from magma_core.protocol.payload.generic_coaching import FormatFixPayload
from .protocol.payloads.failure_coaching import FailureFixPayload
from .protocol.payloads.suboptimal_coaching import SuboptimalFixPayload


class CoachingService:
    def __init__(self, worker: PayloadWorker) -> None:
        self.worker = worker

    def process(self, request: SpecializedCoachingRequest) -> SpecializedCoachingResponse:
        target = next(step for step in request.trajectory if step.id == request.target_step_id)
        trajectory = []
        reached_target = False
        for step in request.trajectory:
            if step.execution:
                evidence = deepcopy(step.execution)
                if reached_target and request.kind == 'failure':
                    evidence.pop('answer', None)
                    evidence.pop('answer_data', None)
                    evidence.pop('execution_results', None)
                    trajectory.append(evidence)
                    break
                trajectory.append(evidence)
            if step.id == target.id:
                reached_target = True
        action = {} if target.output is None else {
            call.target_robot_name: {'name': call.name, 'arguments': call.arguments}
            for call in target.output.tool_calls
        }
        rejected = {'say': '' if target.output is None else target.output.say, 'action': action}
        goal = request.stage.goal
        if request.active_errors:
            goal += '\n\nActive environment errors (Not known by the agent):\n' + '\n'.join(
                '- ' + error.description for error in request.active_errors)
        common = dict(stage_goal=goal, task_attributes=target.input.attributes,
                      trajectory=trajectory, id=target.id)
        diagnosis = request.diagnosis
        if request.kind == 'format':
            payload = FormatFixPayload(
                **common, rejected_answer={'internal_steps': target.internal_steps},
                rejection_reason=target.error.message if target.error else 'Invalid model format',
                desired_output_format={'say': 'user message', 'action': {'robot': {'name': 'tool', 'arguments': {}}}},
                output_format_rules='Return say or actions, never both. Each action must target a robot.',
                task_coaching_hint=request.stage.hint,
            )
        else:
            common.update(constraint_list=target.input.memory.get('memory_list', []),
                          tools_api=target.input.tools, bad_answer=rejected)
            if request.kind == 'failure':
                payload = FailureFixPayload(**common, coach_diagnose={
                    'reason': diagnosis.explanation if diagnosis else '',
                    'expected_decision': diagnosis.expected_decision or '' if diagnosis else '',
                })
            else:
                payload = SuboptimalFixPayload(**common, diagnosis={'reason': diagnosis.explanation if diagnosis else ''})
        try:
            _, raw = self.worker.submit(payload, callback=None).result()
            parsed = json.loads(raw)
            if request.kind == 'failure':
                if not isinstance(parsed, list) or not parsed:
                    raise ValueError('Failure correction must be a nonempty list of tools')
                decision = AgentDecision(tool_calls=[ToolCall(
                    name=call['name'], arguments=call['arguments'], target_robot_name=call['robot'],
                ) for call in parsed])
            else:
                if not isinstance(parsed, dict):
                    raise ValueError('Correction must be an object')
                decision = AgentDecision(say=parsed.get('say', ''), tool_calls=[
                    ToolCall(name=call['name'], arguments=call.get('arguments', {}), target_robot_name=robot)
                    for robot, call in parsed.get('action', {}).items()
                ])
            return SpecializedCoachingResponse(
                request_id=request.request_id, status='corrected', proposals=[CoachingProposal(
                    reinjection_point_id=point.id, extra_keys={
                        'kind': 'replace_decision', 'decision': decision.model_dump(mode='json'),
                    },
                ) for point in request.reinjection_points],
            )
        except (ValueError, TypeError, KeyError) as error:
            return SpecializedCoachingResponse(request_id=request.request_id, status='abandoned', reason=str(error))
        except Exception as error:
            return SpecializedCoachingResponse(request_id=request.request_id, status='error', reason=str(error))
