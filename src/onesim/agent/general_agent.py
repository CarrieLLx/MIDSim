import asyncio
from .base import AgentBase
from onesim.events import Event, DataEvent, DataResponseEvent, DataUpdateEvent, DataUpdateResponseEvent
from typing import Dict, List, Callable, Optional, Any
import json
import os
import uuid
import time
from asyncio import Future
from loguru import logger
from onesim.models.core.message import Message
from onesim.models import (
    parse_llm_json_response,
)
from onesim.profile import AgentProfile
from onesim.memory import *
from onesim.events import *
from onesim.relationship import RelationshipManager, Relationship
import inspect
from onesim.planning.base import PlanningBase
from onesim.distribution.node import get_node
from onesim.distribution import grpc_impl
from onesim.distribution.node import NodeRole
from onesim.distribution.distributed_lock import  get_lock
from onesim.utils.work_graph import WorkGraph
from datetime import datetime

from onesim.agent.backbone import is_llama_backbone
from pathlib import Path
from onesim.utils.prompt_resources import load_prompt
from onesim.utils.recommendation_validation import (
    RecommendationFormatError,
    validate_recommendation,
)
from onesim.utils.llm_validation import (
    LLMRequestError, validate_memory, validate_reaction, response_metadata, recommendation_candidates,
)
from onesim.utils.reaction_failure import (
    SkippedReaction, SkippedMemory, SkippedRecommendation, skipped_social_reaction, register_output_failure,
)
from onesim.utils.model_retry import model_attempt_limit, single_model_attempt, model_audit_scope, ModelRequestSkipped


class GeneralAgent(AgentBase):
    _reaction_semaphore = asyncio.Semaphore(45)

    async def get_prompt_profile(self, observation: str) -> str:
        return self.profile.get_profile_str(include_private=False) if self.profile else "No profile information provided."

    def _mark_llm_failure(self, failure: dict) -> None:
        # Mark before raising: platform handlers may catch exceptions themselves.
        self._llm_failure = getattr(self, '_llm_failure', None) or failure
        env = getattr(self, 'env', None)
        if env is not None:
            env._llm_failure = getattr(env, '_llm_failure', None) or failure
            pause = getattr(env, '_pause_signal', None)
            if pause is not None:
                pause.set()

    async def _validated_model_request(
        self, prompt_text, *, kind, action, instruction, observation, memory='',
        validator, max_tokens=None, error_class=LLMRequestError,
    ):
        """Retry only the model request; never replay actions or regenerate inputs."""
        if getattr(getattr(self, 'env', None), '_llm_failure', None):
            raise LLMRequestError('The environment is stopping after a failed model request.')
        retries = model_attempt_limit() - 1
        request_id, last_error = str(uuid.uuid4()), None
        for attempt in range(1, retries + 2):
            attempt_prompt = prompt_text
            if last_error:
                attempt_prompt += (
                    f'\nOutput validation failed: {last_error}\n'
                    'Return a complete JSON object in the original schema. '
                    'Keep the original decision criteria and input IDs. '
                    'A valid decision to take no action is allowed; do not change it merely to retry.'
                )
            prompt = self.model.format(
                Message('system', self.sys_prompt, role='system'),
                Message('user', attempt_prompt, role='user'),
            )
            response, parse_method, result, caught = None, None, None, None
            validating_output = False
            start_time = time.time()
            try:
                kwargs = {} if max_tokens is None else {'max_tokens': max_tokens}
                with single_model_attempt():
                    response = await self.model.acall(prompt, **kwargs)
                validating_output = True
                if response_metadata(response).get('finish_reason') == 'length':
                    raise ValueError('Response reached its output token limit; return a complete, compact JSON object.')
                raw, parse_method, parse_note = parse_llm_json_response(response, empty_fallback=None)
                if parse_method != 'json_block':
                    raise ValueError(parse_note or 'Response is not complete JSON.')
                result = validator(raw)
                last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                caught = exc
                last_error = str(exc)[:1000] or type(exc).__name__
            status = 'ok' if caught is None else ('retrying' if attempt <= retries else 'failed')
            failure, output_failure, skipped = None, None, None
            if status == 'failed':
                failure = {
                    'agent_id': self.profile_id, 'kind': kind, 'request_id': request_id,
                    'attempts': attempt, 'error': last_error,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                }
                env = getattr(self, 'env', None)
                if kind == 'memory':
                    skipped, fallback = SkippedMemory(), 'skip_memory_write'
                elif kind == 'recommendation':
                    skipped, fallback = SkippedRecommendation(), 'skip_recommendation'
                else:
                    skipped = skipped_social_reaction(instruction, observation)
                    fallback = 'skip_reaction' if skipped is not None else 'skip_event'
                output_failure = register_output_failure(
                    env if env is not None else self,
                    dict(failure, action=action, step=getattr(env, 'current_step', None),
                         fallback=fallback, error_stage='output' if validating_output else 'transport'))
                status = 'skipped_error'
            context = {
                'instruction': instruction, 'observation': observation, 'memory': memory,
                'llm_request_id': request_id, 'llm_request_kind': kind,
                'format_attempt': attempt, 'format_status': status,
                'format_error': last_error, 'parse_method': parse_method,
                'response_metadata': response_metadata(response),
            }
            if output_failure is not None:
                context['reaction_output_failure'] = output_failure
                context['system_skipped'] = status == 'skipped_error'
            if kind == 'recommendation':
                context['recommendation_request_id'] = request_id
            await self._record_decision({
                'agent_id': self.profile_id, 'agent_type': self.profile.agent_type,
                'prompt': attempt_prompt, 'output': getattr(response, 'text', None),
                'processing_time': time.time() - start_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'decision_id': str(uuid.uuid4()), 'action': action,
                'context': context, 'response_validation': 'passed' if caught is None else 'failed',
                'feedback': None, 'rating': None, 'reason': last_error,
            })
            if caught is None:
                return result
            logger.warning(
                'Agent(ID:{}) {} format {}: attempt={}/{} request={} error={}',
                self.profile_id, kind, status, attempt, retries + 1, request_id, last_error,
            )
            if status == 'skipped_error':
                effect = ('memory write skipped; successful reaction preserved' if kind == 'memory'
                          else 'no ranking selected' if kind == 'recommendation'
                          else 'no action or memory generated')
                logger.warning(
                    'Agent(ID:{}) {} skipped after request exhaustion: request={} '
                    'topic_failure_count={} stop_at={}; {}',
                    self.profile_id, kind, request_id, output_failure['count'], output_failure['stop_at'], effect,
                )
                if skipped is None:
                    raise ModelRequestSkipped(f'{kind} {request_id} skipped after {attempt} attempts: {last_error}') from caught
                return skipped
            delay = max(0.0, float(os.environ.get('ONESIM_LLM_RETRY_WAIT_SECONDS', '2')))
            await asyncio.sleep(min(delay * 2 ** (attempt - 1), 30.0))

    def __init__(self,
                 sys_prompt: Optional[str] = None,
                 model_config_name: str = None,
                 event_bus_queue: asyncio.Queue = None,
                 profile: AgentProfile=None,
                 memory: MemoryStrategy=None,
                 planning: PlanningBase=None,
                 relationship_manager: RelationshipManager=None) -> None:
        super().__init__(sys_prompt, model_config_name)


        self._queue = asyncio.Queue()
        '''
        The queue where the agent will store the events.
        '''

        self._event_schema: Dict[str, List[str]] = {}
        '''
        Key: event_kind, value: list of ability names.
        '''
        self._property_futures: Dict[str, Future] = {}
        self._event_bus_queue: asyncio.Queue = event_bus_queue
        self._sync_event = asyncio.Event()
        self.profile = profile
        self.memory=memory
        self.planning=planning
        if relationship_manager is None:
            self.relationship_manager = RelationshipManager(profile_id=self.profile.get_agent_profile_id())
        else:
            self.relationship_manager = relationship_manager
        self.relationships = self.relationship_manager.get_all_relationships()
        if self.memory:
            self.memory.set_agent_context(self.create_context())
        self.stopped=False
        self._run_subtasks: set = set()
        self._hooks = {
            'before_event_handling': [],
            'after_event_handling': [],
            'before_action': [],
            'after_action': [],
            'before_reaction_generation': [],
            'after_reaction_generation': [],
            'before_memory_generation': [],
            'after_memory_generation': [],
        }
        # Register event handler for termination events
        self.register_event("EndEvent", "handle_end_event")

        # Register event handlers for data access
        self.register_event("DataEvent", "handle_data_event")
        self.register_event("DataResponseEvent", "handle_data_response")
        self.register_event("DataUpdateEvent", "handle_data_update_event")
        self.register_event("DataUpdateResponseEvent", "handle_data_update_response")

        # Data request futures dictionary
        self._data_futures: Dict[str, Future] = {}
        # Data update futures dictionary
        self._data_update_futures: Dict[str, Future] = {}

    def is_stopped(self) -> bool:
        return self.stopped

    def count_pending_run_tasks(self) -> int:
        return sum(1 for t in self._run_subtasks if not t.done())

    def has_pending_event_queue(self) -> bool:
        return not self._queue.empty()

    def _on_run_task_done(self, task: asyncio.Task) -> None:
        self._run_subtasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if isinstance(exc, ModelRequestSkipped):
            logger.warning('Agent event skipped after exhausted model request: {}', exc)
            return
        if exc is not None:
            logger.error(
                f"{self.profile.agent_type}(ID:{self.profile_id}) run_task failed: {exc}"
            )
            if not getattr(self, '_llm_failure', None):
                self._mark_llm_failure({
                    'agent_id': self.profile_id, 'kind': 'agent_task',
                    'error': str(exc)[:1000], 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                })
        
    async def cancel_pending_run_tasks(self) -> None:
        tasks = [t for t in self._run_subtasks if not t.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def register_event(self, event_kind: str, ability_name: str) -> None:
        if event_kind not in self._event_schema:
            self._event_schema[event_kind] = []
        self._event_schema[event_kind].append(ability_name)

    def add_event(self, event: Event) -> None:
        self._queue.put_nowait(event)

    async def get_event(self) -> Event:
        pass

    def register_hook(self, hook_name: str, hook_function: Callable):
        if hook_name in self._hooks:
            self._hooks[hook_name].append(hook_function)
        else:
            raise ValueError(f"Unknown hook: {hook_name}")

    async def _execute_hooks(self, hook_name: str, **kwargs):
        for hook in self._hooks.get(hook_name, []):
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(self, **kwargs)
                else:
                    hook(self, **kwargs)
            except Exception as e:
                logger.error(f"Error in hook '{hook_name}': {e}")

    @staticmethod
    def _normalize_llm_reaction(parsed: Any) -> Dict[str, Any]:
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            if not parsed:
                return {"decisions": []}
            if len(parsed) == 1:
                only = parsed[0]
                if isinstance(only, dict):
                    inner = only
                    if "decisions" in inner or "keep_following_note_ids" in inner:
                        return inner
                    if "note_id" in inner:
                        return {"decisions": parsed}
                    return inner
                if isinstance(only, list):
                    return {"decisions": only if only else []}
            return {"decisions": parsed}
        logger.warning(
            "Unexpected LLM reaction JSON type: {}; using empty decisions.",
            type(parsed).__name__,
        )
        return {"decisions": []}

    @staticmethod
    def _parse_llm_json_block_response(response: Any) -> Any:
        from onesim.models.parsers.json_parsers import parse_json_block_loose

        return parse_json_block_loose(response)
        
    async def run_task(self, method: Callable, event: Event):
        # Record the incoming event
        await self.record_event(event)
        reses = await method(event)
        if not isinstance(reses, list):
            reses = [reses]

        for res in reses:
            if res is None:
                continue
            res.parent_event_id = event.event_id
            await self._event_bus_queue.put(res)

    async def run(self):
        while not self.is_stopped():
            try:
                event = await self._queue.get()
                await self._execute_hooks('before_event_handling', event=event)
                if event.event_kind not in self._event_schema:
                    # raise ValueError(
                    #      f"Event type {event.event_kind} not registered in {self.profile.agent_type}(ID:{self.profile_id}).")
                    logger.error(f"Event type {event.event_kind} not registered in {self.profile.agent_type}(ID:{self.profile_id}).")
                    self._queue.task_done()
                    continue

                for ability_name in self._event_schema[event.event_kind]:
                    # call self function with ability_name
                    method: Optional[Callable] = getattr(self, ability_name, None)
                    if not callable(method):
                        logger.error(f"Method {ability_name} not found in {self.__class__.__name__}.")
                        continue
                    await self._execute_hooks('before_action', ability_name=ability_name, event=event)
                    rt = asyncio.create_task(self.run_task(method, event))
                    self._run_subtasks.add(rt)
                    rt.add_done_callback(self._on_run_task_done)
                    await self._execute_hooks('after_action', ability_name=ability_name, event=event)
                await self._execute_hooks('after_event_handling', event=event)
                self._queue.task_done()
            except asyncio.CancelledError:
                # logger.info(f"Agent {self.profile_id} task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in agent {self.profile.agent_type}(ID:{self.profile_id}) run loop: {e}")
                self._queue.task_done()

    async def handle_end_event(self, event: Event) -> None:
        """
        Handle termination event to gracefully stop the agent.
        
        Args:
            event (Event): The EndEvent containing termination details
        """

        # Perform any cleanup tasks
        try:

            # Set stopped flag to terminate the run loop
            self.stopped = True
            logger.info(f"Agent {self.profile.agent_type}(ID:{self.profile_id}) received termination event: {event.get('reason', 'unknown')}")
        except Exception as e:
            logger.error(f"Error during agent {self.profile.agent_type}(ID:{self.profile_id}) termination: {e}")

        # No response event needed for termination
        return None

    async def generate_memory(self, instruction: str, observation: str, reaction: dict) -> str:
        if not self.memory:
            return ""
        profile_str = await self.get_prompt_profile(observation)

        prompt_text = f"""
        ### Agent Profile:
        {profile_str}

        ### Event Details:
        Observation: {observation if observation else "No specific observation provided."}
        Instruction: {instruction}
        Reaction: {json.dumps(reaction, indent=2)}

        Based on the agent's profile and the complete event that occurred (the instruction received, what was observed, and how the agent reacted), generate a concise memory that captures this experience from the agent's perspective. The memory should be personal and reflect the complete interaction including what the agent was asked to do, what they perceived, and how they responded.

        {load_prompt(Path(__file__).parent / 'prompts' / 'memory.txt')}

        Structure the response in JSON format with a single 'memory' field containing the memory text. Format the response in a json fenced code block as follows:
        ```json
        {{"memory": "Your memory text here"}}
        ```
        """
        ### the returned json format should comes from the memory manager
        memory = await self._validated_model_request(
            prompt_text, kind="memory", action="generate_memory",
            instruction=instruction, observation=observation,
            validator=validate_memory,
        )
        if isinstance(memory, SkippedMemory):
            # Keep the existing memory and allow the already validated reaction
            # to reach its handler. Never store an empty or invented memory.
            return ""
        if self.memory:
            with model_audit_scope(getattr(self, 'env', None) or self):
                await self.memory.add(MemoryItem(self.agent_id, memory))
        return memory

    async def generate_reaction(self, instruction: str, observation: str = None, *, prepare_input=None, response_validator=None) -> json:
        async with self._reaction_semaphore:
            if prepare_input is not None:
                instruction, observation = await prepare_input()
            logger.info(f"{self.profile.agent_type}(ID:{self.profile_id}) generate_reaction entry")
            
            # Get the Profile and Memory information of the Agent
            profile_str = await self.get_prompt_profile(observation)
            if self.memory:     
                with model_audit_scope(getattr(self, 'env', None) or self):
                    memory_msgs = await self.memory.retrieve(observation)
                memory=""
                for msg in memory_msgs:
                    memory+=msg.content
            else:
                memory=""
            logger.info(f"{self.profile.agent_type}(ID:{self.profile_id}) after memory.retrieve")

            caller_frame = inspect.currentframe().f_back
            
            # Get the name of the calling function
            action_name=caller_frame.f_code.co_name
            work_graph=WorkGraph()
            successor_types=work_graph.get_successor_agent_types(f"{self.profile.agent_type}.{action_name}")

            if self.planning:
                planning = await self._plan_with_retries(profile_str, memory, observation, instruction)
                planning = "### Planning:\n" + planning + "\n" if planning else ""
            else:
                planning=""
            logger.info(f"{self.profile.agent_type}(ID:{self.profile_id}) after planning.plan")

            if memory.strip():
                _mem_planning_gate = load_prompt(Path(__file__).parent / 'prompts' / 'memory_planning_check.txt')
            else:
                _mem_planning_gate = ""

            prompt_text = f"""
            ### Agent Profile:
            {profile_str}

            ### Memory:
            {memory}

            ### Observation:
            {observation if observation else "No specific observation provided."}

            ### Instruction:
            {instruction}

            {planning}
                        
            Please analyze the Agent's profile, memory, observation and planning. Based on these, generate a response that aligns with the Agent's identity and instruction. 
            
            {_mem_planning_gate}

            Structure the response in JSON format, specifying a detailed action or reaction based on the instruction.
            Keep the JSON compact: only decision rows for items in this batch; short strings; no extra example rows.
            You should respond a json object in a json fenced code block as follows:
            ```json
            Your JSON response here
            ```
            """
            def validate_output(raw):
                reaction = validate_reaction(raw, instruction, observation or "")
                if response_validator is not None:
                    reaction = response_validator(reaction)
                return reaction

            reaction = await self._validated_model_request(
                prompt_text, kind="reaction", action=action_name,
                instruction=instruction, observation=observation, memory=memory,
                validator=validate_output,
                max_tokens=int(os.environ.get("ONESIM_REACTION_MAX_TOKENS", "4096")),
            )

            if isinstance(reaction, SkippedReaction):
                return reaction

            if self.memory:
                event_memory = await self.generate_memory(instruction, observation, reaction)
            else:
                event_memory = ""

            logger.info(f"{self.profile.agent_type}(ID:{self.profile_id}) Action: {action_name} - Event Log: "
            f"Observation: {observation or 'None'}, "
            f"Reaction: {json.dumps(reaction, indent=2)}, "
            f"Memory: {json.dumps(event_memory, indent=2)}")

            return reaction

    async def _plan_with_retries(self, profile, memory, observation, instruction):
        """Planning is optional; exhausted planning never blocks the reaction."""
        request_id = str(uuid.uuid4())
        limit = model_attempt_limit()
        for attempt in range(1, limit + 1):
            try:
                with single_model_attempt():
                    result = await self.planning.plan(profile=profile, memory=memory,
                                                      observation=observation, instruction=instruction)
                if not isinstance(result, str) or not result.strip():
                    raise ValueError('Planning response is empty')
                return result
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning('Planning request={} attempt={}/{} failed: {}', request_id, attempt, limit, exc)
                if attempt == limit:
                    env = getattr(self, 'env', None)
                    register_output_failure(env if env is not None else self, {
                        'agent_id': self.profile_id, 'kind': 'planning', 'request_id': request_id,
                        'attempts': attempt, 'error': str(exc)[:1000], 'fallback': 'skip_planning',
                        'action': 'plan', 'step': getattr(env, 'current_step', None),
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    })
                    return ''
                delay = max(0.0, float(os.environ.get('ONESIM_LLM_RETRY_WAIT_SECONDS', '2')))
                await asyncio.sleep(min(delay * 2 ** (attempt - 1), 30.0))

    async def generate_recommendation(
        self, instruction: str, observation: str = None, *,
        candidate_post_ids: Optional[List[str]] = None,
    ) -> dict:
        async with self._reaction_semaphore:
            logger.info(f"{self.profile.agent_type}(ID:{self.profile_id}) generate_recommendation entry")
    
            caller_frame = inspect.currentframe().f_back
            
            # Get the name of the calling function (for decision record export)
            action_name = caller_frame.f_code.co_name

            prompt_text = f"""
            ### Observation:
            {observation if observation else "No specific observation provided."}

            ### Instruction:
            {instruction}

            Please analyze the observation and instruction. Based on these, generate a recommendation response that aligns with the Agent's identity and instruction. 
            
            Return only the JSON object requested by the instruction.
            """
            rec_max_tokens = int(os.environ.get("ONESIM_RECOMMENDATION_MAX_TOKENS", "4096"))
            if candidate_post_ids is None:
                candidate_post_ids = recommendation_candidates(observation)
            reaction = await self._validated_model_request(
                prompt_text, kind="recommendation", action=action_name,
                instruction=instruction, observation=observation,
                validator=lambda raw: validate_recommendation(raw, candidate_post_ids),
                max_tokens=rec_max_tokens, error_class=RecommendationFormatError,
            )
            logger.info(
                f"{self.profile.agent_type}(ID:{self.profile_id}) Action: {action_name} - generate_recommendation - "
                f"Observation: {observation or 'None'}, "
                f"Reaction: {json.dumps(reaction, indent=2)}"
            )
            return reaction

    async def get_memory(self):
        if not self.memory:
            return []
        return await self.memory.get_all_memory_str()

    def add_relationship(self, target_id: str, description: str,target_info: Optional[Dict]=None):
        self.relationship_manager.add_relationship(target_id, description,target_info)

    def remove_relationship(self, target_id: str):
        self.relationship_manager.remove_relationship(target_id)

    def update_relationship(self, target_id: str, description: str):
        self.relationship_manager.update_relationship(target_id, description)

    def get_relationship(self, target_id: str) -> Optional[Relationship]:
        return self.relationship_manager.get_relationship(target_id)

    def get_all_relationships(self) -> List[Relationship]:
        return self.relationship_manager.get_all_relationships()

    def create_context(self):
        return AgentContext(self.profile_id,self.profile,self.relationship_manager)

    async def interact(self, message: str, chat_history: List[Dict[str, Any]] = None):
        """
        Process a user chat message and generate an in-character response from the agent.
        
        Args:
            message (str): The message from the user
            chat_history (List[Dict[str, Any]], optional): Previous chat history for context
            
        Returns:
            dict: The agent's in-character response
        """
        # Format chat history as context if available
        context = ""
        if chat_history and len(chat_history) > 0:
            context = "Previous conversation:\n"
            for entry in chat_history[-5:]:  # Only use the last 5 messages for context
                role = "User" if entry.get("role") == "user" else self.profile.get_data("name") or "Agent"
                context += f"{role}: {entry.get('message', '')}\n"

        # Construct a prompt that focuses on in-character role-playing
        instruction = f"""A user is talking directly to you. Respond as your character would in a conversation.
Stay completely in character based on your profile and respond naturally to the user's message.
Do not refer to yourself as an AI, a language model, or a simulation - you are the character described in your profile.
Respond in JSON format with a single 'message' field containing your response. Format the response in a json fenced code block as follows:
```json
{{
    "message": "Your response here"
}}
```
"""

        # Add context from chat history if available
        observation = f"{context}\nUser's message: {message}"

        reaction = await self.generate_reaction(instruction, observation)

        # Make sure we have a message field for the frontend
        if isinstance(reaction, dict) and "message" not in reaction:
            # If there's no message field but there is a response or content field, use that
            if "response" in reaction:
                reaction["message"] = reaction["response"]
            elif "content" in reaction:
                reaction["message"] = reaction["content"]
            elif "text" in reaction:
                reaction["message"] = reaction["text"]
            # Otherwise create a message from the reaction
            else:
                reaction["message"] = str(reaction)

        return reaction

    @property
    def profile_id(self):
        return self.profile.get_agent_profile_id() 

    def get_profile_str(self,include_private: bool = None):
        return self.profile.get_profile_str(include_private)

    def get_profile(self, include_private: bool = None):
        return self.profile.get_profile(include_private)

    def set_env(self, env):
        """Set the simulation environment associated with this agent."""
        self.env = env

    async def record_event(self, event: Event):
        """
        Record an event for data storage
        
        Args:
            event: Event object
        """

        if event.event_kind in [
            'DataEvent',
            'DataUpdateEvent',
            'DataResponseEvent',
            'DataUpdateResponseEvent',
            'PauseEvent',
            'ResumeEvent',
            'EndEvent',
            'StartEvent',
        ]:
            return

        # Check if we have a local environment reference
        if hasattr(self, 'env') and self.env is not None and hasattr(self.env, 'queue_event'):
            # Local mode or using ProxyEnv - queue directly
            # Handle both sync and async implementations
            try:
                # If queue_event is synchronous
                await self.env.queue_event(event.to_dict())
            except Exception as e:
                logger.error(f"Error in synchronous queue_event: {e}")
        else:
            # Legacy distributed worker mode - send to master via gRPC
            node = get_node()
            if node.role == NodeRole.WORKER and node.master_address and node.master_port:
                # Create task to send data to master (don't await to avoid blocking)
                asyncio.create_task(
                    grpc_impl.send_storage_event_to_master(
                        node.master_address,
                        node.master_port,
                        event.to_dict()
                    )
                )
            else:
                logger.warning(f"Unable to record event: No environment reference and not in worker mode")

    async def _record_decision(self, decision_data: Dict[str, Any]):
        """
        Record a decision for data storage
        
        Args:
            decision_data: Decision data to record
        """
        # Check if we have a local environment reference
        if hasattr(self, 'env') and self.env is not None and hasattr(self.env, 'queue_decision'):
            # Local mode or using ProxyEnv - queue directly
            # Handle both sync and async implementations
            try:
                # If queue_decision is synchronous
                await self.env.queue_decision(decision_data)
            except Exception as e:
                logger.error(f"Error in synchronous queue_decision: {e}")
        else:
            # Legacy distributed worker mode - send to master via gRPC
            node = get_node()
            if node.role == NodeRole.WORKER and node.master_address and node.master_port:
                # Send data to master
                await grpc_impl.send_decision_record_to_master(
                    node.master_address,
                    node.master_port,
                    decision_data
                )
            else:
                logger.warning(f"Unable to record decision: No environment reference and not in worker mode")

    async def get_data(self, key: str,default: Optional[Any] = None):
        """Get data from agent profile"""
        if not key or not hasattr(self, 'profile') or not self.profile:
            return default
        if "profile" in key:
            key=key.replace("profile.","")
        return self.profile.get_data(key,default)

    async def handle_data_event(self, event: DataEvent) -> Optional[DataResponseEvent]:
        """
        Handle incoming data access requests
        
        Args:
            event (DataEvent): Data access event
            
        Returns:
            Optional[DataResponseEvent]: Response event or None
        """ 
        try:
            # Get data from profile
            data_value = await self.get_data(event.key)
            # Create response event
            response = DataResponseEvent(
                from_agent_id=self.profile_id,
                to_agent_id=event.from_agent_id,
                request_id=event.request_id,
                key=event.key,
                data_value=data_value,
                success=True,
                parent_event_id=event.event_id
            )
            return response
        except Exception as e:
            # Create error response
            error_response = DataResponseEvent(
                from_agent_id=self.profile_id,
                to_agent_id=event.from_agent_id,
                request_id=event.request_id,
                key=event.key,
                data_value=None,
                success=False,
                error=str(e),
                parent_event_id=event.event_id
            )

            return error_response

    async def handle_data_response(self, event: DataResponseEvent) -> None:
        """
        Handle incoming data response events
        
        Args:
            event (DataResponseEvent): Data response event
        """
        # Check if we're waiting for this response
        if event.request_id in self._data_futures:
            future = self._data_futures.pop(event.request_id)

            if not future.done():
                if event.success:
                    future.set_result(event.data_value)
                else:
                    future.set_exception(ValueError(event.error or "Unknown error"))

            # If we have a sync event, set it
            if hasattr(self, '_sync_event'):
                self._sync_event.set()
                # Reset for next operation
                self._sync_event.clear()

    async def get_env_data(self, key: str, default: Optional[Any] = None, parent_event_id: Optional[str] = None) -> Any:
        """
        Get data from the environment
        
        Args:
            key (str): Data key to access
            default (Any, optional): Default value if key not found
            parent_event_id (str, optional): ID of parent event that triggered this request
            
        Returns:
            Any: The requested data or default value
        """
        env = getattr(self, "env", None)
        if env is not None and hasattr(env, "get_data"):
            try:
                return await env.get_data(key, default)
            except Exception as e:
                logger.warning(f"Local env.get_data({key}) failed, falling back to event bus: {e}")

        # Create a unique request ID
        request_id = f"agent_env_req_{time.time()}_{id(self)}"

        # Create future for response
        future = Future()
        self._data_futures[request_id] = future

        # Create data request event
        data_event = DataEvent(
            from_agent_id=self.profile_id,
            to_agent_id="ENV",  # Special target for environment
            source_type="AGENT",
            target_type="ENV",
            key=key,
            default=default,
            request_id=request_id,
            parent_event_id=parent_event_id
        )

        # Send the request
        await self._event_bus_queue.put(data_event)

        # Wait for response: in single-node mode the response is delivered to our _queue,
        # so we must process our queue while waiting (same event loop may not schedule run() in time).
        # 0 / unset = wait until ENV replies (params must not fall back after 60s).
        try:
            wait_seconds = float(os.environ.get("ONESIM_ENV_DATA_WAIT_SECONDS", "0"))
        except ValueError:
            wait_seconds = 0.0
        loop = asyncio.get_event_loop()
        deadline = None if wait_seconds <= 0 else loop.time() + wait_seconds
        last_wait_log = loop.time()
        try:
            while not future.done():
                if deadline is not None and loop.time() >= deadline:
                    raise asyncio.TimeoutError()
                try:
                    event = await asyncio.wait_for(self._queue.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    now = loop.time()
                    if now - last_wait_log >= 60.0:
                        logger.warning(f"Still waiting for environment data: {key}")
                        last_wait_log = now
                    await asyncio.sleep(0)
                    continue
                # Only process data response events here; others go back to queue for run() to handle
                if event.event_kind in ("DataResponseEvent", "DataUpdateResponseEvent"):
                    try:
                        if event.event_kind in self._event_schema:
                            for ability_name in self._event_schema[event.event_kind]:
                                method = getattr(self, ability_name, None)
                                if callable(method):
                                    if asyncio.iscoroutinefunction(method):
                                        await method(event)
                                    else:
                                        method(event)
                                    break
                    finally:
                        if hasattr(self._queue, 'task_done'):
                            self._queue.task_done()
                else:
                    await self._queue.put(event)
                if future.done():
                    break
            if future.done():
                return future.result()
            raise asyncio.TimeoutError()
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for environment data: {key}")
            self._data_futures.pop(request_id, None)
            return default
        except Exception as e:
            logger.error(f"Error getting environment data: {e}")
            self._data_futures.pop(request_id, None)
            return default

    async def get_agent_data(self, agent_id: str, key: str, default: Optional[Any] = None, parent_event_id: Optional[str] = None) -> Any:
        """
        Get data from another agent
        
        Args:
            agent_id (str): ID of agent to get data from
            key (str): Data key to access
            default (Any, optional): Default value if key not found
            parent_event_id (str, optional): ID of parent event that triggered this request
            
        Returns:
            Any: The requested data or default value
        """
        # Prevent self-queries
        if agent_id == self.profile_id:
            return await self.get_data(key)

        # Create a unique request ID
        request_id = f"agent_req_{time.time()}_{id(self)}"

        # Create future for response
        future = Future()
        self._data_futures[request_id] = future

        # Create data request event
        data_event = DataEvent(
            from_agent_id=self.profile_id,
            to_agent_id=agent_id,
            source_type="AGENT",
            target_type="AGENT",
            key=key,
            default=default,
            request_id=request_id,
            parent_event_id=parent_event_id
        )

        # Send the request
        await self._event_bus_queue.put(data_event)

        # Wait for response with timeout
        try:
            if hasattr(self, '_sync_event'):
                # If we have a sync event, wait for it
                await asyncio.wait_for(self._sync_event.wait(), timeout=30.0)
                return await future
            else:
                # Otherwise just wait for the future directly
                return await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for data from agent {agent_id}")
            self._data_futures.pop(request_id, None)
            return default
        except Exception as e:
            logger.error(f"Error getting data from agent {agent_id}: {e}")
            self._data_futures.pop(request_id, None)
            return default

    async def update_data(self, key: str, value: Any) -> bool:
        """Update data in agent profile"""
        if not key or not hasattr(self, 'profile') or not self.profile:
            return False
        return self.profile.update_data(key, value)

    async def handle_data_update_event(self, event: DataUpdateEvent) -> Optional[DataUpdateResponseEvent]:
        """
        Handle incoming data update requests
        
        Args:
            event (DataUpdateEvent): Data update event
            
        Returns:
            Optional[DataUpdateResponseEvent]: Response event or None
        """

        try:
            # Update data in profile - Added await here
            success = await self.update_data(event.key, event.value)

            # Create response event
            response = DataUpdateResponseEvent(
                from_agent_id=self.profile_id,
                to_agent_id=event.from_agent_id,
                request_id=event.request_id,
                key=event.key,
                success=success,
                parent_event_id=event.event_id
            )

            return response
        except Exception as e:
            # Create error response
            error_response = DataUpdateResponseEvent(
                from_agent_id=self.profile_id,
                to_agent_id=event.from_agent_id,
                request_id=event.request_id,
                key=event.key,
                success=False,
                error=str(e),
                parent_event_id=event.event_id
            )

            return error_response

    async def handle_data_update_response(self, event: DataUpdateResponseEvent) -> None:
        """
        Handle incoming data update response events
        
        Args:
            event (DataUpdateResponseEvent): Data update response event
        """
        # Check if we're waiting for this response
        if event.request_id in self._data_update_futures:
            future = self._data_update_futures.pop(event.request_id)

            if not future.done():
                if event.success:
                    future.set_result(True)
                else:
                    future.set_exception(ValueError(event.error or "Unknown error"))

            # If we have a sync event, set it
            if hasattr(self, '_sync_event'):
                self._sync_event.set()
                # Reset for next operation
                self._sync_event.clear()

    async def update_env_data(self, key: str, value: Any, parent_event_id: Optional[str] = None) -> bool:
        """
        Update data in the environment with distributed locking
        
        Args:
            key (str): Data key to update
            value (Any): New value to set
            parent_event_id (str, optional): ID of parent event that triggered this request
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        # Create a unique request ID
        request_id = f"agent_env_update_req_{time.time()}_{id(self)}"

        # Create future for response
        future = Future()
        self._data_update_futures[request_id] = future

        # Create data update event
        data_update_event = DataUpdateEvent(
            from_agent_id=self.profile_id,
            to_agent_id="ENV",  # Special target for environment
            source_type="AGENT",
            target_type="ENV",
            key=key,
            value=value,
            request_id=request_id,
            parent_event_id=parent_event_id
        )

        # Get distributed lock for this key
        lock_id = f"env_data_lock_{key}"
        lock = await get_lock(lock_id)

        try:
            # Acquire lock before sending update
            async with lock:
                # Send the request
                await self._event_bus_queue.put(data_update_event)

                # Wait for response with timeout
                try:
                    if hasattr(self, '_sync_event'):
                        # If we have a sync event, wait for it
                        await asyncio.wait_for(self._sync_event.wait(), timeout=30.0)
                        return await future
                    else:
                        # Otherwise just wait for the future directly
                        return await asyncio.wait_for(future, timeout=30.0)
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout waiting for environment data update: {key}")
                    self._data_update_futures.pop(request_id, None)
                    return False
                except Exception as e:
                    logger.error(f"Error updating environment data: {e}")
                    self._data_update_futures.pop(request_id, None)
                    return False
        except Exception as e:
            logger.error(f"Error acquiring lock for environment data update: {e}")
            return False

    async def update_agent_data(self, agent_id: str, key: str, value: Any, parent_event_id: Optional[str] = None) -> bool:
        """
        Update data in another agent with distributed locking
        
        Args:
            agent_id (str): ID of agent to update data in
            key (str): Data key to update
            value (Any): New value to set
            parent_event_id (str, optional): ID of parent event that triggered this request
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        # Prevent self-queries
        if agent_id == self.profile_id:
            return await self.update_data(key, value)

        # Create a unique request ID
        request_id = f"agent_update_req_{time.time()}_{id(self)}"

        # Create future for response
        future = Future()
        self._data_update_futures[request_id] = future

        # Create data update event
        data_update_event = DataUpdateEvent(
            from_agent_id=self.profile_id,
            to_agent_id=agent_id,
            source_type="AGENT",
            target_type="AGENT",
            key=key,
            value=value,
            request_id=request_id,
            parent_event_id=parent_event_id
        )

        # Get distributed lock for this agent and key
        lock_id = f"agent_data_lock_{agent_id}_{key}"
        lock = await get_lock(lock_id)

        try:
            # Acquire lock before sending update
            async with lock:
                # Send the request
                await self._event_bus_queue.put(data_update_event)

                # Wait for response with timeout
                try:
                    if hasattr(self, '_sync_event'):
                        # If we have a sync event, wait for it
                        await asyncio.wait_for(self._sync_event.wait(), timeout=30.0)
                        return await future
                    else:
                        # Otherwise just wait for the future directly
                        return await asyncio.wait_for(future, timeout=30.0)
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout waiting for data update in agent {agent_id}")
                    self._data_update_futures.pop(request_id, None)
                    return False
                except Exception as e:
                    logger.error(f"Error updating data in agent {agent_id}: {e}")
                    self._data_update_futures.pop(request_id, None)
                    return False
        except Exception as e:
            logger.error(f"Error acquiring lock for agent data update: {e}")
            return False

    async def add_memory(self, memory: str):
        if not self.memory:
            return
        # Verify MemoryStrategy.add is truly async if it performs I/O.
        await self.memory.add(MemoryItem(self.agent_id,memory))

    async def add_memory(self, memory: str):
        if not self.memory:
            return
        await self.memory.add(MemoryItem(self.agent_id, memory))
