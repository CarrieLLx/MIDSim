from abc import ABC, abstractmethod
from typing import Any, Dict
from loguru import logger
import json
from onesim.models.core.message import Message
from onesim.models import JsonBlockParser
from ..memory_item import MemoryItem


def _parse_json_dict_from_llm_response(response: Any) -> Dict[str, Any]:
    """
    先按 JsonBlockParser（```json ... ```）；若无代码块则尝试整段或首尾大括号切片 json.loads。
    避免模型只输出裸 JSON 时误报 Start tag '```json' not found。
    """
    text = getattr(response, "text", None) or ""
    if not text.strip():
        raise ValueError("empty model response")
    parser = JsonBlockParser()
    try:
        res = parser.parse(response)
        if isinstance(res.parsed, dict):
            return res.parsed
    except Exception:
        pass
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j > i:
        try:
            obj = json.loads(s[i : j + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    raise ValueError("no JSON object found in model response")


class MemoryOperation(ABC):
    @abstractmethod
    async def execute(self,strategy, storage_name=None, *args, **kwargs):
        pass

class AddMemoryOperation(MemoryOperation):
    async def execute(self, strategy, storage_name: str, memory_item):
        storage = strategy._storage_map.get(storage_name)
        if not storage:
            raise ValueError(f"Storage not found: {storage_name}")
            
        # 计算所有metrics并存入memory_item
        for metric_name, metric in strategy._metrics.items():
            try:
                memory_item.attributes[metric_name] = await metric.calculate(memory_item)
            except Exception as e:
                logger.error(f"Error calculating metric {metric_name}: {e}")
                
        await storage.add(memory_item)


class RetrieveMemoryOperation(MemoryOperation):
    async def execute(self, strategy, storage_name: str, query, top_k: int):
        storage = strategy._storage_map.get(storage_name)
        if not storage:
            raise ValueError(f"Storage not found: {storage_name}")
        return await storage.query(query, top_k)

class RemoveMemoryOperation(MemoryOperation):
    async def execute(self, strategy, storage_name: str, memory_item):
        storage = strategy._storage_map.get(storage_name)
        if not storage:
            raise ValueError(f"Storage not found: {storage_name}")
        await storage.delete(memory_item)

class ReflectMemoryOperation(MemoryOperation):
    async def execute(self, strategy, *args, **kwargs):
        short_term_storage_name = getattr(strategy, 'short_term_storage_name', 'short_term_storage')
        long_term_storage_name = getattr(strategy, 'long_term_storage_name', 'long_term_storage')
        
        short_term_storage = strategy._storage_map.get(short_term_storage_name)
        if not short_term_storage:
            raise ValueError(f"Short-term storage not found: {short_term_storage_name}")
            
        short_term_memories = await short_term_storage.get_all()
        
        agent_context = strategy.agent_context
        
        memory_descriptions = '\n'.join([str(memory_item) for memory_item in short_term_memories])
        
        prompt = (
            f"Given the following short-term memories:\n\n{memory_descriptions}\n\n"
            f"and the agent context:\n\n{agent_context}\n\n"
            "Generate 3 possible questions that the agent might be concerned with.\n\n"
            "Requirements:\n"
            "- The questions should be relevant to the agent's context and recent memories.\n"
            "- Return the questions as a JSON object with the following format:\n\n"
            "{\n"
            '  "questions": [\n'
            '    "Question 1",\n'
            '    "Question 2",\n'
            '    "Question 3"\n'
            '  ]\n'
            "}\n\n"
            "Output the JSON only: either wrap it in a markdown fence ```json ... ```, "
            "or output a single raw JSON object (no other text)."
        )
        
        formatted_prompt = strategy.model.format(
            Message("user", prompt, role="user")
        )
        
        try:
            agent_id = getattr(agent_context, "agent_id", None)
            model_name = getattr(strategy.model, "config_name", type(strategy.model).__name__)
            logger.info(
                f"[ReflectMemoryOperation] 调用 LLM 生成反思问题 agent_id={agent_id} "
                f"model={model_name} short_term_count={len(short_term_memories)}"
            )
            response = await strategy.model.acall(formatted_prompt)

            parsed = _parse_json_dict_from_llm_response(response)
            questions = parsed.get("questions")
            if not isinstance(questions, list) or not questions:
                raise ValueError("questions missing or not a non-empty list")
            logger.info(f"[ReflectMemoryOperation] Generated questions: {questions}")
        except json.JSONDecodeError as e:
            logger.warning(f"[ReflectMemoryOperation] Failed to parse questions JSON: {e}")
            questions = ["What are my recent experiences?",
                         "What insights can I derive from my memories?",
                         "What should I focus on next?"]
        except Exception as e:
            logger.warning(f"[ReflectMemoryOperation] Error generating questions (using defaults): {e}")
            questions = ["What are my recent experiences?",
                         "What insights can I derive from my memories?",
                         "What should I focus on next?"]
            
        retrieved_memories = []
        for question in questions:
            try:
                long_term_memories = await strategy.execute('retrieve', 
                                                          storage_name=long_term_storage_name, 
                                                          query=question, 
                                                          top_k=5)
                retrieved_memories.extend(long_term_memories)
            except Exception as e:
                logger.error(f"Error retrieving memories for question '{question}': {e}")
                
        retrieved_memories_descriptions = '\n'.join([str(memory_item) for memory_item in retrieved_memories])
        
        # 构建生成见解的提示
        prompt = (
            f"Based on the following retrieved memories:\n\n{retrieved_memories_descriptions}\n\n"
            f"and considering the questions:\n\n{questions}\n\n"
            "Generate 5 high-level insights or reflections that the agent might have.\n\n"
            "Requirements:\n"
            "- The insights should be significant and relevant to the agent's context.\n"
            "- Return the insights as a JSON object with the following format:\n\n"
            "{\n"
            '  "insights": [\n'
            '    "Insight 1",\n'
            '    "Insight 2",\n'
            '    "Insight 3",\n'
            '    "Insight 4",\n'
            '    "Insight 5"\n'
            '  ]\n'
            "}\n\n"
            "Output the JSON only: either wrap it in ```json ... ```, or output a single raw JSON object."
        )
        
        # 调用LLM生成见解
        formatted_prompt = strategy.model.format(
            Message("user", prompt, role="user")
        )
        
        try:
            agent_id = getattr(agent_context, "agent_id", None)
            model_name = getattr(strategy.model, "config_name", type(strategy.model).__name__)
            logger.info(
                f"[ReflectMemoryOperation] 调用 LLM 生成见解 agent_id={agent_id} "
                f"model={model_name} retrieved_count={len(retrieved_memories)} "
                f"questions_count={len(questions)}"
            )
            response = await strategy.model.acall(formatted_prompt)

            parsed = _parse_json_dict_from_llm_response(response)
            insights = parsed.get("insights")
            if not isinstance(insights, list) or not insights:
                raise ValueError("insights missing or not a non-empty list")
            logger.info(f"[ReflectMemoryOperation] Generated insights: {insights}")
        except json.JSONDecodeError as e:
            logger.warning(f"[ReflectMemoryOperation] Failed to parse insights JSON: {e}")
            insights = ["No specific insights could be generated at this time."]
        except Exception as e:
            logger.warning(f"[ReflectMemoryOperation] Error generating insights (using fallback): {e}")
            insights = ["No specific insights could be generated at this time."]
            
        # 为见解创建记忆项并添加到长期存储中
        for insight in insights:
            try:
                memory_item = MemoryItem(agent_context.agent_id, content=insight)
                await strategy.execute('add', storage_name=long_term_storage_name, memory_item=memory_item)
            except Exception as e:
                logger.error(f"Error adding insight to long-term storage: {e}")
                
        # 将记忆从短期存储转移到长期存储
        for memory_item in short_term_memories:
            try:
                await strategy.execute('remove', storage_name=short_term_storage_name, memory_item=memory_item)
                await strategy.execute('add', storage_name=long_term_storage_name, memory_item=memory_item)
            except Exception as e:
                logger.error(f"Error transferring memory to long-term storage: {e}")


class MergeOperation(MemoryOperation):
    async def execute(self, strategy, storage_name: str):
        storage = strategy._storage_map.get(storage_name)
        if not storage:
            raise ValueError(f"Storage not found: {storage_name}")
            
        try:
            merged_memories = await storage.merge()
            await storage.clear()
            for memory_item in merged_memories:
                await storage.add(memory_item)
        except AttributeError:
            logger.error(f"Storage {storage_name} does not support merge operation")
        except Exception as e:
            logger.error(f"Error during merge operation: {e}")

class ForgetOperation(MemoryOperation):
    async def execute(self, strategy, storage_name: str, criteria: Any):
        storage = strategy._storage_map.get(storage_name)
        if not storage:
            raise ValueError(f"Storage not found: {storage_name}")
            
        try:
            await storage.forget(criteria)
        except AttributeError:
            logger.error(f"Storage {storage_name} does not support forget operation")
        except Exception as e:
            logger.error(f"Error during forget operation: {e}")