# -*- coding: utf-8 -*-
""" Base class for Agent """

from __future__ import annotations
from types import GeneratorType
from typing import Optional, Generator, Tuple
from typing import Sequence
from typing import Union
from typing import Any
import json
import uuid
from loguru import logger

from onesim.models.core.model_manager import ModelManager


class AgentBase:
    """Base class for all agents.

    All agents should inherit from this class and implement the `reply`
    function.
    """

    _version: int = 1

    def __init__(
        self,
        sys_prompt: Optional[str] = None,
        model_config_name: str = None,
    ) -> None:
        r"""Initialize an agent from the given arguments.

        Args:
            name (`str`):
                The name of the agent.
            sys_prompt (`Optional[str]`):
                The system prompt of the agent, which can be passed by args
                or hard-coded in the agent.
            model_config_name (`str`, defaults to None):
                The name of the model config, which is used to load model from
                configuration.
        """
        self.sys_prompt = sys_prompt
        
        # Use a different name for the internal attribute
        self._agent_id = self.generate_agent_id()

        # TODO: support to receive a ModelWrapper instance
        if model_config_name is not None:
            model_manager = ModelManager.get_instance()
            self.model = model_manager.get_model(
                config_name=model_config_name,
            )

        # ------------------------------------------------------------------
        # [做法A · 决策模型 + 风格模型] 修改点 1/6 — AgentBase
        #
        # 1) 在 __init__ 参数中增加: style_model_config_name: Optional[str] = None
        # 2) 在 model_config_name 分支之后补充:
        #        self.style_model = None
        #        if style_model_config_name is not None:
        #            self.style_model = model_manager.get_model(
        #                config_name=style_model_config_name)
        #    （若 model_config_name 为 None，需先保证 model_manager 已初始化）
        # 3) __str__ 里可视需要序列化 style_model 的 config_name
        #
        # 说明: config/model_config.json 的 "chat" 数组里加第二项即可，无需新顶层分类；
        #       style_model_config_name 填该项的 config_name（如 vllm-style-lora-9891）。
        # ------------------------------------------------------------------

    @classmethod
    def generate_agent_id(cls) -> str:
        """Generate the agent_id of this agent instance"""
        # TODO: change cls.__name__ into a global unique agent_type
        return uuid.uuid4().hex

    def __str__(self) -> str:
        serialized_fields = {
            "type": self.__class__.__name__,
            "sys_prompt": self.sys_prompt,
            "agent_id": self.agent_id,
        }
        if hasattr(self, "model"):
            serialized_fields["model"] = {
                "model_type": self.model.model_type,
                "config_name": self.model.config_name,
            }
        return json.dumps(serialized_fields, ensure_ascii=False)
    
    @property
    def agent_id(self) -> str:
        """The unique id of this agent.

        Returns:
            str: agent_id
        """
        return self._agent_id

    @agent_id.setter
    def agent_id(self, agent_id: str) -> None:
        """Set the unique id of this agent."""
        self._agent_id = agent_id