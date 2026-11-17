"""
This module implements a load balancer for language models.

The load balancer acts as a model itself, implementing the same interface
as other model adapters, but delegates actual requests to underlying models
based on load balancing strategies.
"""

import random
from typing import Any, Dict, List, Optional, Sequence, Union

from loguru import logger

from .model_base import ModelAdapterBase
from .model_response import ModelResponse
from .message import Message
from onesim.utils.model_retry import single_attempt_active


_INSPECTION_SHRINK_LIMITS = (8000, 3000, 1200)


def _is_content_inspection_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "data_inspection_failed" in text or "inappropriate content" in text


def _shrink_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = "\n...[truncated]...\n"
    keep = max(limit - len(marker), 0)
    head = keep // 2
    tail = keep - head
    return text[:head] + marker + text[-tail:]


def _shrink_messages(messages: List[Any], limit: int) -> List[Any]:
    shrunk: List[Any] = []
    for message in messages:
        if isinstance(message, dict):
            copied = dict(message)
            content = copied.get("content")
            if isinstance(content, str):
                copied["content"] = _shrink_text(content, limit)
            shrunk.append(copied)
        else:
            shrunk.append(message)
    return shrunk


def _shrink_call(args: tuple, kwargs: Dict[str, Any], limit: int) -> tuple:
    new_args = list(args)
    new_kwargs = dict(kwargs)
    if new_args and isinstance(new_args[0], list):
        new_args[0] = _shrink_messages(new_args[0], limit)
    messages = new_kwargs.get("messages")
    if isinstance(messages, list):
        new_kwargs["messages"] = _shrink_messages(messages, limit)
    return tuple(new_args), new_kwargs


class LoadBalancerStrategy:
    """Base class for load balancing strategies."""
    
    def select_model(self, models: List[ModelAdapterBase]) -> ModelAdapterBase:
        """
        Select a model from the available models based on the strategy.
        
        Args:
            models: List of available model instances.
            
        Returns:
            ModelAdapterBase: The selected model instance.
        """
        raise NotImplementedError("Subclasses must implement select_model method")


class RoundRobinStrategy(LoadBalancerStrategy):
    """Round-robin load balancing strategy."""
    
    def __init__(self):
        self.current_index = 0
    
    def select_model(self, models: List[ModelAdapterBase]) -> ModelAdapterBase:
        if not models:
            raise ValueError("No models available for load balancing")
        
        selected_model = models[self.current_index]
        self.current_index = (self.current_index + 1) % len(models)
        return selected_model


class RandomStrategy(LoadBalancerStrategy):
    """Random selection load balancing strategy."""
    
    def select_model(self, models: List[ModelAdapterBase]) -> ModelAdapterBase:
        if not models:
            raise ValueError("No models available for load balancing")
        
        return random.choice(models)


class LoadBalancer(ModelAdapterBase):
    """
    Load balancer for language models.
    
    This class implements the ModelAdapterBase interface but delegates
    actual requests to underlying model instances based on load balancing 
    strategies.
    
    Supports both synchronous and asynchronous calls:
    - For synchronous: model(...) or model.__call__(...)
    - For asynchronous: await model.acall(...)
    """

    def __init__(
        self,
        config_name: str = "chat_load_balancer",
        models: List[Union[str,dict, ModelAdapterBase]] = None,
        strategy: str = "round_robin",
        category: str = "chat",  # 'chat' or 'embedding'
        provider: str = None,
        model_type: str = None,  # For backward compatibility
        model_name: str = None,  # Add explicit model_name parameter
        max_retries: int = 3,  # Maximum number of retries when a model fails
        **kwargs
    ):
        """
        Initialize the load balancer.
        
        Args:
            config_name: Configuration name for this load balancer.
            models: List of model instances or config names to balance between.
            strategy: Load balancing strategy to use (round_robin, random).
            category: The category of models being balanced ('chat' or 'embedding').
            provider: Optional provider to filter models (e.g., 'openai', 'vllm').
            model_type: Alias for category, for backward compatibility.
            model_name: Specific model name this load balancer is for (e.g., "gpt-4").
            max_retries: Maximum number of retries when a model fails.
            **kwargs: Additional parameters for the base model.
        """
        super().__init__(config_name=config_name, **kwargs)

        self.models = models or []
        self._model_instances = []
        self._strategy_name = strategy
        self.category = category.lower()
        self.provider = provider
        self.model_name = model_name  # Store the model name if specified
        self.max_retries = max_retries  # Store max retries; failover still tries every configured key

        if self.category not in ["chat", "embedding"]:
            logger.warning(f"Unknown category '{category}', defaulting to 'chat'")
            self.category = "chat"

        # Initialize the strategy
        if strategy == "round_robin":
            self._strategy = RoundRobinStrategy()
        elif strategy == "random":
            self._strategy = RandomStrategy()
        else:
            raise ValueError(f"Unknown load balancing strategy: {strategy}")

        # For backward compatibility
        self.model_type = self.category

    def initialize_models(self, model_manager=None):
        """
        Initialize model instances from config names.
        
        Args:
            model_manager: The model manager instance to use for getting models.
                           If None, will attempt to import and get the default instance.
        """
        if not model_manager:
            from onesim.models import get_model_manager
            model_manager = get_model_manager()

        self._model_instances = []
        for model in self.models:
            if isinstance(model, str):
                # It's a config name, get the model instance
                model_instance = model_manager.get_model(model)
                self._model_instances.append(model_instance)
            elif isinstance(model, dict):
                # It's a config name, get the model instance
                model_instance = model_manager.get_model(model["config_name"])
                self._model_instances.append(model_instance)
            elif isinstance(model, ModelAdapterBase):
                # It's already a model instance
                self._model_instances.append(model)
            else:
                raise TypeError(f"Expected str or ModelAdapterBase, got {type(model)}")

        if not self._model_instances:
            logger.warning(f"Load balancer '{self.config_name}' has no models configured")

    def _try_with_model(self, model: ModelAdapterBase, *args, **kwargs) -> ModelResponse:
        """
        Try to process a request with a specific model.
        
        Args:
            model: The model to try.
            *args: Arguments to pass to the model.
            **kwargs: Keyword arguments to pass to the model.
            
        Returns:
            ModelResponse: The response from the model.
            
        Raises:
            Exception: If the model call fails.
        """
        try:
            response = model(*args, **kwargs)

            # Add metadata about the load balancer
            if hasattr(response, 'raw') and response.raw is not None:
                if not isinstance(response.raw, dict):
                    response.raw = {"original": response.raw}

                response.raw.update({
                    "load_balancer": self.config_name,
                    "selected_model": model.config_name
                })

            return response
        except Exception as e:
            logger.warning(f"Model '{model.config_name}' failed: {str(e)}")
            raise

    def _next_untried_model(self, tried: set) -> Optional[ModelAdapterBase]:
        """Pick the next key that this request has not already called.

        A failed request walks every configured key once. Duplicate picks from
        the strategy do not consume an attempt.
        """
        pool = self._model_instances
        if not pool:
            return None
        for _ in range(len(pool)):
            model = self._strategy.select_model(pool)
            if model.config_name not in tried:
                return model
        return None

    def __call__(self, *args, **kwargs) -> ModelResponse:
        """
        Process a request synchronously using one of the balanced models.
        
        This method selects a model using the configured load balancing strategy
        and forwards the request to that model. If the selected model fails,
        it tries each remaining configured key once.
        
        Args:
            *args: Arguments to pass to the selected model.
            **kwargs: Keyword arguments to pass to the selected model.
            
        Returns:
            ModelResponse: The response from the selected model.
            
        Raises:
            ValueError: If no models are configured or initialized.
            Exception: If every configured model fails.
        """
        if not self._model_instances:
            raise ValueError(f"Load balancer '{self.config_name}' has no initialized models")

        last_error = None
        tried_models = set()
        saw_inspection = False
        shrink_level = 0

        while True:
            model = self._next_untried_model(tried_models)
            if model is None:
                if saw_inspection and shrink_level < len(_INSPECTION_SHRINK_LIMITS):
                    limit = _INSPECTION_SHRINK_LIMITS[shrink_level]
                    shrink_level += 1
                    args, kwargs = _shrink_call(args, kwargs, limit)
                    tried_models.clear()
                    saw_inspection = False
                    logger.warning(
                        "Content inspection rejected the prompt; retrying all keys with input capped at {} chars",
                        limit,
                    )
                    continue
                break
            tried_models.add(model.config_name)
            try:
                return self._try_with_model(model, *args, **kwargs)
            except Exception as e:
                last_error = e
                saw_inspection = saw_inspection or _is_content_inspection_error(e)
                continue

        raise Exception(
            f"All models failed after {len(tried_models)} attempts. Last error: {str(last_error)}"
        )

    async def _try_with_model_async(self, model: ModelAdapterBase, *args, **kwargs) -> ModelResponse:
        """
        Try to process a request asynchronously with a specific model.
        
        Args:
            model: The model to try.
            *args: Arguments to pass to the model.
            **kwargs: Keyword arguments to pass to the model.
            
        Returns:
            ModelResponse: The response from the model.
            
        Raises:
            Exception: If the model call fails.
        """
        try:
            response = await model.acall(*args, **kwargs)

            # Add metadata about the load balancer
            if hasattr(response, 'raw') and response.raw is not None:
                if not isinstance(response.raw, dict):
                    response.raw = {"original": response.raw}

                response.raw.update({
                    "load_balancer": self.config_name,
                    "selected_model": model.config_name
                })

            return response
        except Exception as e:
            logger.warning(f"Model '{model.config_name}' failed: {str(e)}")
            raise

    async def acall(self, *args, **kwargs) -> ModelResponse:
        """
        Process a request asynchronously using one of the balanced models.
        
        This method selects a model using the configured load balancing strategy
        and forwards the request to that model using its asynchronous interface.
        If the selected model fails, it tries each remaining configured key once.
        
        Args:
            *args: Arguments to pass to the selected model.
            **kwargs: Keyword arguments to pass to the selected model.
            
        Returns:
            ModelResponse: The response from the selected model.
            
        Raises:
            ValueError: If no models are configured or initialized.
            Exception: If every configured model fails.
        """
        if not self._model_instances:
            raise ValueError(f"Load balancer '{self.config_name}' has no initialized models")

        if single_attempt_active():
            # Validation owns the twenty-attempt budget. Do not multiply it by
            # the number of keys, transport retries, or input-shrinking passes.
            model = self._next_untried_model(set())
            return await self._try_with_model_async(model, *args, **kwargs)

        last_error = None
        tried_models = set()
        saw_inspection = False
        shrink_level = 0

        while True:
            model = self._next_untried_model(tried_models)
            if model is None:
                if saw_inspection and shrink_level < len(_INSPECTION_SHRINK_LIMITS):
                    limit = _INSPECTION_SHRINK_LIMITS[shrink_level]
                    shrink_level += 1
                    args, kwargs = _shrink_call(args, kwargs, limit)
                    tried_models.clear()
                    saw_inspection = False
                    logger.warning(
                        "Content inspection rejected the prompt; retrying all keys with input capped at {} chars",
                        limit,
                    )
                    continue
                break
            tried_models.add(model.config_name)
            try:
                logger.info(
                    f"[LoadBalancer] {self.config_name} received a request; selected model: {model.config_name} "
                    f"({len(tried_models)}/{len(self._model_instances)})"
                )
                return await self._try_with_model_async(model, *args, **kwargs)
            except Exception as e:
                last_error = e
                saw_inspection = saw_inspection or _is_content_inspection_error(e)
                continue

        raise Exception(
            f"All models failed after {len(tried_models)} attempts. Last error: {str(last_error)}"
        )
    # FEAT: add list_models and alist_models
    def list_models(self) -> List[str]:
        """
        Sync: return the list of backend model IDs this load-balancer can route to.
        """
        # make sure the model instances are initialized
        if not self._model_instances:
            self.initialize_models()
        # return the model_name of each adapter
        return [m.model_name for m in self._model_instances]

    async def alist_models(self) -> List[str]:
        """
        Async: same as list_models, but async signature.
        """
        # directly reuse the sync method
        return self.list_models()

    def format(self, *args: Union[Message, Sequence[Message]]) -> Any:
        """
        Format messages using the first model's formatter.
        
        This method delegates formatting to the first model in the pool.
        It's assumed that all models in the pool can handle the same format.
        
        Args:
            *args: Messages or sequences of messages.
            
        Returns:
            Formatted messages.
        """
        if not self._model_instances:
            raise ValueError(f"Load balancer '{self.config_name}' has no initialized models")

        # Use the first model for formatting
        return self._model_instances[0].format(*args)

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this load balancer instance.
        
        Returns:
            Dict containing load balancer configuration details.
        """
        base_info = super().get_info()

        if not self._model_instances:
            self.initialize_models()

        model_configs = [
            model.config_name if hasattr(model, "config_name") else str(model)
            for model in self._model_instances
        ]

        load_balancer_info = {
            **base_info,
            "model_type": f"{self.category.capitalize()}LoadBalancer",
            "strategy": self._strategy_name,
            "models": model_configs,
            "max_retries": self.max_retries
        }

        # Include model_name if specified
        if hasattr(self, 'model_name') and self.model_name:
            load_balancer_info["model_name"] = self.model_name

        return load_balancer_info 
