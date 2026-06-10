from onesim.planning.base import PlanningBase
from onesim.models.core.message import Message
from loguru import logger


class COTPlanning(PlanningBase):
    def __init__(self,model_config_name,sys_prompt):
        super().__init__(model_config_name,sys_prompt)

    async def plan(self,**kwargs) -> str:
        prompt=f"""
        ### Agent Profile
        {kwargs["profile"]}

        ### Memory
        {kwargs["memory"]}

        
        ### Observation
        {kwargs["observation"]}
        
        ### Instruction
        {kwargs["instruction"]}

        Please think step by step based on the above concisely.
        """
        prompt=self.model.format(
            Message("system", self.sys_prompt, role="system"),
            Message("user", prompt, role="user")
        )
        logger.info(f"COTPlanning plan prompt entry (model={getattr(self.model, 'config_name', '?')})")
        response = await self.model.acall(prompt)
        logger.info(f"COTPlanning plan response exit (model={getattr(self.model, 'config_name', '?')})")
        return response.text
