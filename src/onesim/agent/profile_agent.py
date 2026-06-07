# -*- coding: utf-8 -*-
"""A general dialog agent."""
from typing import Optional, Union, Sequence, Any

from loguru import logger

from onesim.models.core.message import Message
from .base import AgentBase
import json
from onesim.models.core.message import Message
from onesim.models import JsonBlockParser

class ProfileAgent(AgentBase):
    """A simple agent used to perform a dialogue. Your can set its role by
    `sys_prompt`."""

    def __init__(
        self,
        model_config_name: str,
        sys_prompt: str ='',
        **kwargs: Any,
    ) -> None:
        """Initialize the dialog agent.

        Arguments:
            name (`str`):
                The name of the agent.
            sys_prompt (`Optional[str]`):
                The system prompt of the agent, which can be passed by args
                or hard-coded in the agent.
            model_config_name (`str`):
                The name of the model config, which is used to load model from
                configuration.
        """
        super().__init__(
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
        )
        if sys_prompt == '':
            self.sys_prompt = """You are an agent specialized in analyzing Multi-Agent system. Your task is to identify and extract distinct agent types from a given description. The agent types should be returned in PascalCase format, suitable for use as class names in code."""
        if kwargs:
            logger.warning(
                f"Unused keyword arguments are provided: {kwargs}",
            )

    def assign_agent_portraits(self, agent_types_dict):
        """Assign portrait types to agent categories.
        
        Args:
            agent_types_dict: Dictionary with agent types as keys and descriptions as values.
            
        Returns:
            Dictionary mapping agent types to portrait IDs where:
            1 = Government Official
            2 = Researcher/Scholar
            3 = Worker/Laborer
            4 = Merchant/Business Person
            5 = Citizen
        """
        prompt = f"""
        You are assigning social role portraits to different agent types in a multi-agent system.
        
        The available portrait types are:
        1. Government Official - Represents authority, policy-making, regulation, and public administration
        2. Researcher/Scholar - Represents academia, expertise, analysis, and knowledge creation
        3. Worker/Laborer - Represents practical skills, manual labor, and production
        4. Merchant/Business Person - Represents commerce, trade, business acumen, and entrepreneurship
        5. Citizen - Represents the general public, consumers, and community members
        
        For the following agent types and their descriptions, assign ONE portrait type (1-5) that best matches each agent's role and characteristics:
        
        {json.dumps(agent_types_dict, indent=2)}
        
        Requirements:
        1. Each agent type should be assigned EXACTLY ONE portrait type (1-5)
        2. Try to distribute portrait types so that not all agents have the same portrait
        3. The assignment should be logical and reflect the agent's role in the system
        4. Return your answer as a JSON object with agent types as keys and portrait IDs (integers 1-5) as values
        
        Example output format:
        ```json
        {{
          "AgentType1": 2,
          "AgentType2": 4,
          "AgentType3": 1
        }}
        ```
        """

        parser = JsonBlockParser()
        prompt = self.model.format(
            Message("system", self.sys_prompt, role="system"),
            Message("user", prompt + parser.content_hint, role="user")
        )
        response = self.model(prompt)

        try:
            res = parser.parse(response)
            portrait_assignments = res.parsed
            return portrait_assignments
        except json.JSONDecodeError:
            raise ValueError("LLM response is not valid JSON.")

    def generate_agent_types(self,description):

        prompt = (
            f"Given the following description: {description}, identify or infer relevant agent types "
            "and return them as a JSON object where keys are PascalCase agent type names and values are short descriptions of each agent type.\n\n"
            "Requirements:\n"
            "1. Each agent type should be in English and formatted in PascalCase (capitalize each word, with no spaces or special characters).\n"
            "2. Ensure each agent name is concise, clearly reflects its role, and accurately represents the agent's primary function within a multi-agent system.\n"
            "3. Each agent type should include a brief description (1-2 sentences) explaining its role and responsibility.\n"
            "4. If no explicit agent types are present in the description, infer and include plausible agent types to establish a functional multi-agent system based on the described roles and actions.\n\n"
            "Return the agent types as a JSON object with the following format:\n"
            "```json\n"
            "{\n"
            "  \"AgentType1\": \"Brief description of AgentType1's role\",\n"
            "  \"AgentType2\": \"Brief description of AgentType2's role\"\n"
            "}\n"
            "```"
        )
        parser=JsonBlockParser()
        prompt = self.model.format(
            Message("system", self.sys_prompt, role="system"),
            Message("user", prompt+parser.content_hint, role="user")
        )
        response = self.model(prompt)
        # Parse the LLM's JSON response
        try:

            res=parser.parse(response)
            agent_types = res.parsed
            return agent_types
        except json.JSONDecodeError:
            raise ValueError("LLM response is not valid JSON.")

    def merge_relationships(self,relationships):
        """
        Merge and deduplicate relationships.
        """
        # Create a dictionary to store merged relationships
        merged = {}
        for rel in relationships:
            src = rel['source_agent']
            tgt = rel['target_agent']
            rel_type = rel['relationship_type']
            direction = rel['direction']

            key = (src, tgt)
            reciprocal_key = (tgt, src)

            if reciprocal_key in merged:
                existing_rel = merged[reciprocal_key]
                # Determine the new direction
                if existing_rel['direction'] == 'bidirectional' or direction == 'bidirectional':
                    new_direction = 'bidirectional'
                else:
                    new_direction = existing_rel['direction']  # Both unidirectional in the same direction

                # Combine relationship types if necessary
                if existing_rel['relationship_type'] != rel_type:
                    combined_type = f"{existing_rel['relationship_type']} & {rel_type}"
                else:
                    combined_type = existing_rel['relationship_type']

                merged[reciprocal_key] = {
                    'source_agent': tgt,
                    'target_agent': src,
                    'relationship_type': combined_type,
                    'direction': new_direction
                }
            else:
                merged[key] = rel

        # Convert the merged dict back to a list
        merged_relationships = list(merged.values())

        return merged_relationships

    def generate_relationship_schema(self, agent_types, actions, events):
        """Generate relationship schema based on events graph.
        
        Args:
            agent_types: List of agent types
            actions: Dictionary of actions
            events: Dictionary of events
            
        Returns:
            List of relationship objects with source_agent, target_agent, relationship_type, and direction
        """
        # Track connections between agent types
        connections = {}

        # Analyze events to build connection graph
        for event_id, event in events.items():
            from_agent = event['from_agent_type']
            to_agent = event['to_agent_type']

            # Skip environment agent connections
            if from_agent == 'EnvAgent' or to_agent == 'EnvAgent':
                continue

            # Initialize connection if not exists
            key = (from_agent, to_agent)
            reverse_key = (to_agent, from_agent)

            if key not in connections and reverse_key not in connections:
                connections[key] = {
                    'forward': False,  # Connection from source to target
                    'reverse': False   # Connection from target to source
                }

            # Mark connection direction
            if key in connections:
                connections[key]['forward'] = True
            else:
                connections[reverse_key]['reverse'] = True

        # Generate relationships based on connections
        relationships = []
        for (source, target), connection in connections.items():
            # Determine direction based on connections
            if connection['forward'] and connection['reverse']:
                direction = 'bidirectional'
            else:
                direction = 'unidirectional'

            # Create relationship object
            relationship = {
                'source_agent': source,
                'target_agent': target,
                # 'relationship_type': f'Workflow interaction between {source} and {target}',
                'direction': direction
            }
            relationships.append(relationship)

        return relationships

    def generate_profile_schema(self, scenario_description, agent_name, agent_data_model):
        prompt = """ Please generate a Profile Config Schema for an Agent in a multi-agent scenario, based on the following scenario description, Agent name, and existing data model. The Schema should adhere to the following rules:
    1. **Scenario Description**: {scenario_description}
    2. **Agent Name**: {agent_name}
    3. **Existing Data Model**: {agent_data_model}

    4. **Schema Format**:
    - IMPORTANT: Return a FLAT JSON structure. DO NOT use nested structures with "properties" or "required" fields.
    - IMPORTANT: Each field should be directly at the root level of the JSON object.
    - DO NOT use any JSON Schema validation format. Just provide the direct field definitions.

    5. **Schema Guidelines**:
    - The schema MUST contain a 'name' field as a required property.
    - IMPORTANT: Do NOT include any field with 'id' in its name (such as 'agent_id', 'worker_id', etc.), as unique identifiers will be assigned separately.
    - Incorporate all relevant attributes from the provided data model.
    - Extend the schema beyond the existing data model to create a comprehensive profile.
    - Each property in the Schema should represent either:
        a) A static profile attribute that defines the agent's characteristics
        b) A dynamic runtime variable that changes during simulation
    - Each property should include the following fields:
        - **"type"**: The data type of the property, which can be "int", "str", "list", or "float". ONLY use these four types.
        - **"default"**: The default value of the property (see improved guidelines below).
        - **"private"**: A boolean indicating whether the property is visible to others.
        - **"sampling"**: Specifies how the property's value is generated:
        - "llm": For static attributes needing language model generation
        - "random": For static attributes with defined choices/ranges
        - "default": For dynamic variables that change during simulation
        - **"range"** or **"choices"**: Used to limit possible values for static attributes when sampling is "random"
        - **"sample_size"**: When type is "list", specifies number of items to sample (for static attributes)
        - **"description"**: Required when sampling is "llm"; provides generation guidance

    6. **IMPROVED DEFAULT VALUE GUIDELINES**:
    - For properties with sampling="llm" or sampling="random": 
        - Provide sensible example values that might be generated
    
    - For properties with sampling="default" (dynamic variables):
        - AVOID empty strings, zeros, or empty lists unless absolutely necessary
        - Instead, provide realistic initial values that make sense for the simulation start:
        - For "str" type: Use a meaningful initial state (e.g., "awaiting_input", "ready", "inactive")
        - For "int"/"float" types: Use plausible starting values (e.g., 100 for points, 50 for percentage)
        - For "list" type: Include at least one sample item that demonstrates the expected structure
        - These meaningful defaults will:
        - Make the simulation behave more realistically from the start
        - Provide examples of the expected data structure
        - Avoid null/empty value errors during simulation

    7. **Additional Requirements**:
    - For ALL properties, ensure that default values are relevant to the scenario context
    - The Schema should be output in valid JSON format
    - Only return the JSON Schema, do not include other text/comments
    - CRITICAL: Return a FLAT structure directly at the root level. DO NOT include "properties" or "required" fields.

        **Examples of GOOD vs. POOR Default Values for Dynamic Variables**:

        *POOR (avoid this pattern):*
        ```json
        {{
            "transaction_history": {{
                "type": "list",
                "default": [],
                "private": true,
                "sampling": "default"
            }},
            "current_status": {{
                "type": "str",
                "default": "",
                "private": false,
                "sampling": "default"
            }},
            "customer_satisfaction": {{
                "type": "int",
                "default": 0,
                "private": false,
                "sampling": "default"
            }}
        }}
        ```

        *GOOD (use this pattern):*
        ```json
        {{
            "transaction_history": {{
                "type": "list",
                "default": [{{ "date": "2025-04-01", "amount": 125.50, "type": "initial" }}],
                "private": true,
                "sampling": "default"
            }},
            "current_status": {{
                "type": "str",
                "default": "available",
                "private": false,
                "sampling": "default"
            }},
            "customer_satisfaction": {{
                "type": "int",
                "default": 75,
                "private": false,
                "sampling": "default"
            }}
        }}
        ```

    8. **Expected Output Format**:
    - Your output should look exactly like the example below, with each field directly at the root level:
    
    ```json
    {{
        "name": {{
            "type": "str",
            "default": "John Smith",
            "private": false,
            "sampling": "llm",
            "description": "The agent's full name"
        }},
        "age": {{
            "type": "int",
            "default": 35,
            "private": false,
            "sampling": "random",
            "range": [25, 65]
        }},
        ... other fields directly at root level ...
    }}
    ```

    IMPORTANT: Do NOT add any nested "properties" or "required" fields. Each field should be a direct key-value pair at the root of the JSON object.

    Respond code in a markdown's fenced code block as follows:
    ```json
    Your Profile Config Schema here
    ```
    """.format(scenario_description=scenario_description, agent_name=agent_name,agent_data_model=agent_data_model)
        parser=JsonBlockParser()
        prompt = self.model.format(
            Message("system", self.sys_prompt, role="system"),
            Message("user", prompt+parser.format_instruction, role="user")
        )
        response = self.model(prompt)
        # Parse the LLM's JSON response
        try:
            res=parser.parse(response)
            agent_schema = res.parsed
            for key, value in agent_schema.items():
                if value['type']=='array':
                    value['type']='list'
                elif value['type']=='string':
                    value['type']='str'

            agent_schema['id']={"type": "str",
            "default": "0",
            "private": True,
            "sampling": "default",
            "description": "Unique id of agent"}
            return agent_schema
        except json.JSONDecodeError:
            raise ValueError("LLM response is not valid JSON.")

    def generate_env_data(self, env_data_schema, description):
        prompt = f""" Transform the provided environment data schema into a realistic JSON object with contextually appropriate values for simulation.

        **Simulation Context**: {description}

        **Input Schema**: {env_data_schema}

        **Transformation Requirements**:
        - Extract each variable from the "variables" array in the input schema
        - For each variable:
          - Use the "name" field as the key in the output
          - For all cases, generate a REALISTIC, SPECIFIC value based on:
            - The simulation context described above
            - The variable's name and type
            - Real-world plausibility for the scenario

        **Value Generation Guidelines**:
        - **"list"**: Generate 3-5 diverse, specific items relevant to the context
        - **"str"**: Create descriptive, specific strings (avoid generic terms like "value" or "item")
        - **"int"**: Use realistic numbers within expected ranges for the domain (avoid 0, 1, or round numbers)
        - **"float"**: Use precise decimal values with appropriate precision for the context
        - **"bool"**: Choose true/false based on what makes sense in the scenario

        **IMPORTANT**: 
        - NEVER use null, empty strings, zeros, or placeholder values
        - Make values varied and realistic as if from a real-world dataset
        - Ensure all generated values directly relate to the simulation context
        - Consider relationships between variables (e.g., if generating inventory and demand, make them proportional)
        
        **Examples of Good vs Poor Value Generation**:

        For a retail simulation context:
        
        POOR:
        ```json
        {{
            "product_price": "0",
            "customer_demand": "100",
            "store_location": "Location"
        }}
        ```
        
        GOOD:
        ```json
        {{
            "product_price": "24.99",
            "customer_demand": "843",
            "store_location": "Downtown Shopping Center"
        }}
        ```
        
        Return ONLY the transformed key-value JSON object without any additional explanations or comments.
        """

        parser = JsonBlockParser()
        prompt = self.model.format(
            Message("system", self.sys_prompt, role="system"),
            Message("user", prompt + parser.format_instruction, role="user")
        )
        response = self.model(prompt)
        # Parse the LLM's JSON response
        try:
            res = parser.parse(response)
            initialized_data = res.parsed
            return initialized_data
        except json.JSONDecodeError:
            raise ValueError("LLM response is not valid JSON.")
