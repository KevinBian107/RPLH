from typing import Dict, List
from pydantic import BaseModel

class HCA(BaseModel):
    future_step: List[str]
    reasoning: str
    actions_plan: Dict

class HCA_AgentModel(HCA):
    agent_model: Dict[str, str]
    spy_model: Dict[str, str]
    strategy_model: Dict[str, str]


class HCA_Judge(HCA_AgentModel):
    justification: str


class Judge(BaseModel):
    justification: str
    actions_plan: Dict

class LocalAgent(BaseModel):
    reasoning: str
    actions_plan: Dict
