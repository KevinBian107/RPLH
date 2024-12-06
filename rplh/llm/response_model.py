from typing import Dict, List
from pydantic import BaseModel


class HCA(BaseModel):
    attitude: List[str]
    future_step: List[str]
    reasoning: str
    actions_plan: Dict


class HCA_AgentModel(HCA):
    agent_model: Dict[str, List[str]]
    actual_model: Dict[str, List[str]]
    strategy_model: Dict[str, List[str]]


class HCA_Judge(HCA_AgentModel):
    justification: str


class Judge(BaseModel):
    justification: str
    actions_plan: Dict


class LocalAgent(BaseModel):
    attitude: List[str]
    reasoning: str
    actions_plan: Dict
