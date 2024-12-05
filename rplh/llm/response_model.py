from typing import Dict, List
from pydantic import BaseModel

class HCA(BaseModel):
    attitude: List[str]
    world_model: List[str]
    future_step: List[str]
    reasoning: str
    actions_plan: Dict

class Judge(BaseModel):
    justification: str
    actions_plan: Dict

class LocalAgent(BaseModel):
    attitude: List[str]
    reasoning: str
    actions_plan: Dict
