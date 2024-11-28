from typing import Dict, List
from pydantic import BaseModel

class HCA(BaseModel):
    attitude: List[str]
    reasoning: str
    future_step: str
    actions_plan: Dict

# We also need response_model for judge prompt
class Judge(BaseModel):
    justification: str
    actions_plan: Dict
