from typing import Dict, List
from pydantic import BaseModel

class LocalAgent(BaseModel):
    attitude: List[str]
    reasoning: str
    actions_plan: Dict