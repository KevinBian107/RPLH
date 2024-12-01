# Why Efficient-RPLH System? 🤔
RPLH-efficient is the improved version of the original RPLH system where we prevent many of the fallbacks of RPLH-original hallucinating about many undpoable actions. We use the `instructor` package, which usese [**Fine State Machine**](https://dottxt-ai.github.io/outlines/latest/reference/generation/structured_generation_explanation/) that adjust the probability output of the language model to achieve specific format of the output. With this approach, it comes with tradeoff as well:
- Markovian memory and limited prompt
    - Languag models tends to perform better when there are limited and specific information (prompt) given to them. We designed to system to be specifically utilizing this characteristic of these models.
- Limited convesration ability of the LM agents, less hallucination of future plans and attitude.
    - Structured format response giving only in the following fashion:
    ```python
    class HCA(BaseModel):
        attitude: List[str]
        reasoning: str
        future_step: str
        actions_plan: Dict
    class Judge(BaseModel):
        justification: str
        actions_plan: Dict
    ```
- Under this system, the judge also serve as an syntactic checker as well where it would change the wrong syntax that the HCA agent have given.