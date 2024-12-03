# Why Efficient-RPLH System? 🤔
RPLH-efficient is the improved version of the original RPLH system where we prevent many of the fallbacks of RPLH-original hallucinating about many undpoable actions. We use the `instructor` package, which usese [**Fine State Machine**](https://dottxt-ai.github.io/outlines/latest/reference/generation/structured_generation_explanation/) that adjust the probability output of the language model to achieve specific format of the output. With this approach, it comes with tradeoff as well:

1. Markovian memory and limited prompt
    - Languag models tends to perform better when there are limited and specific information (prompt) given to them. We designed to system to be specifically utilizing this characteristic of these models.
2. Limited convesration ability of the LM agents, less hallucination of future plans and attitude.
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

3. Under this system, the judge also serve as an syntactic checker as well where it would change the wrong syntax that the HCA agent have given.
4. To remain the clearness of information flow and decision making, when needed to retake action, the LMs that is making the new decision would use teh original functions that the original HCA agent/Local agent/Judge uses with additional feedbacks giving through functional pass in with `functool.partial`.
5. To still promote some conversation going on, only output agent (HCA and judge) uses the strict output formatter on LMs, the local_agent and attitude agent outputs strings directly and is free to have more conversations.

## Optimization Tricks
We used many optimization hacks:

1. When an local agent does not have boxes or targets in its block or when it does not have an action provided from the HCA agent, it would not be considered as a valid conversation source as its views are limited.
2. When vote counts (consecutive agreement number) surpasses half the number of agents, the HCA action is executed directly. If not consecutive, the count is re-initiated.
3. If the local agent agrees the HCA decision, then no judge should be involved and the HCA action would be passed on directly (relapsed) to the next round of local agent or directly get executed.
4. Instead of letting the HCA agent figuring out attitude, we limit the prompt length that can be given by assigning an attitude agent taht judges the attitude of each agent on the field with current round information only, then only this attitude information is passed to teh next round HCA agent.