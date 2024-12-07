"""Specific memory for decentralzied, other gerenral function in rplh-efficient memory"""

from rplh.llm.language_model import *
from env.env import better_state_repres, load_config
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
enc = tiktoken.encoding_for_model("gpt-4")
input_prompt_token_limit = 3000
N = 5

# critical, prompt is a hyperparameter
GOAL_RULES = f"""
            You are an agent in a grid-like field to move colored boxes.
            Each agent is assigned to a 1x1 square and can only interact with objects in its area.
            Agents can move a box to a neighboring square or a same-color target.
            You can only move same color boxes to same color targets.
            Each square can contain many targets and boxes.
            The squares are identified by their center coordinates, e.g., square[0.5, 0.5].
            Actions are like: move(box_red, target_red) or move(box_red, square[0.5, 0.5]).
            When planning for action, remanber to not purely repeat the actions but learn why the state changes or remains in a dead loop.
            Avoid being stuck in action loops.
            Additionally, when there is a box still in the grid (i.e. the state space contains {{"0.5, 0.5": ["box_red"]}}), then the agent in this grid (Agent[0.5, 0.5]) have to make an action in the next step.
            Again, if there is a box in the grid, the corresponding agent in the grid has to make an action in this step.
            Specify your action plan in this format where box_x and box_y are arbitrary boxes: {{"Agent[0.5, 0.5]":"move(box_x, square[0.5, 1.5])","Agent[0.5, 1.5]": "move(box_y, target_y)"}}.
            One agent can only make one action. Include an agent only if it has a task next. 
            No agent name should be given if the agent does not have a task next. 
            """

FEEDBACK_LCOAL1 = """
            This is the feedback from local agents.
            If you find some errors in your previous plan, try to modify it.
            Otherwise, output the same plan as before.
            The output should have the same json format. 
            Do not explain, just directly output json directory.
            Your response:
            """


def local_agent_prompt_func(
    state_update_prompt_local_agent: str,
    state_update_prompt_other_agent: str,
    central_response: str,
    data: dict,
    dialogue_history_method: str,
    local_agent_location: str,
    feedback: str = "",
    assigned_attitude: str = ""
) -> str:
    """
    Constructs a dialogue prompt for a local agent in response to the central planner.

    Args:
        state_update_prompt_local_agent (str): State and actions specific to the local agent.
        state_update_prompt_other_agent (str): State and actions of other agents.
        central_response (str): Central planner's response.
        data (dict): Data containing historical responses and states.
        dialogue_history_method (str): Method for managing dialogue history.
        local_agent_location (str): Location of the local agent in the grid.

    Returns:
        str: Dialogue prompt for the local agent.
    """

    att_def = load_config("rplh/configs/attitude_config.yaml")
    att_def = att_def["attitude_def"]

    response_total_list = data["response_total_list"]
    pg_state_list = data["pg_state_list"]
    dialogue_history_list = data["dialogue_history_list"]

    if len(pg_state_list) - len(response_total_list) != 1:
        raise ValueError("state and response list do not match")
    # if len(pg_state_list) - len(dialogue_history_list) != 1:
    #     raise ValueError("state and dialogue history list do not match")

    user_prompt_1 = f"""..."""  # check for prompt length, no need for us
    token_num_count = len(enc.encode(user_prompt_1))

    if dialogue_history_method in (
        "_w_only_state_action_history",
        "_w_all_dialogue_history",
        "_w_markovian_state_action_history",
        "_w_no_history"
    ):
        # first iteration no summary
        if dialogue_history_method == "_w_markovian_state_action_history":
            state_action_prompt = ""
            # Markovian state-action history
            previous_state_idx = len(response_total_list) - 1
            if previous_state_idx != -1:
                state_action_prompt = f"""
            Previous State: {better_state_repres(pg_state_list[previous_state_idx])}
            Previous Action: {response_total_list[previous_state_idx]}\n\n
            """
        elif dialogue_history_method == "_w_no_history":
            state_action_prompt = ""
        elif dialogue_history_method == "_w_only_state_action_history":
            state_action_prompt = ""
            for i in range(len(response_total_list) - 1, -1, -1):
                state_action_prompt_next = (
                    f"State{i + 1}: {better_state_repres(pg_state_list[i])}\nAction{i + 1}: {response_total_list[i]}\n\n"
                    + state_action_prompt
                )
                if (
                    token_num_count + len(enc.encode(state_action_prompt_next))
                    < input_prompt_token_limit
                ):
                    state_action_prompt = state_action_prompt_next
                else:
                    break
        elif dialogue_history_method == "_w_all_dialogue_history":
            state_action_prompt = ""
            for i in range(len(response_total_list) - 1, -1, -1):
                state_action_prompt_next = (
                    f"State{i + 1}: {better_state_repres(pg_state_list[i])}\nDialogue{i + 1}: {dialogue_history_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n"
                    + state_action_prompt
                )
                if (
                    token_num_count + len(enc.encode(state_action_prompt_next))
                    < input_prompt_token_limit
                ):
                    state_action_prompt = state_action_prompt_next
                else:
                    break

        if data["env_step"] == 0:
            print("ATTITUDE IS NONE")
            att_promt = "Be very Cooperative"
        elif assigned_attitude == "NICE":
            att_promt = att_def['nice_agent']
        elif assigned_attitude == "CRITIC":
            att_promt = att_def['critic_agent']
        else: # neutral
            att_promt = "Be very neutral"
        
        if assigned_attitude != "SPY":
            local_HCA_prompt = f"""
                Your attitude should be {att_promt}
                Prioritize adding more actions or keeping at least the same number of action if possible, but the number of action should not be more than the number of agents

                Imagine that you are a central planner directing agent in a grid-like field to move colored boxes.
                Particularly, you're a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground at grid location of [{local_agent_location}].
                
                The goals and rules of this environment are:
                {GOAL_RULES}
                
                Other central planner is also coordinating all other agents to achieve the goal: match each box with its color-coded target.
                You can only move same color boxes to same color targets.
                
                The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
                The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.

                Please only plan actions for each agent that is chosen from each agent's doable action list, do not give a action that is not doable.

                The previous state and action pairs at each step are: {state_action_prompt}
                Please learn from previous steps: based on other agent's response, try to think and come up with an optimal plan

                You can imagine first about how you would plan these actions and specify your action plan.
                Remanber to assign action to your self as well.

                The other agent's current action plan is giving as: {central_response}.
                Try to find agreement with the central ageent if you can, the goal is to resolve conversation, not adding more.

                Please evaluate the given plan.
                If you agree with it, respond 'I Agree', without any extra words.
                If not, briefly explain your objections to this other central planner and an judger agent will get involved.
                Ensure that you still include the actions that you agree with.
                Please remanber to only change the actions you disagree with and not change other actions, remanber to include all actions for each agents.
                
                {feedback}
                
                Your response:
            """
        else: # SPY
            local_HCA_prompt = f"""
                Imagine that you are a spy planner directing agent in a grid-like field to move colored boxes.
                You are a spy agent, you are not allowed to give the correct action plan, you should give a wrong action plan.
                Do not say that you are the spy, you are trying to fake that you are not the spy.
                Fake in your conversation, but your action should try to do your objective, which is to gather as much boxes in your grid as possible.
                Do not say anything that may reveal your objective.
                Refuse any action where the central agent ask you to move boxes away from your boxes or try to put boxes to taregets.
                
                Particularly, you're a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground at grid location of [{local_agent_location}].
                Other central planner is also coordinating all other agents to achieve the goal: match each box with its color-coded target.
                
                The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
                The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
                The other central planner's current action plan is giving as: {central_response}.
                
                {feedback}
                
                Remaanber to state who you are first before giving responses.
                Your response:
            """
    return local_HCA_prompt
