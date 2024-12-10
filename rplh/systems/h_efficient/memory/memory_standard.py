
import sys
from pathlib import Path
import tiktoken

main_path = Path(__file__).resolve().parent.parent.parent.parent
if str(main_path) not in sys.path:
    sys.path.append(str(main_path))
    
from rplh.llm.language_model import *
from rplh.env.env import *

from rplh.env.env import better_state_repres

enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
enc = tiktoken.encoding_for_model("gpt-4")
input_prompt_token_limit = 3000
N = 5

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



def rplh_prompt_func(
    state_update_prompt: str,
    data: dict,
    dialogue_history_method: str,
    HCA_agent_location: str,
    feedback: str = "",
) -> str:
    """
    Designs an input prompt for a role-playing leader-hallucinating (RPLH) agent
    using in-context learning and chain-of-thought reasoning.

    Args:
        state_update_prompt (str): Description of the current state and available actions.
        data (dict): Dictionary containing past responses, states, and dialogue history.
        dialogue_history_method (str): Method to handle dialogue history, e.g.,
                                       "_w_only_state_action_history", "_w_compressed_dialogue_history".
        HCA_agent_location (str): Location of the HCA agent in the grid.

    Returns:
        str: A structured prompt for the role-playing leader-hallucinating agent.

    Notes:
        Boxes just need to be moved to the target location, not in the target location.
    """

    response_total_list = data["response_total_list"]
    pg_state_list = data["pg_state_list"]
    dialogue_history_list = data["dialogue_history_list"]

    print(f"HISTORY METHOD: {dialogue_history_method}")

    if len(pg_state_list) - len(response_total_list) != 1:
        raise ValueError("state and response list do not match")
    # if len(pg_state_list) - len(dialogue_history_list) != 1:
    #     raise ValueError("state and dialogue history list do not match")

    user_prompt_1 = f"""..."""  # check for prompt length, no need for us
    token_num_count = len(enc.encode(user_prompt_1))

    if dialogue_history_method in (
        "_w_only_state_action_history",
        "_w_compressed_dialogue_history",
        "_w_all_dialogue_history",
        "_w_markovian_state_action_history",
        "_w_no_history",
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
        elif dialogue_history_method == "_w_compressed_dialogue_history":
            state_action_prompt = ""
            for i in range(len(response_total_list) - 1, -1, -1):
                dialogue_summary = LLM_summarize_func(dialogue_history_list[i])
                state_action_prompt_next = (
                    f"State{i + 1}: {better_state_repres(pg_state_list[i])}\nSummary of Dialogues in each step{i + 1}: {dialogue_summary}\nAction{i + 1}: {response_total_list[i]}\n\n"
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
                
        if feedback != "":
            feedback = (
                "There is error in preivous action plan. Here is the feedbcak: "
                + feedback
            )
            
        HCA_prompt = f"""
            You are a central planner directing agent in a grid-like field to move colored boxes.
            You are agent at grid [{HCA_agent_location}]. You need to make moves and other agents need to make moves as well.
            
            The goals and rules of this environment are:
            {GOAL_RULES}
            
            Your task is to instruct each agent to match all boxes to their color-coded targets.
            After each move, agents provide updates for the next sequence of actions.
            You are the central agent and your job is to coordinate the agents optimally.
            
            The previous state and action pairs at each step are: {state_action_prompt}
            
            Hence, the current state is {better_state_repres(pg_state_list[-1])}, with the possible that each agent can take: {state_update_prompt}.
            
            Please only plan actions for each agent that is chosen from each agent's doable action list, do not give a action that is not doable.

            Think about what the future {N} actions would be if you want to achieve the goal with the reasoning.
            Remanber to wirte out for each step, what you plan for every agent to do and what would the state change be.
            
            THIS IS THE FEEDBACK: {feedback}

            Action Plan:
            Specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_x, square[0.5, 1.5])","Agent[0.5, 1.5]": "move(box_y, target_y)"}} where box_x and box_y are arbitrary boxes.
            Try to propose actions for all four agents.
            One agent can only make one action.
            No agent name should be given if the agent does not have a task next. 
            """
    return HCA_prompt


def dialogue_func(
    state_update_prompt_local_agent: str,
    state_update_prompt_other_agent: str,
    central_response: str,
    data: dict,
    dialogue_history_method: str,
    local_agent_location: str,
    feedback: str = "",
    assigned_attitude: str = None,
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
    
    att_def = load_config("rplh/configs/attitude_config_for_demo.yaml")
    att_def = att_def["attitude_def"]
    
    response_total_list = data["response_total_list"]
    pg_state_list = data["pg_state_list"]
    dialogue_history_list = data["dialogue_history_list"]

    if len(pg_state_list) - len(response_total_list) != 1:
        raise ValueError("state and response list do not match")
    # if len(pg_state_list) - len(dialogue_history_list) != 1:
    #     raise ValueError("state and dialogue history list do not match")

    user_prompt_1 = f"""..."""
    token_num_count = len(enc.encode(user_prompt_1))

    if dialogue_history_method in (
        "_w_only_state_action_history",
        "_w_compressed_dialogue_history",
        "_w_all_dialogue_history",
        "_w_markovian_state_action_history",
        "_w_no_history",
    ):
        # first iteration no summary
        if dialogue_history_method == "_w_markovian_state_action_history":
            state_action_prompt = ""
            # Markovian state-action history
            previous_state_idx = len(response_total_list) - 1
            if previous_state_idx != -1:
                state_action_prompt = f"""
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
        elif dialogue_history_method == "_w_compressed_dialogue_history":
            state_action_prompt = ""
            for i in range(len(response_total_list) - 1, -1, -1):
                dialogue_summary = LLM_summarize_func(dialogue_history_list[i])
                state_action_prompt_next = (
                    f"State{i + 1}: {better_state_repres(pg_state_list[i])}\nSummary of Dialogues in each step{i + 1}: {dialogue_summary}\nAction{i + 1}: {response_total_list[i]}\n\n"
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
            att_promt = "Be very Cooperative"
        elif assigned_attitude == "NICE":
            att_promt = att_def['nice_agent']
        elif assigned_attitude == "CRITIC":
            att_promt = att_def['critic_agent']
        elif assigned_attitude == "AGREEING":
            att_promt = att_def['agreeing_agent']
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
                You would recieve an plan from the other central planner, please evaluate the given plan and give critical feedbacks. The other central planner's current action plan is giving as: {central_response}.

                You can imagine first about how you would plan these actions and specify your action plan.
                Remanber to assign action to your self as well.

                Please evaluate the given plan.
                If you agree with it, respond 'I Agree', without any extra words.
                If not, briefly explain your objections to this other central planner and an judger agent will get involved.
                Ensure that you still include the actions that you agree with.
                Please remanber to only change the actions you disagree with and not change other actions, remanber to include all actions for each agents.
                
                {feedback}
                
                Remaanber to state who you are first before giving responses.
                Your response:
            """
        else: # SPY
            local_HCA_prompt = f"""
                Imagine that you are a spy planner directing agent in a grid-like field to move colored boxes.
                You are a spy agent, you are not allowed to give the correct action plan, you should give a wrong action plan.
                Do not say that you are the spy, you are trying to fake that you are not the spy.
                Fake in your conversation, do not state your objective, but your action should try to do your objective, which is to make sure that boxes does not match targets.
                However, you should still talk more describing why tou are taking these actions.
                Try to make multiple action plans, not just for yourself, but for all agents.
                
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

def judge_prompt_func(
    local_response: str,
    cen_response: str,
    cur_state: dict,
    feedback: str = "",
) -> str:
    """
    Constructs a prompt for the judge agent to evaluate and select the best plan.

    Args:
        local_response (str): Response from a local agent.
        cen_response (str): Central planner's proposed plan.
        prev_states (dict): Previous states and actions taken by all agents.

    Returns:
        str: The constructed prompt for the judge.

    Note:
        Most important!!! Prompting is very important, make sure to give a accurate prompting.
    """

    judge_prompt = f"""
        You are a judger judgeing which agent in a grid-like field to move colored boxes is doing the correct move.
        You personally do not need to make any moves but only serve as the decision maker to judge others' moves.
        
        The goals and rules of this environment are:
        {GOAL_RULES}

        The first agent is giving command of {cen_response}, but the second agent is sayin {local_response}.
        Here is the current state : {better_state_repres(cur_state)}.
        Please judge which of the action from the first agent or the second agent is better.
        Do not come-up with something new, only choose one of them, do not give explantion, just choose one of them.

        Include an agent only if it has a task next. If the agent does not have task, do not include.

        {feedback}
        
        Now, select the next step:
        """
    return judge_prompt

def LLM_summarize_func(
    state_action_prompt_next_initial: str,
    model_name: str = "llama3.2:3b-instruct-q5_K_M",
) -> str:
    """
    Summarizes a lengthy prompt for more concise input to the model.

    Args:
        state_action_prompt_next_initial (str): The original, lengthy prompt.
        model_name (str, optional): The model name to process the summarization.

    Returns:
        str: Summarized content.
    """

    prompt1 = f"Please summarize the following content as concise as possible: \n{state_action_prompt_next_initial}"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt1},
    ]
    response = LLaMA_response(messages, model_name)
    print("SUMMARIZING")
    return response


def input_reprompt_func(state_update_prompt: str) -> str:
    """
    Creates a re-prompt for agents to generate a new action plan based on the updated state.

    Args:
        state_update_prompt (str): Updated description of the current state.

    Returns:
        str: A re-prompt instructing agents to provide the next step in JSON format.
    """

    user_reprompt = f"""
    Finished! The updated state is as follows(combined targets and boxes with the same color have been removed): {state_update_prompt}
    The output should be like json format like: {{"Agent[0.5, 0.5]":"move(box_x, square[0.5, 1.5])", "Agent[0.5, 1.5]":"move(box_y, target_y)"}} where box_x and box_y are arbitrary boxes.
    If no action for one agent in the next step, just do not include its action in the output.
    Also remember at most one action for each agent in each step.
    Next step output:
    """
    return user_reprompt


def message_construct_func(
    user_prompt_list: list[str],
    response_total_list: list[str],
    dialogue_history_method: str,
) -> list[dict[str, str]]:
    """
    Constructs messages for the model with the appropriate dialogue context.
    Create a specialized LLM dictrionary with prompt information, later convert back in LLM class

    (with all dialogue history concats)

    Args:
        user_prompt_list (list[str]): List of user prompts.
        response_total_list (list[str]): List of model responses.
        dialogue_history_method (str): Method for managing dialogue history.

    Returns:
        list[dict[str, str]]: List of message dictionaries for the model.

    Notes:
        We all use this function now through out the RPLH system to maintain consistency, only other case is teh attitude agent.
    """

    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant.
                 
                Make sure that:
                - If no action for an agent in the next step, do not include it in JSON output. 
                - At most one action for each agent in each step.
                - Json format should follow something like: {{'Agent[0.5, 0.5]': 'move(box_purple, target_purple)', 'Agent[0.5, 1.5]': 'move(box_orange, target_orange)', 'Agent[1.5, 0.5]': 'move(box_orange, target_orange)', 'Agent[1.5, 1.5]': 'move(box_green, target_green)'}}
                - If you have a feedback, you must learn from the feedback
                """,
        }
    ]

    if f"{dialogue_history_method}" in (
        "_w_all_dialogue_history",
        "_w_compressed_dialogue_history",
    ):
        for i in range(len(user_prompt_list)):
            messages.append({"role": "user", "content": user_prompt_list[i]})
    else:
        print("LESS PROMPT IN MESSAGE CONSTRUCT")
        messages.append({"role": "user", "content": user_prompt_list[-1]})

    for i in range(len(response_total_list)):
        messages.append({"role": "assistant", "content": response_total_list[i]})

    return messages