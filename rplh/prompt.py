from LLM import *
import tiktoken
from typing import Dict, List, Tuple, Union

enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
enc = tiktoken.encoding_for_model("gpt-4")
input_prompt_token_limit = 3000
N = 5

# critical, prompt is a hyperparameter
GOAL_RULES = f"""You are an agentin a grid-like field to move colored boxes.
                Each agent is assigned to a 1x1 square and can only interact with objects in its area.
                Agents can move a box to a neighboring square or a same-color target.
                You can only move same color boxes to same color targets.
                Each square can contain many targets and boxes.
                The squares are identified by their center coordinates, e.g., square[0.5, 0.5].
                Actions are like: move(box_red, target_red) or move(box_red, square[0.5, 0.5]).
                When planning for action, remanber to not purely repeat the actions but learn why the state changes or remains in a dead loop.
                Avoid being stuck in action loops.
                Additionally, when there is a box still in the grid (i.e. the state space contains {{"0.5_0.5": ["box_red"]}}), then the agent in this grid (Agent[0.5, 0.5]) have to make an action in the next step.
                Again, if there is a box in the grid, the corresponding agent in the grid has to make an action in this step.
                One agent can make any numbers of action if needed (i.e. {{"Agent[1.5, 1.5]":"move(box_blue, square[0.5, 1.5])","Agent[1.5, 1.5]": "move(box_green, target_green)"}}).
                Specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])","Agent[0.5, 1.5]": "move(box_blue, target_blue)"}}.
                Include an agent only if it has a task next. No agent name should be given if the agent does not have a task next.
                You do not need to say json format, just use it directly in the format of {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[0.5, 1.5]": "move(box_blue, target_blue)"}}.
                """


def judge_prompt_func(local_response: str, cen_response: str, cur_state: Dict) -> str:
    """
    Constructs a prompt for the judge agent to evaluate and select the best plan.

    Args:
        local_response (str): Response from a local agent.
        cen_response (str): Central planner's proposed plan.
        prev_states (Dict): Previous states and actions taken by all agents.

    Returns:
        str: The constructed prompt for the judge.

    Note:
        Most important!!! Prompting is very important, make sure to give a accurate prompting.
    """

    judge_prompt = f"""
        You a a judger judgeing which agent in a grid-like field to move colored boxes is doing the correct move.
        You personally do not need to make any moves but only serve as the decision maker to judge others' moves.
        
        The goals and rules of this environment are:
        {GOAL_RULES}

        The first agent is giving command of {cen_response}, but the second agent is sayin {local_response}.
        Here is the current state : {cur_state}.
        Please judge which of the action from the first agent or the second agent is better.
        Do not come-up with something new, only choose one of them, do not give explantion, just choose one of them.

        Include an agent only if it has a task next. If the agent does not have task, do not include.
        Now, select the next step:
        """
    return judge_prompt


def LLM_summarize_func(
    state_action_prompt_next_initial: str,
    model_name: str = "qwen2.5:14b-instruct-q3_K_L",
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


def rplh_prompt_func(
    state_update_prompt: str,
    data: Dict,
    dialogue_history_method: str,
    HCA_agent_location: str,
) -> str:
    """
    Designs an input prompt for a role-playing leader-hallucinating (RPLH) agent
    using in-context learning and chain-of-thought reasoning.

    Args:
        state_update_prompt (str): Description of the current state and available actions.
        data (Dict): Dictionary containing past responses, states, and dialogue history.
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
    ):

        # first iteration no summary
        if dialogue_history_method == "_w_only_state_action_history":
            state_action_prompt = ""
            for i in range(len(response_total_list) - 1, -1, -1):
                state_action_prompt_next = (
                    f"State{i + 1}: {pg_state_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n"
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
                    f"State{i + 1}: {pg_state_list[i]}\nSummary of Dialogues in each step{i + 1}: {dialogue_summary}\nAction{i + 1}: {response_total_list[i]}\n\n"
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
                    f"State{i + 1}: {pg_state_list[i]}\nDialogue{i + 1}: {dialogue_history_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n"
                    + state_action_prompt
                )
                if (
                    token_num_count + len(enc.encode(state_action_prompt_next))
                    < input_prompt_token_limit
                ):
                    state_action_prompt = state_action_prompt_next
                else:
                    break

        HCA_prompt = f"""
            You are a central planner directing agent in a grid-like field to move colored boxes.
            You are agent at grid [{HCA_agent_location}]. You need to make moves and other agents need to make moves as well.
            
            The goals and rules of this environment are:
            {GOAL_RULES}
            
            Your task is to instruct each agent to match all boxes to their color-coded targets.
            After each move, agents provide updates for the next sequence of actions.
            You are the central agent and your job is to coordinate the agents optimally.
            The previous state and action pairs at each step are: {state_action_prompt}

            If the previous state and action pairs at each step is empty, you don't need to deduct any of the reaosning below.

            Please learn from previous steps in the following ways:

                1. Please undrstand the attitude of each agents in this environment,
                including yourself based on what you see from previous conversation that all agents have.
                Please write out some justification and list out the attitute of each agent, including yourself.

                2. Based on this charcteristics of each agent, pelase do two things and added them after each agent's attitude:
                    i. Reason about the reactions each agent would have towards your command.
                    ii. Reason about how they would give actions if they are the central agent.
            
            Use the following format:
            - Attitude of agent...
            - Reaction of agent...
            - Commanding action of agent...
  
            Hence, the current state is {pg_state_list[-1]}, with the possible actions: {state_update_prompt}.

            Think about waht the future {N} actions would be if you want to achieve the goal and write this justification out.
            Remanber to wirte out for each step, what you plan fro every agent to do and what would teh consequences state change be.
            
            Please use thf ollowing format:
            - hallucination of future {N} steps...

            Based on this, generate the action plan for the immediate next step for each agent and specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[0.5, 1.5]":"move(box_blue, target_blue)"}}.
            Remanber to assign action to your self as well.
            Now, plan the next step:
            """
    return HCA_prompt


def dialogue_func(
    state_update_prompt_local_agent: str,
    state_update_prompt_other_agent: str,
    central_response: str,
    data: Dict,
    dialogue_history_method: str,
    local_agent_location: str,
) -> str:
    """
    Constructs a dialogue prompt for a local agent in response to the central planner.

    Args:
        state_update_prompt_local_agent (str): State and actions specific to the local agent.
        state_update_prompt_other_agent (str): State and actions of other agents.
        central_response (str): Central planner's response.
        data (Dict): Data containing historical responses and states.
        dialogue_history_method (str): Method for managing dialogue history.
        local_agent_location (str): Location of the local agent in the grid.

    Returns:
        str: Dialogue prompt for the local agent.
    """

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
        "_w_compressed_dialogue_history",
        "_w_all_dialogue_history",
    ):
        if dialogue_history_method == "_w_only_state_action_history":
            state_action_prompt = ""
            for i in range(len(response_total_list) - 1, -1, -1):
                state_action_prompt_next = (
                    f"State{i + 1}: {pg_state_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n"
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
                    f"State{i + 1}: {pg_state_list[i]}\nSummary of Dialogues in each step{i + 1}: {dialogue_summary}\nAction{i + 1}: {response_total_list[i]}\n\n"
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
                    f"State{i + 1}: {pg_state_list[i]}\nDialogue{i + 1}: {dialogue_history_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n"
                    + state_action_prompt
                )
                if (
                    token_num_count + len(enc.encode(state_action_prompt_next))
                    < input_prompt_token_limit
                ):
                    state_action_prompt = state_action_prompt_next
                else:
                    break

        local_HCA_prompt = f"""
            Imagine that you are a central planner directing agent in a grid-like field to move colored boxes.
            Particularly, you're a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground at grid location of [{local_agent_location}].
            
            The goals and rules of this environment are:
            {GOAL_RULES}
            
            Other central planner is also coordinating all other agents to achieve the goal: match each box with its color-coded target.
            You can only move same color boxes to same color targets.
            
            The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
            The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
            The previous state and action pairs at each step are: {state_action_prompt}
            Please learn from previous steps in a few steps:
                
                1. Please pick up an attitude on this problem for yourself, a "characteristic" that you think you should have based on what you see from previous conversation you have,
                please write the justification for your attitude and state clearly what your attitude is.

                2. You would recieve an plan from the other central planner, please evaluate the given plan and give critical feedbacks.

            You can imagine first about how you would plan these actions and specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])",  Agent[0.5, 1.5]":"move(box_blue, target_blue)"}}.
            Remanber to assign action to your self as well.

            The other central planner's current action plan is giving as: {{{central_response}}}.
            Please be critical in thinking about this plan.

            Please evaluate the given plan.
            If you agree with it, respond 'I Agree', without any extra words.
            If not, briefly explain your objections to this other central planner and an judger agent will get involved.
            Ensure that you still include the actions that you agree with.
            
            Pleas remanber to only change the actions you disagree with and not change other actions, remanber to include all actions for each agents.
            Your response:
        """
    return local_HCA_prompt


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
    The output should be like json format like: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[0.5, 1.5]":"move(box_blue, target_blue)"}}.
    If no action for one agent in the next step, just do not include its action in the output.
    Also remember at most one action for each agent in each step.
    Next step output:
    """
    return user_reprompt


def message_construct_func(
    user_prompt_list: List[str],
    response_total_list: List[str],
    dialogue_history_method: str,
) -> List[Dict[str, str]]:
    """
    Constructs messages for the model with the appropriate dialogue context.
    Create a specialized LLM dictrionary with prompt information, later convert back in LLM class

    (with all dialogue history concats)

    Args:
        user_prompt_list (List[str]): List of user prompts.
        response_total_list (List[str]): List of model responses.
        dialogue_history_method (str): Method for managing dialogue history.

    Returns:
        List[Dict[str, str]]: List of message dictionaries for the model.
    """

    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant. 
                 
                 When asked to specifiy your action plan, specificy it strictly in JSON format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_blue, target_blue])"}}. 
                 
                 Make sure that:
                 - If no action for an agent in the next step, do not include it in JSON output. 
                 - At most one action for each agent in each step.
                 """,
        }
    ]

    if f"{dialogue_history_method}" in (
        "_w_all_dialogue_history",
        "_w_compressed_dialogue_history",
    ):

        # print('length of user_prompt_list', len(user_prompt_list))
        for i in range(len(user_prompt_list)):
            messages.append({"role": "user", "content": user_prompt_list[i]})

            # if i < len(user_prompt_list) - 1:
            #     messages.append(
            #         {"role": "assistant", "content": response_total_list[i]}
            #     )

    elif f"{dialogue_history_method}" in (
        "_wo_any_dialogue_history",
        "_w_only_state_action_history",
    ):
        messages.append({"role": "user", "content": user_prompt_list[-1]})
    return messages


def judge_message_construct_func(user_prompt_list: List[str]) -> List[Dict[str, str]]:
    """
    Constructs a message sequence for a judge agent to evaluate conflicting plans.

    Args:
        user_prompt_list (List[str]): List of user prompts to provide context for the judge.

    Returns:
        List[Dict[str, str]]: A structured sequence of messages for the judge to process.
    """
    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant specialized for judging conflicting plans.
                 
                 When asked to specifiy your action plan, specificy it strictly in JSON format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_blue, target_blue])"}}. 
                 
                 Make sure that:
                 - If no action for an agent in the next step, do not include it in JSON output. 
                 - At most one action for each agent in each step.
                 """,
        }
    ]
    for i in range(len(user_prompt_list)):
        messages.append({"role": "user", "content": user_prompt_list[i]})

    return messages


def json_check_message_construct_func(user_prompt_list: str) -> List[Dict[str, str]]:
    """
    Constructs a message for validating and fixing JSON format in a response.

    Args:
        response (str): The response string to check and fix.

    Returns:
        List[Dict[str, str]]: Message sequence to fix the JSON.

    Notes:
        Must give example or else LLM give {"Agent0_50_5":"move(box_green, target_green)", "Agent1_50_5":"move(box_red, target_red)"}
    """

    EX = f""" Here are three wrong and correct json example pairs that you can learn from:
    
    Wrong format (missing quotation mark in the start):
        {{Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_blue, target_blue])"}}
    Correct format:
        {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_blue, target_blue])"}}
    
    Wrong format (missing bracket in the end)
        {{"Agent[1.5, 1.5]":"move(box_green, target_green])", "Agent[1.5, 1.5]":"move(box_purple, square[0.5, 0.5]"}}
    Correct format:
        {{"Agent[1.5, 1.5]":"move(box_green, target_green])", "Agent[1.5, 1.5]":"move(box_purple, square[0.5, 0.5])"}}
    
    Wrong format (missing multiple quotation marks):
        {{"Agent[0.5, 1.5]":"move(box_red, square[1.5, 1.5])", Agent[0.5, 0.5]":move(box_blue, target_blue])"}}
    Correct format:
        {{"Agent[0.5, 1.5]":"move(box_red, square[1.5, 1.5])", "Agent[0.5, 0.5]":"move(box_blue, target_blue])"}}
        
        """

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant specialized for fixingJson format output by agents in a grid-like environment.",
        },
        {
            "role": "user",
            "content": f"""Please fix the Json message in here {user_prompt_list} and give only this JSON as output.
                            You should not change the content of the message that is  passed in.
                            When asked to give json format, specificy it strictly in JSON format. Here is some example of corerct and wrong json pairs: {EX}.
                            Now the fixed json format message is:""",
        },
    ]
    messages.append({"role": "user", "content": user_prompt_list[-1]})

    return messages
