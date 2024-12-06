from rplh.llm.language_model import *
from rplh.env.env import better_state_repres
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
enc = tiktoken.encoding_for_model("gpt-4")
input_prompt_token_limit = 3000
N = 5

# critical, prompt is a hyperparameter
GOAL_RULES = f"""
            You are an agentin a grid-like field to move colored boxes.
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
            Specify your action plan in this format where box_x and box_y are arbitrary boxes: {{"Agent[0.5, 0.5]":"move(box_x, square[0.5, 1.5])","Agent[0.5, 1.5]": "move(box_y, target_y)"}}.
            One agent can only make one action. Include an agent only if it has a task next. 
            No agent name should be given if the agent does not have a task next.
            """

FEEDBACK_LCOAL1 = """
            This is the feedback from local agents.
            If you find some errors in your previous plan, try to modify it.
            Otherwise, output the same plan as before.
            The output should have the same format. 
            Do not explain, just directly output the action plan.
            Your response:
            """

def better_state_repres(pg_dict: dict) -> dict:
    """
    Transforms the state representation the original pg_dict envirionemnt

    Args:
        pg_dict (dict): A dictionary representing BoxNet envirionment.

    Returns:
        dict: A new dictionary with transformed state representation.

    Example:
        Input: {'0.5_0.5': ['box_blue'], 
                '0.5_1.5': ['box_red'], 
                '1.5_0.5': ['target_blue'], 
                '1.5_1.5': ['target_red']},
        Output: {'0.5, 0.5': ['box_blue'], 
                 '0.5, 1.5': ['box_red'], 
                 '1.5, 0.5': ['target_blue'], 
                 '1.5, 1.5': ['target_red']},
    """
    new_pg_dict = {}

    for key, value in pg_dict.items():
        new_pg_dict[f'{key[:3]}, {key[-3:]}'] = value

    return new_pg_dict

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

    if data["env_step"] == 0:
        attitude = None
        # success_action = f"""No previous action, here is an sample:
        # {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_blue, target_blue])"}}"""
    else:
        attitude = data["attitude_info"][-1]
        # success_action = data["response_total_list"][-1]

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
        "_w_no_history",
    ):

        # first iteration no summary
        if dialogue_history_method == "_w_only_state_action_history":
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
        elif dialogue_history_method == "_w_no_history":
            state_action_prompt = ""
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

        if attitude == None:
            print("ATTITUDE IS NONE")
            att_promt = ""
        else:
            att_promt = f"""
            Please learn from attitude in the following ways:

                1. Please undrstand the attitude of each agents in this environment,
                including yourself based on this attitude report given from another agent: 
                
                {attitude}.

                2. Based on this charcteristics of each agent, pelase do two things and added them after each agent's attitude:
                    i. Reason about the reactions each agent would have towards your command.
                    ii. Reason about how they would give actions if they are the central agent.
            
            Use the following format:
            - Attitude of agent...
            - Reaction of agent...
            """

        if feedback != "":
            feedback = (
                "There is error in preivous action plan. Here is the feedbcak: "
                + feedback
            )


        HCA_prompt = f"""
            You are a central planner directing agent in a grid-like field to move colored boxes.
            You are agent [{HCA_agent_location}]. You need to make moves and other agents need to make moves as well.
            
            The goals and rules of this environment are:
            {GOAL_RULES}
            
            Your task is to instruct each agent to match all boxes to their color-coded targets.
            After each move, agents provide updates for the next sequence of actions.
            You are the central agent and your job is to coordinate the agents optimally.
  
            Hence, the current state is {better_state_repres(pg_state_list[-1])}, with the possible actions: {state_update_prompt}.

            Please only plan actions for each agent that is chosen from each agent's doable action list, do not give a action that is not doable.
            
            {att_promt}

            Think about what the future {N} actions would be if you want to achieve the goal and write this justification out.
            Remanber to wirte out for each step, what you plan for every agent to do and what would the consequences state change be.
            
            Please use the following format:
            - hallucination of future {N} steps...

            Based on this, generate the action plan for the immediate next step for each agent.
            Remanber to assign action to your self as well.
            
            {feedback}
            
            Now, plan the next step:
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

    if data["env_step"] == 0:
        attitude = None
        success_action = f"""No previous action, here is an sample:
        {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_blue, target_blue])"}}"""
    else:
        attitude = data["attitude_info"][-1]
        success_action = data["response_total_list"][-1]

    if attitude == None:
        print("ATTITUDE IS NONE")
        att_promt = "Be very critical"
    else:
        att_promt = f"""
            Please pick up an attitude on this problem for yourself based on the attitude that this attitude report assigned to you: {attitude}.
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
        "_w_no_history",
    ):
        if dialogue_history_method == "_w_only_state_action_history":
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
        elif dialogue_history_method == "_w_no_history":
            state_action_prompt = ""
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

        local_HCA_prompt = f"""
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
            Please learn from previous steps in a few steps:
                
                1. {att_promt}

                2. You would recieve an plan from the other central planner, please evaluate the given plan and give critical feedbacks.

            You can imagine first about how you would plan these actions and specify your action plan.
            This is the success response of previous state: {success_action}
            Remanber to assign action to your self as well.

            The other central planner's current action plan is giving as: {central_response}.
            Try to find agreement with the central ageent if you can, the goal is to resolve conversation and modify inappropriate action plan.
            Prioritize adding more actions or keeping at least the same number of action if possible, but the number of action should not be more than the number of agents.

            Please evaluate the given plan.
            If you agree with it, respond 'I Agree', without any extra words.
            If not, briefly explain your objections to this other central planner and an judger agent will get involved.
            Ensure that you still include the actions that you agree with.
            
            Please remanber to only change the actions you disagree with and not change other actions, remanber to include all actions for each agents.
            
            {feedback}
            
            Your response:
        """
    return local_HCA_prompt


def judge_prompt_func(
        local_response: str, 
        cen_response: str, 
        cur_state: dict, 
        feedback: str = ""
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


def attitude_agent_prompt_func(history: dict) -> str:
    """
    Generates a prompt to analyze and derive the attitudes of agents based on their dialogue history.
    Usage for condensed memory

    Args:
        history (str): A string representing the dialogue history of the agents.

    Returns:
        str: The attitudes are expected in the format:
             {Agent[0.5, 0.5]: attitude, Agent[0.5, 1.5]: attitude}.
    """
    attitude_prompt = f"""
        The goals and rules of this environment are:
        {GOAL_RULES}

        Given the dialogue history of each agent {history}. 

        Please derive the attitude of each agents given their response.  
        Please list out the attitute of each agent in the folloing format:
        {{Agent[0.5, 0.5]: attitude, Agent[0.5, 1.5]: attitude}}
        
        Example: 
        {{Agent[0.5, 0.5]: "A Good Decision Maker", 
          Agent[0.5, 1.5]: "Too Aggressive",
          Agent[1.5, 0.5]: "Serious",
          Agent[1.5, 1.5]: "Smart Agent"}}
        
        State your justification after listing out attitudes
        Justification: ...
        """
    return attitude_prompt


def LLM_summarize_func(
    state_action_prompt_next_initial: str,
    model_name: str,
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
    response = GPT_response(messages, model_name)
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
    The output should be like json format like: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[0.5, 1.5]":"move(box_blue, target_blue)"}}.
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
                 
                 When asked to specifiy your action plan, specificy it in JSON format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_blue, target_blue])"}}. 
                 
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
        for i in range(len(user_prompt_list)):
            messages.append({"role": "user", "content": user_prompt_list[i]})
    else:
        print('LESS PROMPT IN MESSAGE CONSTRUCT')
        messages.append({"role": "user", "content": user_prompt_list[-1]})
        
    for i in range(len(response_total_list)):
        messages.append({"role": "assistant", "content": response_total_list[i]})

    return messages


def attitude_message_construct_func(user_prompt: str) -> list[dict[str, str]]:
    """
    Constructs a message sequence for a attitude agent to evaluate conflicting plans.

    Args:
        user_prompt_list (str): User prompts to provide context for the judge.

    Returns:
        list[dict[str, str]]: A structured sequence of messages for the judge to process.
    """
    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant specialized in deriving attitude from dialogue""",
        },
    ]
    messages.append({"role": "user", "content": user_prompt})
    return messages


# def json_check_message_construct_func(user_prompt_list: str) -> List[Dict[str, str]]:
#     """
#     Constructs a message for validating and fixing JSON format in a response.

#     Args:
#         response (str): The response string to check and fix.

#     Returns:
#         List[Dict[str, str]]: Message sequence to fix the JSON.

#     Notes:
#         Must give example or else LLM give {"Agent0_50_5":"move(box_green, target_green)", "Agent1_50_5":"move(box_red, target_red)"}
#     """

#     EX = f""" Here are three wrong and correct json example pairs that you can learn from:

#     Wrong format (missing quotation mark in the start):
#         {{Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_blue, target_blue])"}}
#     Correct format:
#         {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_blue, target_blue])"}}

#     Wrong format (missing bracket in the end)
#         {{"Agent[1.5, 1.5]":"move(box_green, target_green])", "Agent[1.5, 1.5]":"move(box_purple, square[0.5, 0.5]"}}
#     Correct format:
#         {{"Agent[1.5, 1.5]":"move(box_green, target_green])", "Agent[1.5, 1.5]":"move(box_purple, square[0.5, 0.5])"}}

#     Wrong format (missing multiple quotation marks):
#         {{"Agent[0.5, 1.5]":"move(box_red, square[1.5, 1.5])", Agent[0.5, 0.5]":move(box_blue, target_blue])"}}
#     Correct format:
#         {{"Agent[0.5, 1.5]":"move(box_red, square[1.5, 1.5])", "Agent[0.5, 0.5]":"move(box_blue, target_blue])"}}

#         """

#     messages = [
#         {
#             "role": "system",
#             "content": "You are a helpful assistant specialized for fixingJson format output by agents in a grid-like environment.",
#         },
#         {
#             "role": "user",
#             "content": f"""Please fix the Json message in here {user_prompt_list} and give only this JSON as output.
#                             You should not change the content of the message that is  passed in.
#                             When asked to give json format, specificy it strictly in JSON format. Here is some example of corerct and wrong json pairs: {EX}.
#                             Now the fixed json format message is:""",
#         },
#     ]
#     messages.append({"role": "user", "content": user_prompt_list})

#     return messages
