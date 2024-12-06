import sys
from pathlib import Path
import re
import argparse

main_path = Path(__file__).resolve().parent.parent.parent
if str(main_path) not in sys.path:
    sys.path.append(str(main_path))

from rplh.llm.language_model import *
import tiktoken
from rplh.h_efficient.memory.memory_standard import *

enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
enc = tiktoken.encoding_for_model("gpt-4")
input_prompt_token_limit = 3000
N = 5


def rplh_prompt_agent_func(
    state_update_prompt: str,
    data: dict,
    dialogue_history_method: str,
    HCA_agent_location: str,
    local_agent_location: str,
    agent_model: dict[str, list[str]],
    actual_model: dict[str, list[str]],
    strategy_model: dict[str, list[str]],
    local_response: str,
    cen_response: str,
    feedback: str = "",
    judging_mode: bool = False,
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
        feedback (str): Feedback on the previous action plan.

    Returns:
        str: A structured prompt for the role-playing leader-hallucinating agent.

    Notes:
        Boxes just need to be moved to the target location, not in the target location.
    """

    if data["env_step"] == 0:
        attitude = None
        success_action = f"""No previous action, here is an sample where box_x and box_y are arbitrary boxes:
        {{"Agent[0.5, 0.5]":"move(box_x, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_y, target_y])"}}"""
    else:
        attitude = data["attitude_info"][-1]
        success_action = data["response_total_list"][-1]

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

        if attitude == None:
            print("ATTITUDE IS NONE")
            att_promt = "You are the first agent, please leave the agent_model, actual_model, and strategy_model response field as empty."
        else:
            att_promt = f"""
            You have a agent_model and actual_model from the rpevious HCA agent, please learn from them before constructing your own:
            Previous agent model" {agent_model}
            Previous actual model {actual_model}
            Previous strategy model {strategy_model}
            
            Please learn from attitude in the following ways:

                1. Please undrstand the attitude of each agents in this environment,
                including yourself based on this attitude report given from another agent: 
                
                {attitude}.

                2. Based on this charcteristics of each agent, please do two things and added them after each agent's attitude:
                    i. Reason about the reactions each agent would have towards your command.
                    ii. Reason about how they would give actions if they are the central agent.
                    
                3. Please build your belief on what each agent would do and outpute in agent_model. You should build model for each agent in the format agent_model[{{Agent[0.5, 0.5]: [...], Agent[0.5, 1.5]: [...], Agent[1.5, 0.5]: [...], Agent[1.5, 1.5]: [...]}}].
                    You will recieve information about what each agent actually do and think later, so pleae leave actual_models blank for now.
                    Do not just say what the agents are doing, but rather use texts to explain their characteristics and behaviors.
                
                4. Based on the strategy model, please be very careful in giving action plan to each agent, make plans that makes it more likely for each local agent to obey and agree directly without argument.
            """
        if judging_mode:
            re_eval_prompt = f"""{local_agent_location} has provided their perspective on your plan as this feedback {local_response}, with this information, do two things:
            
                            1. Please modify your original plan of {cen_response}.
                            
                            2. Please summarize what this particular agent's perspective is and put it in actual_model, actual_model is each agent's characteristcs and behaviors, not actions.
                            
                            3. Modify the agent_model based on your summmary in actual_model.
                            """
        else:
            re_eval_prompt = ""

        if feedback != "":
            feedback = (
                "There is error in preivous action plan. Here is the feedbcak: "
                + feedback
            )

        # escaped_agent = re.escape(HCA_agent_location)
        # pattern = fr"{escaped_agent}:.*?(?=Agent\[|$)"
        # match = re.search(pattern, state_update_prompt, re.DOTALL)
        # agent_data = match.group(0) if match else None
        # print(f'AGENT CAN SEE AND CAN DO: {agent_data}')
        # print(f'HISTORY IS {state_action_prompt}')

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

            {att_promt}
            
            {re_eval_prompt}

            Think about what the future {N} actions would be if you want to achieve the goal with the reasoning.
            Remanber to wirte out for each step, what you plan for every agent to do and what would the state change be.
            
            {feedback}

            Action Plan:
            Specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_x, square[0.5, 1.5])","Agent[0.5, 1.5]": "move(box_y, target_y)"}} where box_x and box_y are arbitrary boxes.
            Try to propose actions for all four agents.
            One agent can only make one action.
            No agent name should be given if the agent does not have a task next. 
            """
    return HCA_prompt


def dialogue_agent_func(
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
        success_action = f"""No previous action, here is an sample where box_x and box_y are arbitrary boxes:
        {{"Agent[0.5, 0.5]":"move(box_x, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_y, target_y])"}}"""
    else:
        attitude = data["attitude_info"][-1]
        success_action = data["response_total_list"][-1]

    if attitude == None:
        print("ATTITUDE IS NONE")
        att_promt = "Be very critical"
    else:
        att_promt = f"""
            Please pick up an attitude on this problem for yourself based on the attitude that this attitude report assigned to you: {attitude}.
            your response should reflect such attitude.
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
                
                3. Give a self-evaluation of your attitude and your characteristcs.

            You can imagine first about how you would plan these actions and specify your action plan.
            This is the success response of previous state: {success_action}
            Remanber to assign action to your self as well.

            The other central planner's current action plan is giving as: {central_response}.
            Try to find agreement with the central ageent if you can, the goal is to resolve conversation.
            
            Prioritize adding more actions or keeping at least the same number of action if possible, but the number of action should not be more than the number of agents.

            Please evaluate the given plan.
            If you agree with it, respond 'I Agree', without any extra words.
            If not, briefly explain your objections to this other central planner and an judger agent will get involved.
            Ensure that you still include the actions that you agree with.
            
            Please remanber to only change the actions you disagree with and not change other actions, remanber to include all actions for each agents.
            
            {feedback}
            
            Remaanber to state who you are first before giving responses.
            Your response:
        """
    return local_HCA_prompt


def attitude_agent_prompt_func_for_agent(history: dict, prev_attitude: str) -> str:
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
        Try to make this new attitude align with previous attitude that you give. The previous attitudes are {prev_attitude}.
        
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
