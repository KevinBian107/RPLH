import sys
from pathlib import Path
import tiktoken

main_path = Path(__file__).resolve().parent.parent.parent
if str(main_path) not in sys.path:
    sys.path.append(str(main_path))

from rplh.llm.language_model import *
from rplh.env.env import *
from rplh.h_efficient.memory.memory_standard import *
from rplh.env.env import better_state_repres


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
    local_responses: list,
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
        local_agent_location (str): Location of the local agent in the grid.
        local_responses (list): Response list of the local agent to the central planner.
        feedback (str): Feedback on the previous action plan.
        judging_mode (bool): Flag to indicate if the agent is in judging mode.
        
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
                
        if (judging_mode == False) and (data["env_step"] == 0):
            agent_reason_prompt = "You are the first agent. Leave spy_model and agent_model empty"
            print('FIRST HCA')
        else:
            # when being HCA, judeg who is spy directly
            print("NEW HCA DOING SOCIAL REASONING")
            agent_reason_prompt = f"""
            You have a agent_model and spy_model from the rpevious HCA agent, please learn from them before constructing your own:
            Previous agent model" {data['agent_model']}
            Previous spt model {data['spy_model']}
            
            Do not fully beleive these models, look at actions from each agent and reason who is the spy.
            
            Please learn from attitude in the following ways:                

                1. Based on the action taken by each agent and the reasoning they give charcteristics of each agent, please do two things and added them after each agent's attitude:
                    i. Reason about the reactions each agent would have towards your command.
                    ii. Reason about how they would give actions if they are the central agent.
                    
                2. Please build your belief on what each agent would do and outpute in agent_model. Use one world to describe their attitude and then some reasoning.
                    Do not just say what the agents are doing, but rather use texts to explain their characteristics.
                    You should build model for all the agent in the format agent_model[{{Agent[0.5, 0.5]: [proactive: good in moving box], Agent[0.5, 1.5]: [This agent is not really cooperative hence should try to avoid moving boxes to him], ...}}].
                
                3. Notice that there may be multiple spy agent in your environment where their objective is different than yours.
                    Write out who you think the spy is and their objectives in the spy_model.
                    Do your reasoning based on agent's action, not just their conversation. Think if their action makes sense.
                    Here are the actions {local_responses}
                    At least half of the agents are spys.
                    
                4.  Notice that when one agent is not in your action plan, they will not be participated in conversation, so it may be smart to not give actions to uncooperative agents. That is, try to make plans to skip the spy agent.
                    Consider this when giving action plan
                   
            """
        if judging_mode:
            # when being judge, find spy incrementally
            re_eval_prompt = f"""{local_agent_location} has provided their perspective on your plan
                            (notice that they will think that they are the central planner) as this feedback {local_responses}.
                            Use this inofrmation to judge if this agent is a spy and then update your agent_model and spy_moddel.
                            Remamber to append ths previous agent_model and spy_model.
                            
                            If you think an loca agent is an spy, don't listen to his opinions.
                            """
        else:
            re_eval_prompt = ""

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
            
            Notice that you should try to utilize all agents by assigning actions to agents that doesn't have boxes in their region for transportations.
            Please only plan actions for each agent that is chosen from each agent's doable action list, do not give a action that is not doable.

            {agent_reason_prompt}
            
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
    
    att_def = load_config("rplh/configs/attitude_config.yaml")
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