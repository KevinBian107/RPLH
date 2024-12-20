"""Synthetic checker module/execution module"""

from rplh.systems.h_efficient.memory.memory_standard import *
from rplh.env.env import *
from rplh.llm.language_model import *
from rplh.llm.response_model import *
import json
import re
import copy
import numpy as np
from typing import Union, Callable

CHECK_ITER = 10


def is_valid_action(
    response: str, 
    central_response: str, 
    pg_dict_input: list, 
    state_update_prompt: str, 
    is_judge: bool, 
    agent_action: dict
) -> str:
    """
    Validates the actions proposed in a response against the playground's state.

    Args:
        response (str): The proposed action plan in JSON format.
        central_response (str): central response
        pg_dict_input (list): The current state of the playground as a dictionary.
        is_judge (bool): Whether the check is performed in judge mode.

    Returns:
        str: Feedback about any invalid actions. An empty string if all actions are valid.
    """

    print("----------EXECUTION AVAILABILITY CHECKER----------")
    original_response_dict = response
    pg_dict_original = copy.deepcopy(pg_dict_input)
    transformed_dict = {}

    for key, value in original_response_dict.items():
        coordinates = tuple(map(float, re.findall(r"\d+\.?\d*", key)))

        # match the item and location in the value
        try:
            match = re.match(r"move\((.*?),\s(.*?)\)", value)
            if match:
                item, location = match.groups()

                if "square" in location:
                    location = tuple(map(float, re.findall(r"\d+\.?\d*", location)))

                transformed_dict[coordinates] = [item, location]
        except:
            print(f"NO MATCHING: SYNTAX ERROR, NEED TO RETAKE ACTION")
            feedback = "Agent name should be in the form like Agent[0.5, 0.5], you give the wrong format."

            return feedback

    feedback = ""
    for key, value in transformed_dict.items():
        if (
            value[0] in pg_dict_original[str(key[0]) + "_" + str(key[1])]
            and type(value[1]) == tuple
            and (
                (
                    np.abs(key[0] - value[1][0]) == 0
                    and np.abs(key[1] - value[1][1]) == 1
                )
                or (
                    np.abs(key[0] - value[1][0]) == 1
                    and np.abs(key[1] - value[1][1]) == 0
                )
            )
        ):
            pass
        elif (
            value[0] in pg_dict_original[str(key[0]) + "_" + str(key[1])]
            and type(value[1]) == str
            and value[1] in pg_dict_original[str(key[0]) + "_" + str(key[1])]
            and value[0][:4] == "box_"
            and value[1][:7] == "target_"
            and value[0][4:] == value[1][7:]
        ):
            pass
        else:

            agent_name = f'Agent[{key[0]}, {key[1]}]'
            if agent_name in agent_action:
                doable_action_lst = agent_action[agent_name]
                doable_action = f'The list of doable action for {agent_name} is {doable_action_lst}'
            else:
                print(f'ERROR: NOT DOABLE ACTION FOR {agent_name}')
                doable_action = 'No doable action for this agent'

            if is_judge:
                feedback = f"""
        You are the judge and your assigned task for {key[0]}, {key[1]} is not in the doable action list,
        {doable_action}, please don't make the same action plan for Agent {key[0]}, {key[1]}. So choose the alternative action from the central central planner {central_response}.
                """
            else:
                feedback = f"""
        Your assigned task for {key[0]}, {key[1]} is not in the doable action list, 
        {doable_action}.
                """
    return feedback


def retake_action(
    feedback: str,
    user_prompt_list: list[str],
    response_total_list: list[str],
    token_num_count_list_add: list[str],
    dialogue_history_method: str,
    model_name: str,
    prompt_func: Callable[[str, dict, str, str], str],
    is_judge: bool = False,
) -> tuple[str, list]:
    """
    Prompts the model to regenerate actions based on feedback.

    Args:
        feedback (str): Feedback on why the action plan needs correction.
        user_prompt_list (list): The list of user prompts.
        response_total_list (list): List of responses to maintain dialogue context.
        dialogue_history_method (str): Method to manage dialogue history.
        model_name (str): The name of the model generating responses.
        prompt_func (Callable): specific partial function passed in based on local/hca/judge agent

    Returns:
        tuple[str, list]: A tuple containing the corrected JSON response and updated token count list.

    Note:
        Sending back to HCA agent for retaking action, keep things simple and direct.
    """

    print("----------RETAKE ACTION----------")
    retake_action_prompt_1 = prompt_func(feedback=feedback)
    messages = message_construct_func(
        [retake_action_prompt_1], [], dialogue_history_method
    )

    print(f"Length of messages {len(messages)}")

    # choose response_model depends on
    response_model = None
    if is_judge:
        response_model = Judge
    else:
        response_model = HCA

    raw_response, token_num_count = LLaMA_response_json(
        messages, model_name, response_model
    )
    response = json.loads(raw_response)
    response = response["actions_plan"]
    token_num_count_list_add.append(token_num_count)

    return response, token_num_count_list_add


def with_action_syntactic_check_func(
    pg_dict_input: dict[str, list[str]],
    response: dict,
    user_prompt_list_input: list[str],
    response_total_list_input: list[str],
    model_name: str,
    dialogue_history_method: str,
    prompt_func: Callable[[str, dict, str, str], str],
    state_update_prompt: str,
    agent_action: dict,
    is_judge: bool = False,
) -> tuple[Union[str, dict], list[int]]:
    """
    Checks and validates the syntactic correctness of the action plan.

    Args:
        pg_dict_input (dict[str, list[str]]): Current state of the playground.
        response (dict): Proposed action plan in JSON format.
        user_prompt_list_input (list[str]): List of user prompts.
        response_total_list_input (list[str]): List of previous responses.
        model_name (str): Name of the model generating the response.
        dialogue_history_method (str): Method for managing dialogue history.
        prompt_func (Callable): specific partial function passed in based on local/hca/judge agent
        is_judge (bool, optional): Flag to indicate if the check is for a judge's response.

    Returns:
        tuple[Union[str, dict], list[int]]: Validated response and token count list.

    Notes:
        This only checks if the actions are valid, doesn't care about if it's json, if not json, directly fails it.
    """
    if not is_judge:
        user_prompt_list = copy.deepcopy(
            user_prompt_list_input
        )  # only one passed in if not judge
        central_response = ""  # wouldn't be used if not judge
    else:
        user_prompt_list = copy.deepcopy(
            [user_prompt_list_input[0]]
        )  # 0 is always judge
        central_response = copy.deepcopy(
            user_prompt_list_input[1]
        )  # 1 is always central

    response_total_list = copy.deepcopy(response_total_list_input)
    iteration_num = 0
    token_num_count_list_add = []

    # logic gate: put on DSC20 final exam please
    while iteration_num < CHECK_ITER:
        # preventing JSON checker change action
        feedback = is_valid_action(
            response, central_response, pg_dict_input, state_update_prompt, is_judge, agent_action
        )
        
        print(f'ORIGINAL PROPOSED ACTION IS {response}')
        print(f"FEEDBACK IS {feedback}")

        if feedback != "":  # this is fine
            print("ENTERING RETAKE ACTION")
            response, token_num_count_list_add = retake_action(
                feedback,
                user_prompt_list,
                response_total_list,
                token_num_count_list_add,
                dialogue_history_method,
                model_name,
                prompt_func, 
                is_judge,
            )
            print(f"ACTION RETAKEN: {response}")
            response_total_list.append(response)
            if response == "Out of tokens":
                return response, token_num_count_list_add
        else:
            #  no feedback
            return response, token_num_count_list_add

        iteration_num += 1

    return "Syntactic Error", token_num_count_list_add

# TODO: might have issue with processing raw response, need change
def process_response(response: dict) -> dict:
    """
    Processes a response dictonary that is not suitable for execution, extracts relevant information,
    and converts it into a format suitable for environment execution.

    Args:
        response (dict): A not suitable dictonary can be in format like:
            {"Agent[0.5_1.5]": "move(box_red, square[1.5_1.5])"}
            {'Step_1': {'Agent[0.5, 0.5]': 'move(box_red, target_red)'}
            {'Agent[0.5, 0.5]': {'action': 'move(box_red, target_red)', 'reasoning': ''},

    Returns:
        dict: A dictionary where keys are agent (e.g., "Agent[0.5, 0.5]")
              and values are actions (e.g., "move(box_blue, square[0.5, 1.5])").
    Note:
        Do nothing and return original response back if error in parsing the response.
    """

    def find_idx(substring, dic):
        for key in dic.keys():
            if substring in key.lower():
                return key
        return False

    try:
        transformed_dict = dict()

        for key, value in response.items():
            if "agent" in key.lower():
                coord = tuple(map(float, re.findall(r"\d+\.?\d*", key)))
                value_action = value
            if isinstance(value, dict):
                agent = find_idx("agent", value)
                action = find_idx("action", value)
                if agent:
                    coord = tuple(map(float, re.findall(r"\d+\.?\d*", agent)))
                    value_action = value[agent]
                elif action:
                    value_action = value[action]

            # match the item and location in the value
            match = re.match(r"move\((.*?),\s(.*?)\)", value_action)
            if match:
                item, location = match.groups()
                if "square" in location:
                    location = tuple(map(float, re.findall(r"\d+\.?\d*", location)))
                    location = f"square[{location[0]}, {location[1]}]"

                transformed_dict[f"Agent[{coord[0]}, {coord[1]}]"] = (
                    f"move({item}, {location})"
                )
        return transformed_dict

    except:
        return response
