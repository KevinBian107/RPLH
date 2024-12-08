"""Synthetic checker module/execution module"""

from rplh.systems.h_vanilla.memory import *
from rplh.env.env import *
from rplh.llm.language_model import *
import json
import re
import copy
import numpy as np
from typing import Union, Callable

CHECK_ITER = 10


def is_valid_json(response: str) -> bool:
    """
    Checks if a response string is in valid JSON format.

    Args:
        response (str): The response string to validate.

    Returns:
        bool: True if the response is valid JSON, False otherwise.
    """
    try:
        json.loads(response)
        return True
    except:  # all error return false
        return False


def is_valid_action(
    response: str, central_response: str, pg_dict_input: list, is_judge: bool
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

    # need to ensure this must be json
    original_response_dict = json.loads(response)
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
            if is_judge:
                feedback = f"""You are the judge and your assigned task for {key[0]}, {key[1]} is not in the doable action list,
                                so choose the alternative action from the central central planner {central_response};"""
            else:
                feedback = f"Your assigned task for {key[0]}, {key[1]} is not in the doable action list; "

            return feedback
    print(f"Transformed_dict: {transformed_dict}")
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
            if is_judge:
                feedback += f"""You are the judge and your assigned task for {key[0]}, {key[1]} is not in the doable action list, 
so choose the alternative action from the central central planner {central_response};"""
            else:
                feedback += f"Your assigned task for {key[0]}, {key[1]} is not in the doable action list; "

    return feedback


def reformat_json(
    response: str,
    token_num_count_list_add: list,
    model_name: str,
    prompt_func: Callable[[str, dict, str, str], str],
    dialogue_history_method: str,
) -> tuple[str, list]:
    """
    Continuously checks and corrects the JSON format of a response.
    Args:
        response (str): The initial response string to validate and correct.
        token_num_count_list_add (list): A list to store token count data for each attempt.
        model_name (str): The name of the model generating responses.
        prompt_func (Callable): specific partial function passed in based on local/hca/judge agent
        dialogue_history_method (str): Method for managing dialogue history.
    Returns:
        tuple[str, list]: A tuple containing the corrected JSON response and updated token count list.
    """
    valid = is_valid_json(response)
    count = 0
    feedback = "This json format is incorrect, please check"
    while not valid:
        count += 1
        print(f"----------JSON CHECKER PERFORMING {count} NUMBER OF TIMES----------")
        check_prompt_1 = prompt_func(feedback)
        messages = message_construct_func([check_prompt_1], [], dialogue_history_method)
        raw_response, token_num_count = GPT_response(messages, model_name)

        match = re.search(r"\{.*?\}", raw_response, re.DOTALL)
        if match:
            possible_action_lst = re.findall(r"\{.*?\}", response, re.DOTALL)
            response = possible_action_lst[-1]
            response = process_response(response)

            token_num_count_list_add.append(token_num_count)
            valid = is_valid_json(response)

    return response, token_num_count_list_add


def retake_action(
    feedback: str,
    user_prompt_list: list[str],
    response_total_list: list[str],
    token_num_count_list_add: list[str],
    dialogue_history_method: str,
    model_name: str,
    prompt_func: Callable[[str, dict, str, str], str],
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

    # must specify this
    retake_action_prompt_1 = prompt_func(feedback=feedback)
    messages = message_construct_func(
        [retake_action_prompt_1], [], dialogue_history_method
    )

    print(f"Length of messages {len(messages)}")

    raw_response, token_num_count = GPT_response(messages, model_name)

    match = re.search(r"\{.*?\}", raw_response, re.DOTALL)
    if match:
        possible_action_lst = re.findall(r"\{.*?\}", raw_response, re.DOTALL)
        response = possible_action_lst[-1]
        response = process_response(response)
        token_num_count_list_add.append(token_num_count)
    else:
        print(f"ERROR: NO CURLY BRACKET FOUND IN RETAKE ACTION STAGE: {response}")

    return response, token_num_count_list_add


def with_action_syntactic_check_func(
    pg_dict_input: dict[str, list[str]],
    response: str,
    user_prompt_list_input: list[str],
    response_total_list_input: list[str],
    model_name: str,
    dialogue_history_method: str,
    prompt_func: Callable[[str, dict, str, str], str],
    is_judge: bool = False,
) -> tuple[Union[str, dict], list[int]]:
    """
    Checks and validates the syntactic correctness of the action plan.

    Args:
        pg_dict_input (dict[str, list[str]]): Current state of the playground.
        response (str): Proposed action plan in JSON format.
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
    feedback = "INITIAL_CHECK"

    # logic gate: put on DSC20 final exam please
    while iteration_num < CHECK_ITER:

        # gate loop: no feedback + no error -> pass (always check Json + action (round 0), later Json + feedback)
        if (not is_valid_json(response)) or (feedback != ""):
            # print(f"IN GATE CHECKING: {response}")

            if not is_valid_json(response):
                # print("BEFORE JSON: ", response)
                response, token_num_count_list_add = reformat_json(
                    response,
                    token_num_count_list_add,
                    model_name,
                    prompt_func,
                    dialogue_history_method,
                )
                # print("AFTER JSON:", response)

                # preventing JSON checker change action
                feedback = is_valid_action(
                    response, central_response, pg_dict_input, is_judge
                )

            # if no json error, then check action
            elif feedback == "INITIAL_CHECK":
                feedback = is_valid_action(
                    response, central_response, pg_dict_input, is_judge
                )

            if feedback != "":
                print("RETAKE ACTION")
                response, token_num_count_list_add = retake_action(
                    feedback,
                    user_prompt_list,
                    response_total_list,
                    token_num_count_list_add,
                    dialogue_history_method,
                    model_name,
                    prompt_func,
                )
                print(f"ACTION RETAKEN: {response}")
                print(f"FEEDBACK: {feedback}")
                response_total_list.append(response)

                if response == "Out of tokens":
                    return response, token_num_count_list_add

            # for action validity check, it must be in json format
            feedback = is_valid_action(
                response, central_response, pg_dict_input, is_judge
            )

        else:
            # no feedback
            return response, token_num_count_list_add

        iteration_num += 1

    return "Syntactic Error", token_num_count_list_add


def process_response(response: str) -> dict:
    """
    Processes a raw response string containing agent locations and actions,
    extracts relevant information, and converts it into dictionary format for loading in json.

    Args:
        raw_response (str): The string is expected to include an "EXECUTE" keyword followed by
            JSON-like content describing agent locations and actions
            (e.g "EXECUTE
            {"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])")

    Returns:
        dict: A dictionary where keys are agent identifiers (e.g., "Agent[0.5, 0.5]")
              and values are actions (e.g., "move(box_blue, square[0.5, 1.5])").
    Note:
        Return input response back if error in parsing the response.
    """

    try:
        pattern_1 = r"agent\[(\d+\.\d+), (\d+\.\d+)\]"
        agent_loc = re.findall(pattern_1, response, re.IGNORECASE)

        pattern_2 = r"move\((.*?),\s(.*?)\)"  # r'move(\(\w+, (?:square.\d+\.\d+, \d+\.\d+.|\w+)\))'
        agent_action = re.findall(pattern_2, response, re.IGNORECASE)

    except:
        return response
        # raise ValueError(f'Error in parsing the response: {response}')

    num_agent = len(agent_loc)
    action = []
    if len(agent_loc) == len(agent_action):
        for i in range(num_agent):
            action.append(
                f'"Agent[{agent_loc[i][0]}, {agent_loc[i][1]}]": "move({agent_action[i][0]}, {agent_action[i][1]})"'
            )

        json_str = "{" + ", ".join(action) + "}"
    else:
        print(f"Agent-action pair is inconsistent: {response}")
        return response
        # raise ValueError(f'Agent-action pair is inconsistent: {response}')

    return json_str
