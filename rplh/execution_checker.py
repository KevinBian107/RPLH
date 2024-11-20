"""Synthetic checker module/execution module"""

from prompt import *
from env_create import *
import json
import re
import copy
import numpy as np
from typing import Union

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
    except ValueError:
        return False


def json_checker(
    response: str, token_num_count_list_add: list, model_name: str
) -> tuple[str, list]:
    """
    Continuously checks and corrects the JSON format of a response.

    Args:
        response (str): The initial response string to validate and correct.
        token_num_count_list_add (list): A list to store token count data for each attempt.
        model_name (str): The name of the model generating responses.

    Returns:
        tuple[str, list]: A tuple containing the corrected JSON response and updated token count list.
    """
    valid = is_valid_json(response)
    count = 0
    while not valid:
        count += 1
        print(f"----------JSON CHECKER PERFORMING {count} NUMBER OF TIMES----------")
        messages = json_check_message_construct_func(response)
        response, token_num_count = LLaMA_response(messages, model_name)
        
        print(response)

        match = re.search(r"{.*}", response, re.DOTALL)
        if match:
            response = match.group()
            token_num_count_list_add.append(token_num_count)
            valid = is_valid_json(response)

    return response, token_num_count_list_add


def action_checker(response: str, pg_dict_input: list, is_judge: bool) -> str:
    """
    Validates the actions proposed in a response against the playground's state.

    Args:
        response (str): The proposed action plan in JSON format.
        pg_dict_input (list): The current state of the playground as a dictionary.
        is_judge (bool): Whether the check is performed in judge mode.

    Returns:
        str: Feedback about any invalid actions. An empty string if all actions are valid.
    """
    original_response_dict = json.loads(response)
    pg_dict_original = copy.deepcopy(pg_dict_input)
    transformed_dict = {}
    for key, value in original_response_dict.items():
        coordinates = tuple(map(float, re.findall(r"\d+\.?\d*", key)))

        # match the item and location in the value
        match = re.match(r"move\((.*?),\s(.*?)\)", value)
        if match:
            item, location = match.groups()

            if "square" in location:
                location = tuple(map(float, re.findall(r"\d+\.?\d*", location)))

            transformed_dict[coordinates] = [item, location]

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
                feedback += f"You are the judge and your assigned task for {key[0]}_{key[1]} is not in the doable action list, so choose the alternative action of the central planner;"
            else:
                feedback += f"Your assigned task for {key[0]}_{key[1]} is not in the doable action list; "

    return feedback


def retake_action(
    feedback: str,
    user_prompt_list: list[str],
    response_total_list: list[str],
    token_num_count_list_add: list[str],
    dialogue_history_method: str,
    model_name: str,
) -> tuple[str, list]:
    """
    Prompts the model to regenerate actions based on feedback.

    Args:
        feedback (str): Feedback on why the action plan needs correction.
        user_prompt_list (list): The list of user prompts.
        response_total_list (list): List of responses to maintain dialogue context.
        dialogue_history_method (str): Method to manage dialogue history.
        model_name (str): The name of the model generating responses.

    Returns:
        tuple[str, list]: A tuple containing the corrected JSON response and updated token count list.
    """

    print("----------EXECUTION AVAILABILITY CHECKER----------")

    feedback += f"""Please replan for all the agents again with the same ouput format. The output should have the same json format:
    {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_blue, target_blue])"}}.
    Do not explain, just directly output json directory. Remenber to output in json format Your response:"""

    user_prompt_list.append(feedback)
    messages = message_construct_func(
        user_prompt_list, response_total_list, dialogue_history_method
    )

    print(f"Length of messages {len(messages)}")
    response, token_num_count = LLaMA_response(messages, model_name)
    token_num_count_list_add.append(token_num_count)

    return response, token_num_count_list_add


def with_action_syntactic_check_func(
    pg_dict_input: dict[str, list[str]],
    response: str,
    user_prompt_list_input: list[str],
    response_total_list_input: list[str],
    model_name: str,
    dialogue_history_method: str,
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
        is_judge (bool, optional): Flag to indicate if the check is for a judge's response.

    Returns:
        tuple[Union[str, dict], list[int]]: Validated response and token count list.

    Notes:
        This only checks if the actions are valid, doesn't care about if it's json, if not json, directly fails it.
    """
    user_prompt_list = copy.deepcopy(user_prompt_list_input)
    response_total_list = copy.deepcopy(response_total_list_input)
    iteration_num = 0
    token_num_count_list_add = []

    # initial check json
    response, token_num_count_list_add = json_checker(
        response,
        token_num_count_list_add,
        model_name,
    )
    response_total_list.append(response)

    while iteration_num < CHECK_ITER:

        # for action validity check, it must be in json format
        feedback = action_checker(
            response, pg_dict_input, is_judge
        )

        # gate loop: no feedback + no error -> pass

        if feedback != "":
            response, token_num_count_list_add = retake_action(
                feedback,
                user_prompt_list,
                response_total_list,
                token_num_count_list_add,
                dialogue_history_method,
                model_name,
            )
            response_total_list.append(response)

            if response == "Out of tokens":
                return response, token_num_count_list_add
            iteration_num += 1

        elif not is_valid_json(response):
            response, token_num_count_list_add = json_checker(
                response, token_num_count_list_add, model_name
            )

        else:
            # no feedback
            return response, token_num_count_list_add

    return "Syntactic Error", token_num_count_list_add
