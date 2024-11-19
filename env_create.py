# Box moving to target without collision

"""Working on changing environment to self charcterized env, also working on visualization"""

from prompt import *
from env_create import *
import os
import json
import re
import copy
import numpy as np
import shutil
import time
from typing import Dict, List, Tuple, Union


def surround_index_func(
    row_num: int, coloum_num: int, row_index: int, coloum_index: int
) -> List[List[float]]:
    """
    Calculates the indices of surrounding cells for a given cell in a grid.

    Args:
        row_num (int): Total number of rows.
        col_num (int): Total number of columns.
        row_index (int): Row index of the target cell.
        col_index (int): Column index of the target cell.

    Returns:
        List[List[float]]: List of coordinates for surrounding cells.
    """
    surround_index_list = []
    for i, j in (
        [row_index - 1, coloum_index],
        [row_index + 1, coloum_index],
        [row_index, coloum_index - 1],
        [row_index, coloum_index + 1],
    ):
        if (
            i >= 0
            and i <= row_num - 1
            and j >= 0
            and j <= coloum_num - 1
            and not (i == row_index and j == coloum_index)
        ):
            surround_index_list.append([i + 0.5, j + 0.5])
    return surround_index_list


def state_update_func(
    pg_row_num: int, pg_column_num: int, pg_dict: Dict[str, List[str]]
) -> str:
    """
    Describes the environment and possible actions for the central agent.

    Args:
        pg_row_num (int): Number of rows in the playground.
        pg_column_num (int): Number of columns in the playground.
        pg_dict (Dict[str, List[str]]): State of the playground.

    Returns:
        str: State update prompt for the central agent.
    """

    pg_dict_copy = copy.deepcopy(pg_dict)
    state_update_prompt = ""
    for i in range(pg_row_num):
        for j in range(pg_column_num):
            square_item_list = pg_dict_copy[str(i + 0.5) + "_" + str(j + 0.5)]
            square_item_only_box = [
                item for item in square_item_list if item[:3] == "box"
            ]
            surround_index_list = surround_index_func(pg_row_num, pg_column_num, i, j)
            state_update_prompt += f"Agent[{i+0.5}, {j+0.5}]: I am in square[{i+0.5}, {j+0.5}], I can observe {square_item_list}, I can do "
            action_list = []
            for box in square_item_only_box:
                for surround_index in surround_index_list:
                    action_list.append(f"move({box}, square{surround_index})")
                if "target" + box[3:] in square_item_list:
                    action_list.append(f"move({box}, target{box[3:]})")
            state_update_prompt += f"{action_list}\n"
    return state_update_prompt


def state_update_func_local_agent(
    pg_row_num: int,
    pg_column_num: int,
    pg_row_i: int,
    pg_column_j: int,
    pg_dict: Dict[str, List[str]],
) -> Tuple[str, str]:
    """
    Describes the environment and possible actions for a specific local agent.

    Args:
        pg_row_num (int): Number of rows in the playground.
        pg_column_num (int): Number of columns in the playground.
        pg_row_i (int): Row index of the local agent.
        pg_column_j (int): Column index of the local agent.
        pg_dict (Dict[str, List[str]]): State of the playground.

    Returns:
        Tuple[str, str]: Prompts describing the environment and actions for the local agent
        and for other agents.
    """

    pg_dict_copy = copy.deepcopy(pg_dict)
    state_update_prompt_local_agent = ""
    state_update_prompt_other_agent = ""

    for i in range(pg_row_num):
        for j in range(pg_column_num):
            if not (i == pg_row_i and pg_column_j == j):
                square_item_list = pg_dict_copy[str(i + 0.5) + "_" + str(j + 0.5)]
                square_item_only_box = [
                    item for item in square_item_list if item[:3] == "box"
                ]
                surround_index_list = surround_index_func(
                    pg_row_num, pg_column_num, i, j
                )
                state_update_prompt_other_agent += f"Agent[{i+0.5}, {j+0.5}]: I am in square[{i+0.5}, {j+0.5}], I can observe {square_item_list}, I can do "
                action_list = []
                for box in square_item_only_box:
                    for surround_index in surround_index_list:
                        action_list.append(f"move({box}, square{surround_index})")
                    if "target" + box[3:] in square_item_list:
                        action_list.append(f"move({box}, target{box[3:]})")
                state_update_prompt_other_agent += f"{action_list}\n"

    square_item_list = pg_dict_copy[str(pg_row_i + 0.5) + "_" + str(pg_column_j + 0.5)]
    square_item_only_box = [item for item in square_item_list if item[:3] == "box"]
    surround_index_list = surround_index_func(
        pg_row_num, pg_column_num, pg_row_i, pg_column_j
    )
    state_update_prompt_local_agent += f"Agent[{pg_row_i+0.5}, {pg_column_j+0.5}]: in square[{pg_row_i+0.5}, {pg_column_j+0.5}], can observe {square_item_list}, can do "
    action_list = []
    for box in square_item_only_box:
        for surround_index in surround_index_list:
            action_list.append(f"move({box}, square{surround_index})")
        if "target" + box[3:] in square_item_list:
            action_list.append(f"move({box}, target{box[3:]})")
    state_update_prompt_local_agent += f"{action_list}\n"
    return state_update_prompt_local_agent, state_update_prompt_other_agent


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


def with_action_syntactic_check_func(
    pg_dict_input: Dict[str, List[str]],
    response: str,
    alt_response: str,
    user_prompt_list_input: List[str],
    response_total_list_input: List[str],
    model_name: str,
    dialogue_history_method: str,
    is_judge: bool = False,
) -> Tuple[Union[str, Dict], List[int]]:
    """
    Checks and validates the syntactic correctness of the action plan.

    Args:
        pg_dict_input (Dict[str, List[str]]): Current state of the playground.
        response (str): Proposed action plan in JSON format.
        alt_response (str): Fall back response
        user_prompt_list_input (List[str]): List of user prompts.
        response_total_list_input (List[str]): List of previous responses.
        model_name (str): Name of the model generating the response.
        dialogue_history_method (str): Method for managing dialogue history.
        is_judge (bool, optional): Flag to indicate if the check is for a judge's response.

    Returns:
        Tuple[Union[str, Dict], List[int]]: Validated response and token count list.

    Notes:
        This only checks if the actions are valid, doesn't care about if it's json, if not json, directly fails it.
    """
    user_prompt_list = copy.deepcopy(user_prompt_list_input)
    response_total_list = copy.deepcopy(response_total_list_input)
    iteration_num = 0
    token_num_count_list_add = []
    while iteration_num < 6:
        valid = is_valid_json(response)
        count = 0
        print(response)
        while not valid:
            count += 1
            print(f"----------JSON Check {count} TIME----------")

            # need to give example or else LLM give {"Agent0_50_5": "move(box_green, target_green)", "Agent1_50_5": "move(box_red, target_red)"}
            messages = json_check_message_construct_func(response)
            response, token_num_count = LLaMA_response(messages, model_name)

            match = re.search(r"{.*}", response, re.DOTALL)
            if match:
                response = match.group()
                token_num_count_list_add.append(token_num_count)
                valid = is_valid_json(response)
            
            if count == 6:
                # if judge can't make right choice, fall back to HCA
                response = alt_response
                break

        response_total_list.append(response)
        # print(iteration_num, response_total_list)

        try:
            original_response_dict = json.loads(response)

            # Can't be '', must be ""
            # print(original_response_dict)
            # print('SUCCSESS')

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
                # print(f"Key: {key}, Value1: {value[0]}, Value2: {value[1]}")
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
                        # print(f"Error, Iteration Num: {iteration_num}, Key: {key}, Value1: {value[0]}, Value2: {value[1]}")
                        feedback += f"Your assigned task for {key[0]}_{key[1]} is not in the doable action list; "
        except:
            raise error(f"The response in wrong json format: {response}")
            feedback = "Your assigned plan is not in the correct json format as before. If your answer is empty dict, please check whether you miss to move box into the same colored target like move(box_blue, target_blue)"

        if feedback != "":
            feedback += "Please replan for all the agents again with the same ouput format. The output should have the same json format {Agent[0.5, 0.5]:move(box_blue, square[0.5, 1.5]), Agent[1.5, 0.5]:move...}. Do not explain, just directly output json directory. Your response:"
            print("----------Syntactic Check----------")
            # print(f"Response original: {response}")
            # print(f"Feedback: {feedback}")
            user_prompt_list.append(feedback)
            messages = message_construct_func(
                user_prompt_list, response_total_list, dialogue_history_method
            )  # message construction
            print(f"Length of messages {len(messages)}")
            response, token_num_count = LLaMA_response(messages, model_name)
            token_num_count_list_add.append(token_num_count)
            # print(f"Response new: {response}\n")
            if response == "Out of tokens":
                return response, token_num_count_list_add
            iteration_num += 1
        else:
            return response, token_num_count_list_add
    return "Syntactic Error", token_num_count_list_add


def action_from_response(
    pg_dict_input: Dict[str, List[str]], original_response_dict: Dict
) -> Tuple[str, Dict[str, List[str]]]:
    """
    Updates the environment state based on the actions in the response.

    Args:
        pg_dict_input (Dict[str, List[str]]): Current state of the playground.
        original_response_dict (Dict): Actions to be executed.

    Returns:
        Tuple[str, Dict[str, List[str]]]: Feedback string and updated playground state.
    """
    system_error_feedback = ""
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

    for key, value in transformed_dict.items():
        # print(f"Key: {key}, Value1: {value[0]}, Value2: {value[1]}")
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
            pg_dict_original[str(key[0]) + "_" + str(key[1])].remove(value[0])
            pg_dict_original[str(value[1][0]) + "_" + str(value[1][1])].append(value[0])
        elif (
            value[0] in pg_dict_original[str(key[0]) + "_" + str(key[1])]
            and type(value[1]) == str
            and value[1] in pg_dict_original[str(key[0]) + "_" + str(key[1])]
            and value[0][:4] == "box_"
            and value[1][:7] == "target_"
            and value[0][4:] == value[1][7:]
        ):
            pg_dict_original[str(key[0]) + "_" + str(key[1])].remove(value[0])
            pg_dict_original[str(key[0]) + "_" + str(key[1])].remove(value[1])
        else:
            # print(f"Error, Iteration Num: {iteration_num}, Key: {key}, Value1: {value[0]}, Value2: {value[1]}")
            system_error_feedback += f"Your assigned task for {key[0]}_{key[1]} is not in the doable action list; "

    return system_error_feedback, pg_dict_original


def env_create(
    pg_row_num: int = 5,
    pg_column_num: int = 5,
    box_num_low_bound: int = 2,
    box_num_upper_bound: int = 2,
    color_list: List[str] = ["blue", "red", "green", "purple", "orange"],
) -> Dict[str, List[str]]:
    """
    Creates a randomized environment state for the playground.

    Args:
        pg_row_num (int): Number of rows in the playground.
        pg_column_num (int): Number of columns in the playground.
        box_num_low_bound (int): Minimum number of boxes per color.
        box_num_upper_bound (int): Maximum number of boxes per color.
        color_list (List[str]): List of colors for boxes and targets.

    Returns:
        Dict[str, List[str]]: Initial state of the playground.
    """

    # pg_dict records the items in each square over steps, here in the initial setting, we randomly assign items into each square
    pg_dict = {}
    for i in range(pg_row_num):
        for j in range(pg_column_num):
            pg_dict[str(i + 0.5) + "_" + str(j + 0.5)] = []

    for color in color_list:
        box_num = random.randint(box_num_low_bound, box_num_upper_bound)
        for _ in range(box_num):
            N_box = random.randint(0, pg_row_num * pg_column_num - 1)
            a_box = N_box // pg_column_num
            b_box = N_box % pg_column_num
            N_target = random.randint(0, pg_row_num * pg_column_num - 1)
            a_target = N_target // pg_column_num
            b_target = N_target % pg_column_num
            pg_dict[str(a_box + 0.5) + "_" + str(b_box + 0.5)].append("box_" + color)
            pg_dict[str(a_target + 0.5) + "_" + str(b_target + 0.5)].append(
                "target_" + color
            )
    return pg_dict


def create_env1(Saving_path, repeat_num=10):
    """
    multi-agent-env/
    └── env_pg_state_2_2/
        ├── pg_state0/
        │   └── pg_state0.json
        ├── pg_state1/
        │   └── pg_state1.json
    ...

    Each is unique configuration of the environment


    """
    if not os.path.exists(Saving_path):
        os.makedirs(Saving_path, exist_ok=True)
    else:
        shutil.rmtree(Saving_path)
        os.makedirs(Saving_path, exist_ok=True)

    # for i, j in [(2, 2), (2, 4), (4, 4), (4, 8)]:
    for i, j in [(2, 2)]:

        if not os.path.exists(Saving_path + f"/env_pg_state_{i}_{j}"):
            os.makedirs(Saving_path + f"/env_pg_state_{i}_{j}", exist_ok=True)
        else:
            shutil.rmtree(Saving_path + f"/env_pg_state_{i}_{j}")
            os.makedirs(Saving_path + f"/env_pg_state_{i}_{j}", exist_ok=True)

        for iteration_num in range(repeat_num):
            # Define the total row and column numbers of the whole playground, and the item number of each colored target and box
            pg_row_num = i
            pg_column_num = j
            box_num_low_bound = 1
            box_num_upper_bound = 3
            # Define the used colors
            color_list = ["blue", "red", "green", "purple", "orange"]
            pg_dict = env_create(
                pg_row_num,
                pg_column_num,
                box_num_low_bound,
                box_num_upper_bound,
                color_list,
            )
            os.makedirs(
                Saving_path + f"/env_pg_state_{i}_{j}/pg_state{iteration_num}",
                exist_ok=True,
            )
            with open(
                Saving_path
                + f"/env_pg_state_{i}_{j}/pg_state{iteration_num}/pg_state{iteration_num}.json",
                "w",
            ) as f:
                json.dump(pg_dict, f)


Code_dir_path = "multi-agent-env/"
# The first time to create the environment, after that you can comment it

# Here we only create 1 instance of the random environment
create_env1(Code_dir_path, repeat_num=1)
