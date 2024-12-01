"""BoxMove environment for multi-agent collaboration."""

import sys
from pathlib import Path

main_path = Path(__file__).resolve().parent.parent
if str(main_path) not in sys.path:
    sys.path.append(str(main_path))
    
from rplh_efficient.memory import *
import os
import json
import re
import copy
import numpy as np
import shutil
import random


def surround_index_func(
    row_num: int, coloum_num: int, row_index: int, coloum_index: int
) -> list[list[float]]:
    """
    Calculates the indices of surrounding cells for a given cell in a grid.

    Args:
        row_num (int): Total number of rows.
        col_num (int): Total number of columns.
        row_index (int): Row index of the target cell.
        col_index (int): Column index of the target cell.

    Returns:
        list[list[float]]: List of coordinates for surrounding cells.
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
    pg_row_num: int, pg_column_num: int, pg_dict: dict[str, list[str]]
) -> str:
    """
    Describes the environment and possible actions for the central agent.

    Args:
        pg_row_num (int): Number of rows in the playground.
        pg_column_num (int): Number of columns in the playground.
        pg_dict (dict[str, list[str]]): State of the playground.

    Returns:
        str: State update prompt for the central agent.
    """

    pg_dict_copy = copy.deepcopy(pg_dict)
    state_update_prompt = ""
    # agent_action = dict()
    for i in range(pg_row_num):
        for j in range(pg_column_num):
            square_item_list = pg_dict_copy[str(i + 0.5) + "_" + str(j + 0.5)]
            square_item_only_box = [
                item for item in square_item_list if item[:3] == "box"
            ]
            surround_index_list = surround_index_func(pg_row_num, pg_column_num, i, j)
            state_update_prompt += f"Agent[{i+0.5}, {j+0.5}]: I am in square[{i+0.5}, {j+0.5}], I can observe {square_item_list}, "
            action_list = []
            for box in square_item_only_box:
                for surround_index in surround_index_list:
                    action_list.append(f"move({box}, square{surround_index})")
                if "target" + box[3:] in square_item_list:
                    action_list.append(f"move({box}, target{box[3:]})")
            if len(action_list) != 0:
                state_update_prompt += (
                    f"I can do one of the following action: {action_list}\n"
                )
            else:
                state_update_prompt += "\n"  # I can do nothing
            # agent_action[f"Agent[{i+0.5}, {j+0.5}]"] = action_list
    return state_update_prompt


def state_update_func_local_agent(
    pg_row_num: int,
    pg_column_num: int,
    pg_row_i: int,
    pg_column_j: int,
    pg_dict: dict[str, list[str]],
) -> tuple[str, str]:
    """
    Describes the environment and possible actions for a specific local agent.

    Args:
        pg_row_num (int): Number of rows in the playground.
        pg_column_num (int): Number of columns in the playground.
        pg_row_i (int): Row index of the local agent.
        pg_column_j (int): Column index of the local agent.
        pg_dict (dict[str, list[str]]): State of the playground.

    Returns:
        tuple[str, str]: Prompts describing the environment and actions for the local agent
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


def action_from_response(
    pg_dict_input: dict[str, list[str]], original_response_dict: dict
) -> tuple[str, dict[str, list[str]]]:
    """
    Updates the environment state based on the actions in the response.

    Args:
        pg_dict_input (dict[str, list[str]]): Current state of the playground.
        original_response_dict (dict): Actions to be executed.

    Returns:
        tuple[str, dict[str, list[str]]]: Feedback string and updated playground state.
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
    color_list: list[str] = ["blue", "red", "green", "purple", "orange"],
) -> dict[str, list[str]]:
    """
    Creates a randomized environment state for the playground.

    Args:
        pg_row_num (int): Number of rows in the playground.
        pg_column_num (int): Number of columns in the playground.
        box_num_low_bound (int): Minimum number of boxes per color.
        box_num_upper_bound (int): Maximum number of boxes per color.
        color_list (list[str]): List of colors for boxes and targets.

    Returns:
        dict[str, list[str]]: Initial state of the playground.
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
