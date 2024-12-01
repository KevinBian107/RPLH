"""Modified from rplh-efficient, a particular instance of the decnetralzied efficient version"""

import sys
from pathlib import Path
import argparse

main_path = Path(__file__).resolve().parent.parent.parent
if str(main_path) not in sys.path:
    sys.path.append(str(main_path))

from rplh.llm.language_model import *
from rplh.h_efficient.env import *
from rplh.h_efficient.execution_checker import *
from rplh.llm.response_model import *

from rplh.h_efficient.memory import *
from rplh.d_efficient.memory import *

from rplh.rendering.render_state import *

import os
import json
import sys
import os
import pandas as pd
from functools import partial


def run_exp(
    Saving_path: str,
    pg_row_num: int,
    pg_column_num: int,
    iteration_num: int,
    query_time_limit: int,
    dialogue_history_method: str,
    model_name: str,
) -> tuple[
    list[str],
    list[str],
    list[dict],
    list[int],
    str,
    int,
    str,
]:
    """
    Runs the experiment for a multi-agent environment.

    Args:
        Saving_path (str): Path to save results.
        pg_row_num (int): Number of rows in the grid.
        pg_column_num (int): Number of columns in the grid.
        iteration_num (int): Iteration number of the environment.
        query_time_limit (int): Maximum number of queries allowed.
        dialogue_history_method (str): Method to handle dialogue history.
        model_name (str): Name of the model.

    Returns:
        Tuple: Contains lists of user prompts, responses, states, token counts,
        success/failure status, query index, and saving path result.
    """
    
    print('RUNNIN DECENTRALZIED RPLH')
    
    num_agent = pg_row_num * pg_column_num

    Saving_path_result = (
        Saving_path
        + f"/env_pg_state_{pg_row_num}_{pg_column_num}/pg_state{iteration_num}/{dialogue_history_method}_{model_name}"
    )
    os.makedirs(Saving_path_result, exist_ok=True)
    os.makedirs(Saving_path_result + f"/prompt", exist_ok=True)
    os.makedirs(Saving_path_result + f"/response", exist_ok=True)
    os.makedirs(Saving_path_result + f"/pg_state", exist_ok=True)
    os.makedirs(Saving_path_result + f"/dialogue_history", exist_ok=True)

    """This is information constant"""
    # TODO: Put this in a data tree
    data_dict = {
        "user_prompt_list": [],
        "response_total_list": [],
        "pg_state_list": [],
        "dialogue_history_list": [],
        "token_num_count_list": [],
        "attitude_info": [],
        "attitude_dialogue_dict": {},
        "pg_dict": None,  # For initial environment state
        "env_step": -1,
        "agree_num": 0,
    }

    # Load initial environment state
    with open(
        Saving_path
        + f"/env_pg_state_{pg_row_num}_{pg_column_num}/pg_state{iteration_num}/pg_state{iteration_num}.json",
        "r",
    ) as file:
        pg_dict = json.load(file)
        data_dict["pg_dict"] = pg_dict

    data_dict["pg_state_list"].append(data_dict["pg_dict"])

    with open("conversation.txt", "a") as f:
        f.truncate(0)

    print(f"query_time_limit: {query_time_limit}")

    render_graph_terminal_popup(data_dict["pg_dict"])

    for index_query_times in range(query_time_limit):
        
        data_dict['env_step']+=1
        
        result = {
            key: {
                "targets": sum(1 for item in items if "target" in item),
                "boxes": sum(1 for item in items if "box" in item),
            }
            for key, items in data_dict["pg_dict"].items()
        }
        result_df = pd.DataFrame(result).T
        print(result_df)
        print(result_df.sum(axis=0))
        
        if all(result_df.sum(axis=0)) == 0:
            break

        dialogue_history = ""
        data_local = {
            "prompt_list_dir": {},
            "response_list_dir": {},
            "local_agent_response_list_dir": {},
            "agent_dict": {},
            "agent_in_action_count": num_agent
        }

        data_local["local_agent_response_list_dir"]["feedback1"] = ""
        
        response = "You are the first agent, no one made response yet"
        
        counter = 0
        while data_dict['agree_num'] != data_local['agent_in_action_count'] // 2:
            # not half of people doing action agree, does not execute
            counter +=1
            print(f'#{counter} TIME IN WHILE LOOP')

            for local_agent_row_i in range(pg_row_num):
                for local_agent_column_j in range(pg_column_num):
                    
                    region_key = f"{local_agent_row_i+0.5}_{local_agent_column_j+0.5}"
                    if len(data_dict["pg_dict"][region_key]) == 0:
                        print(f"SKIPPING Agent[{local_agent_row_i+0.5},{local_agent_column_j+0.5}] as no blocks are present in its region.")
                        response_local_agent = 'I Agree'
                        continue
                    
                    # need to relapse responses to each agents
                    data_local["agent_dict"] = response

                    print(
                        f"-------###-------###-------###-------LOCAL_ROW_{local_agent_row_i}_COL_{local_agent_column_j}-------###-------###-------###-------"
                    )

                    local_agent_location = (
                        f"{local_agent_row_i}, {local_agent_column_j}"
                    )
                    
                    print(f'CURRENT AGENT IS Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]')
                    
                    if (
                        f"Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]"
                        in data_local["agent_dict"]
                    ) or (data_dict['env_step']==0):
                        
                        print(f'AGENT ACTION DICT UPDATING:{data_local["agent_dict"]}')
                        
                        # note, dict, this have space
                        data_local["prompt_list_dir"][
                            f"Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]"
                        ] = []
                        data_local["response_list_dir"][
                            f"Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]"
                        ] = []

                        (
                            state_update_prompt_local_agent,
                            state_update_prompt_other_agent,
                        ) = state_update_func_local_agent(
                            pg_row_num,
                            pg_column_num,
                            local_agent_row_i,
                            local_agent_column_j,
                            data_dict["pg_dict"],
                        )

                        # take in other agent's plan and give local prompt
                        local_reprompt = local_agent_prompt_func(
                            state_update_prompt_local_agent,
                            state_update_prompt_other_agent,
                            response,
                            data_dict,
                            dialogue_history_method,
                            local_agent_location,
                        )
                        
                        data_dict["user_prompt_list"].append(local_reprompt)
                        
                        partial_local_prompt_func = partial(
                            local_agent_prompt_func,
                            state_update_prompt_local_agent=state_update_prompt_local_agent,
                            state_update_prompt_other_agent=state_update_prompt_other_agent,
                            central_response=response,
                            data=data_dict,
                            dialogue_history_method=dialogue_history_method,
                            local_agent_location=local_agent_location
                        )
                        
                        data_local["prompt_list_dir"][
                            f"Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]"
                        ].append(local_reprompt)
                        message_raw = data_local["prompt_list_dir"][
                            f"Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]"
                        ]
                        response_raw = data_local["response_list_dir"][
                            f"Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]"
                        ]

                        messages = message_construct_func(
                            message_raw,
                            response_raw,
                            dialogue_history_method,
                        )

                        # given to other LLM, no synthetic check needed
                        response_local_agent, token_num_count =LLaMA_response_json(
                            messages, model_name, LocalAgent
                        )
                        data_dict["token_num_count_list"].append(token_num_count)
                        
                        print(f'RAW: {response_local_agent}')
                        raw_response = json.loads(response_local_agent)
                        raw_response = process_response(raw_response)
                        response = raw_response["actions_plan"]

                        # save user prompt
                        with open(
                            Saving_path_result
                            + "/prompt"
                            + "/user_prompt_"
                            + str(index_query_times),
                            "w",
                        ) as f:
                            f.write(data_dict["user_prompt_list"][-1])
                        
                        response, token_num_count_list_add = (
                        with_action_syntactic_check_func(
                            data_dict["pg_dict"],
                            response,
                            [local_reprompt],
                            [response],
                            model_name,
                            dialogue_history_method,
                            partial_local_prompt_func,
                            )
                        )
                        data_dict["token_num_count_list"] = (
                            data_dict["token_num_count_list"] + token_num_count_list_add
                        )

                        with open("conversation.txt", "a") as f:
                            message = f"------###------###------LOCAL_ROW_{local_agent_row_i}_COL_{local_agent_column_j}------###------###------: \n {response_local_agent} \n \n"
                            f.write(message)

                        if response_local_agent != "I Agree":
                            data_local["local_agent_response_list_dir"][
                                "feedback1"
                            ] += f"Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]: {response_local_agent}\n"

                            dialogue_history += f"Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]: {response_local_agent}\n"
                        else:
                            print("I Agree")
                            data_dict["agree_num"] += 1
                            # agree no judge, use HCA response diretcly, avoid error.
                            continue
                        
                    else:
                        # no action assiged, noes not participate
                        data_local['agent_in_action_count']-=1

                    # -----------------------------------------RECONSTRUCT MESSAGES-----------------------------------------#
                    if (
                        data_local["local_agent_response_list_dir"]["feedback1"] != ""
                    ):  # if not I agree
                        print("I Don't Agree")

                        # TODO: Do we need this?
                        data_local["local_agent_response_list_dir"][
                            "feedback1"
                        ] += FEEDBACK_LCOAL1
                    
                    
                    data_dict["dialogue_history_list"].append(dialogue_history)
                
                    # not acting agent does not communicate, resolve missing variable issue
                    if (
                        f"Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]"
                        in data_local["agent_dict"]
                        ):
                    
                        data_dict["attitude_dialogue_dict"][
                            f"Agent[{local_agent_location}]"
                        ] = response_local_agent
                    
                     # inner for loop ends here
                     
        # one response to give, outside while
        data_dict["response_total_list"].append(
            response
        )     
                
        print(
            "-------###-------###-------###-------ATTITUDE CHECK-------###-------###-------###-------"
        )

        attitude_prompt = attitude_agent_prompt_func(
            data_dict["attitude_dialogue_dict"]
        )
        attitude_message = attitude_message_construct_func(attitude_prompt)
        attitude_info, token_num_count = LLaMA_response(
            attitude_message, model_name
        )
        data_dict["token_num_count_list"].append(token_num_count)

        data_dict["attitude_info"].append(attitude_info)

        with open("conversation.txt", "a") as f:
            message = f"------###------###------ATTITUDE_AGENT------###------###------: \n {attitude_info} \n \n"
            f.write(message)
        
        # second outer while loop ends here
        
        print(
            "-------###-------###-------###-------EXECUTION-------###-------###-------###-------"
        )

        original_response_dict = data_dict["response_total_list"][-1]

        with open(
            Saving_path_result
            + "/response"
            + "/response"
            + str(data_dict["env_step"])
            + ".json",
            "w",
        ) as f:
            print("SAVE RESPONSE \n")
            json.dump(original_response_dict, f)

        with open(
            Saving_path_result
            + "/dialogue_history"
            + "/dialogue_history"
            + str(data_dict["env_step"])
            + ".txt",
            "w",
        ) as f:
            print("SAVE DIALOGUE \n")
            json.dump(data_dict["dialogue_history_list"], f)

        try:
            system_error_feedback, pg_dict_returned = action_from_response(
                data_dict["pg_dict"], original_response_dict
            )
            if system_error_feedback != "":
                print(system_error_feedback)

            data_dict["pg_dict"] = pg_dict_returned
            render_graph_terminal_popup(data_dict["pg_dict"])

        except:
            success_failure = "Hallucination of wrong plan"
            pass

        # need to append new states to state list
        data_dict["pg_state_list"].append(data_dict["pg_dict"])

        data_dict["agree_num"] = 0

        # -----------------------------------------TASK SUCCESS CHECK-----------------------------------------#
        count = 0
        for ky, value in data_dict["pg_dict"].items():
            count += len(value)
            print(f'STILL HAVE {count} LEFT')
        if count == 0:
            break

    # -----------------------------------------TASK SUCCESS OUT-----------------------------------------#
    if index_query_times < query_time_limit - 1:
        success_failure = "success"
    else:
        success_failure = "failure over query time limit"
    print(success_failure)

    return (
        data_dict["user_prompt_list"],
        data_dict["response_total_list"],
        data_dict["pg_state_list"],
        data_dict["token_num_count_list"],
        success_failure,
        index_query_times,
        Saving_path_result,
    )


# -----------------------------------------RUNNING EXPERIMENT-----------------------------------------#

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a 4-agent experiment.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name.")

    args = parser.parse_args()

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    Code_dir_path = os.path.join(os.getcwd())
    os.makedirs(Code_dir_path, exist_ok=True)
    saving_path = Code_dir_path + "/multi-agent-env"

    # 4 agent in total
    pg_row_num = 2
    pg_column_num = 2
    iteration_num = 0
    query_time_limit = 10
    model_name = args.model_name
    print(f"-------------------Model name: {model_name}-------------------")

    (
        user_prompt_list,
        response_total_list,
        pg_state_list,
        success_failure,
        index_query_times,
        token_num_count_list,
        Saving_path_result,
    ) = run_exp(
        saving_path,
        pg_row_num,
        pg_column_num,
        iteration_num,
        query_time_limit,
        dialogue_history_method="_w_markovian_state_action_history",
        model_name=model_name,
    )