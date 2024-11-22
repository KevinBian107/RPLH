"""Need good documentation"""

from LLM import *
from prompt import *
from env_create import *
from execution_checker import *
import os
import json
import re
import sys
import os
from typing import Dict, List, Tuple, Union
import pandas as pd
from render_func import *


def run_exp(
    Saving_path: str,
    pg_row_num: int,
    pg_column_num: int,
    iteration_num: int,
    query_time_limit: int,
    dialogue_history_method: str,
) -> Tuple[
    List[str],
    List[str],
    List[Dict],
    List[int],
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

    Returns:
        Tuple: Contains lists of user prompts, responses, states, token counts,
        success/failure status, query index, and saving path result.
    """

    Saving_path_result = (
        Saving_path
        + f"/env_pg_state_{pg_row_num}_{pg_column_num}/pg_state{iteration_num}/{dialogue_history_method}_{model_name}"
    )
    os.makedirs(Saving_path_result, exist_ok=True)
    os.makedirs(Saving_path_result + f"/prompt", exist_ok=True)
    os.makedirs(Saving_path_result + f"/response", exist_ok=True)
    os.makedirs(Saving_path_result + f"/pg_state", exist_ok=True)
    os.makedirs(Saving_path_result + f"/dialogue_history", exist_ok=True)
    os.makedirs(Saving_path_result + f"/hca_agent_response", exist_ok=True)

    """This is information constant"""
    # TODO: Put this in a data tree
    data_dict = {
        "user_prompt_list": [],
        "response_total_list": [],
        "pg_state_list": [],
        "dialogue_history_list": [],
        "token_num_count_list": [],
        "hca_agent_response_list": [],
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
        # print(pg_dict)
        data_dict["pg_dict"] = pg_dict

    num_agent = pg_row_num * pg_column_num
    data_dict["pg_state_list"].append(data_dict["pg_dict"])

    with open("conversation.txt", "a") as f:
        f.truncate(0)

    print(f"query_time_limit: {query_time_limit}")

    render_graph_terminal_popup(data_dict["pg_dict"])

    for index_query_times in range(query_time_limit):
        # -----------------------------------------ONE HCA AGENT THINK BY THEMSELVES ONCE-----------------------------------------#
        for a in range(num_agent):
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

            print(
                f"-------###-------###-------###-------HCA_AGENT_{a}-------###-------###-------###-------"
            )

            """FOR NUM_AGENT, ITERATIVELY DO"""

            HCA_agent_location = list(data_dict["pg_dict"].keys())[a]
            print(f"HCA Agent {a} is at: [{HCA_agent_location}]")

            data_dict["env_step"] += 1

            # sate0 is initial state
            with open(
                Saving_path_result
                + "/pg_state"
                + "/pg_state"
                + str(data_dict["env_step"])
                + ".json",
                "w",
            ) as f:
                print("SAVE INITIAL STATE \n")
                json.dump(data_dict["pg_dict"], f)

            # at second iter, should have more info, get available actions
            state_update_prompt = state_update_func(
                pg_row_num, pg_column_num, data_dict["pg_dict"]
            )

            if data_dict["env_step"] == 0:
                att_prompt = None
            else:
                att_prompt = data_dict["attitude_info"][-1]

            user_prompt_1 = rplh_prompt_func(
                state_update_prompt,
                data_dict,
                dialogue_history_method,
                HCA_agent_location,
                att_prompt,
            )
            data_dict["user_prompt_list"].append(user_prompt_1)
            messages = message_construct_func(
                [user_prompt_1], [], dialogue_history_method
            )

            raw_response, token_num_count = LLaMA_response(messages, model_name)

            # save user prompt
            with open(
                Saving_path_result
                + "/prompt"
                + "/user_prompt_"
                + str(index_query_times),
                "w",
            ) as f:
                f.write(data_dict["user_prompt_list"][-1])

            # -----------------------------------------SYNTHACTIC CHECK-----------------------------------------#
            data_dict["token_num_count_list"].append(token_num_count)

            # TODO: ADD BAN HERE

            match = re.search(r"{.*}", raw_response, re.DOTALL)
            if match:
                response = match.group()

            if response[0] == "{" and response[-1] == "}":
                response, token_num_count_list_add = with_action_syntactic_check_func(
                    data_dict["pg_dict"],
                    response,
                    [user_prompt_1],
                    [],
                    model_name,
                    "_w_all_dialogue_history",
                    False,
                )
                data_dict["token_num_count_list"] = (
                    data_dict["token_num_count_list"] + token_num_count_list_add
                )
                print(f"AGENT ACTION RESPONSE: {response}")
            else:
                raise ValueError(f"Response format error: {response}")
            if response == "Out of tokens":
                pass
            elif response == "Syntactic Error":
                pass

            data_dict["hca_agent_response_list"].append(raw_response)
            data_dict['attitude_dialogue_dict'][f'Agent[{HCA_agent_location}]'] = raw_response

            with open(
                Saving_path_result
                + "/hca_agent_response"
                + "/hca_agent_response"
                + str(index_query_times)
                + ".txt",
                "w",
            ) as f:
                print("SAVE HCA RESPONSE \n")
                json.dump(data_dict["hca_agent_response_list"], f)

            # write after syntactic check
            with open("conversation.txt", "a") as f:
                message = f"------###------###------HCA_{a}------###------###------: \n {raw_response} \n \n"
                sep = f"\n-------###-------###-------###--------------###-------###-------###-------\n"
                f.write(sep)
                length = str(len(data_dict["pg_state_list"]))
                f.write(f"ALL STATE STORAGE LENGTH: {length} \n")
                f.write(message)
            
            """This for loop ends here for all agents doing centralized planning by themselves"""

            # -----------------------------------------FOR EACH AGENT RECIEVES COMMAND FROM THE CURRENT HELLUCINATING MAIN AGENT-----------------------------------------#
            dialogue_history = ""
            print(f"ORIGINAL PLAN: {response}")
            data_local = {
                "prompt_list_dir": {},
                "response_list_dir": {},
                "local_agent_response_list_dir": {},
                "agent_dict": {},
                "judge_resonse": {},
            }

            data_local["local_agent_response_list_dir"]["feedback1"] = ""
            data_local["agent_dict"] = json.loads(response)

            for local_agent_row_i in range(pg_row_num):
                
                for local_agent_column_j in range(pg_column_num):

                    # TODO: IF BAN TRIGGERED, SKIP

                    print(
                        f"-------###-------###-------###-------LOCAL_ROW_{local_agent_row_i}_COL_{local_agent_column_j}-------###-------###-------###-------"
                    )

                    local_agent_location = (
                        f"{local_agent_row_i}, {local_agent_column_j}"
                    )

                    if (
                        f"Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]"
                        in data_local["agent_dict"]
                    ):
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

                        # take in original HCA plan and give local prompt
                        local_reprompt = dialogue_func(
                            state_update_prompt_local_agent,
                            state_update_prompt_other_agent,
                            response,
                            data_dict,
                            dialogue_history_method,
                            local_agent_location,
                            att_prompt,  # refer back to HCA part
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
                            dialogue_history_method,  # "_w_all_dialogue_history"
                        )

                        # given to other LLM, no synthetic check needed
                        response_local_agent, token_num_count = LLaMA_response(
                            messages, model_name
                        )
                        data_dict["token_num_count_list"].append(token_num_count)

                        with open("conversation.txt", "a") as f:
                            message = f"------###------###------LOCAL_{a}_ROW_{local_agent_row_i}_COL_{local_agent_column_j}------###------###------: \n {response_local_agent} \n \n"
                            f.write(message)

                        if response_local_agent != "I Agree":
                            data_local["local_agent_response_list_dir"][
                                "feedback1"
                            ] += f"Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]: {response_local_agent}\n"
                            dialogue_history += f"Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]: {response_local_agent}\n"

                            data_dict["agree_num"] += 1

                            if (
                                data_dict["agree_num"]
                                >= (pg_column_num + pg_row_num) // 2
                            ):
                                break

                            # agree no judge, use HCA response diretcly, avoid error.
                            continue

                    # -----------------------------------------RECONSTRUCT MESSAGES-----------------------------------------#
                    if (
                        data_local["local_agent_response_list_dir"]["feedback1"] != ""
                    ):  # if not I agree

                        # once not agree, set to zero to re-discuss lat plan
                        data_dict["agree_num"] = 0
                        print("I Don't Agree")
                        data_local["local_agent_response_list_dir"][
                            "feedback1"
                        ] += FEEDBACK_LCOAL1

                    # -----------------------------------------JUDGE IF NO AGREEMENT MET, SEND MESSAGE IF AGREE-----------------------------------------#
                    # This message should be constructed for teh judge, include both central and local response, agree on global plan
                    print(
                        f"-------###-------###-------###-------JUDGE_ON_ROW_{local_agent_row_i}_COL_{local_agent_column_j}-------###-------###-------###-------"
                    )
                    local_response = data_local["local_agent_response_list_dir"][
                        "feedback1"
                    ]
                    cen_response = data_dict["user_prompt_list"][-1]

                    judge_prompt = judge_prompt_func(
                        local_response, cen_response, data_dict["pg_dict"]
                    )
                    messages = judge_message_construct_func([judge_prompt])
                    response_judge, token_num_count = LLaMA_response(
                        messages, model_name
                    )

                    # -----------------------------------------SYNTACTIC CHECK FOR JUDGE-----------------------------------------#
                    data_dict["token_num_count_list"].append(token_num_count)
                    match = re.search(r"{.*}", response_judge, re.DOTALL)
                    # match not right
                    if match:
                        response = match.group()

                        response, token_num_count_list_add = (
                            with_action_syntactic_check_func(
                                data_dict["pg_dict"],
                                response,
                                [judge_prompt],
                                [response],
                                model_name,
                                "_w_all_dialogue_history",
                                is_judge=True,
                            )
                        )
                        data_dict["token_num_count_list"] = (
                            data_dict["token_num_count_list"] + token_num_count_list_add
                        )

                    # after syntactic checks
                    with open("conversation.txt", "a") as f:
                        messages = f"------###------###------JUDGE_{a}_ROW_{local_agent_row_i}_COL_{local_agent_column_j}------###------###------: \n {response_judge} \n \n"
                        f.write(messages)

                    print(f"JUDGE MODIFIED:\n {response}")

                else:
                    print(f"ORIGINAL PLAN:\n {response}")
                    pass
                data_dict["dialogue_history_list"].append(dialogue_history)
                
                data_dict['attitude_dialogue_dict'][f'Agent[{local_agent_location}]'] = response_local_agent

            data_dict["response_total_list"].append(response)

            # -----------------------------------------ATTITUDE CHECK AFTER ALL AGENT-----------------------------------------#
            print(
                "-------###-------###-------###-------ATTITUDE CHECK-------###-------###-------###-------"
            )
            
            # print(data_dict['attitude_dialogue_dict'])

            attitude_info = attitude_agent_prompt_func(
                data_dict['attitude_dialogue_dict']
            )

            pg_dict["attitude_info"].appned(attitude_info)

            # -----------------------------------------EXECUTION OF ACTION AT EACH HCA AGENT LEVEL-----------------------------------------#
            print(
                "-------###-------###-------###-------EXECUTION-------###-------###-------###-------"
            )

            original_response_dict = json.loads(
                data_dict["response_total_list"][index_query_times]
            )

            render_map_terminal_popup(data_dict["pg_dict"], [original_response_dict])

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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
Code_dir_path = os.path.join(os.getcwd())
os.makedirs(Code_dir_path, exist_ok=True)
saving_path = Code_dir_path + "/multi-agent-env"

# 4 agent in total
pg_row_num = 2
pg_column_num = 2
iteration_num = 0
query_time_limit = 10  # now it's iteration
model_name = "qwen2.5:14b-instruct-q3_K_L"
print(f"-------------------Model name: {model_name}-------------------")

#'_w_all_dialogue_history', '_w_compressed_dialogue_history', '_w_only_state_action_history'
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
    dialogue_history_method="_w_only_state_action_history",
)

with open(Saving_path_result + "/token_num_count.txt", "w") as f:
    print("SAVE TOKEN NUM \n")
    for token_num_num_count in token_num_count_list:
        f.write(str(token_num_num_count) + "\n")

with open(Saving_path_result + "/success_failure.txt", "w") as f:
    print("SAVE RESULT \n")
    f.write(success_failure)

with open(Saving_path_result + "/env_action_times.txt", "w") as f:
    print("SAVE ACTION TIME \n")
    f.write(f"{index_query_times+1}")
