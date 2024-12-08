import sys
from pathlib import Path
import argparse
import os
import json
import pandas as pd
from functools import partial

main_path = Path(__file__).resolve().parent.parent.parent.parent
if str(main_path) not in sys.path:
    sys.path.append(str(main_path))

from rplh.llm.language_model import *
from rplh.llm.response_model import *

from rplh.systems.h_efficient.memory.memory_standard import *
from rplh.systems.h_efficient.memory.memory_agent import (
    rplh_prompt_agent_func,
    dialogue_agent_func,
)

from rplh.env.env import *
from rplh.systems.h_efficient.execution_checker import *

from rplh.rendering.render_state import *


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

    print("RUNNIN EFFICIENT RPLH")
    
    # load attitudes
    att_config = load_config("rplh/configs/attitude_config.yaml")
    att_config = att_config["h_efficient_agent"]

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
    os.makedirs(Saving_path_result + f"/agent_model", exist_ok=True)
    os.makedirs(Saving_path_result + f"/spy_model", exist_ok=True)

    """This is information constant"""
    # TODO: Put this in a data tree
    data_dict = {
        "user_prompt_list": [],
        "response_total_list": [],
        "pg_state_list": [],
        "dialogue_history_list": [],
        "token_num_count_list": [],
        "hca_agent_response_list": [],
        "hca_conversation_list": [],
        "pg_dict": None,  # For initial environment state
        "env_step": -1,
        "agree_num": {},
        "agent_model": [],
        "spy_model": [],
        "spy_detect": 0,
    }

    # Load initial environment state
    with open(
        Saving_path
        + f"/env_pg_state_{pg_row_num}_{pg_column_num}/pg_state{iteration_num}/pg_state{iteration_num}.json",
        "r",
    ) as file:
        pg_dict = json.load(file)
        data_dict["pg_dict"] = pg_dict

    num_agent = pg_row_num * pg_column_num
    data_dict["pg_state_list"].append(data_dict["pg_dict"])

    with open("conversation.txt", "a") as f:
        f.truncate(0)

    print(f"query_time_limit: {query_time_limit}")

    # render_graph_terminal_popup(
    #     data_dict["pg_dict"],
    #     pg_row_num=pg_row_num,
    #     pg_column_num=pg_column_num,
    # )

    for index_query_times in range(10):
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

            if all(result_df.sum(axis=0)) == 0:
                break

            print(
                f"-------###-------###-------###-------HCA_AGENT_{a}-------###-------###-------###-------"
            )

            """FOR NUM_AGENT, ITERATIVELY DO"""

            location = (list(data_dict["pg_dict"].keys())[a]).split("_")
            HCA_agent_location = f"Agent[{location[0]}, {location[1]}]"
            print(f"HCA Agent {a} is [{HCA_agent_location}]")

            agent_key = list(data_dict["pg_dict"].keys())[a]
            box_num = len(data_dict["pg_dict"][agent_key])
            print(f"NUM OF BOX IN HCA LOCATION: {box_num}")
            if box_num == 0:
                print(
                    f"Skip HCA {HCA_agent_location} since no boxes and targets in its region."
                )
                continue

            data_dict["env_step"] += 1
            
            if data_dict["env_step"] >= query_time_limit:
                print("QUERY TIME LIMIT REACHED")
                break

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
            state_update_prompt, agent_action = state_update_func(
                pg_row_num, pg_column_num, data_dict["pg_dict"]
            )
            # print(f"STATE UPDATE PROMPT: {state_update_prompt}")
            
            if data_dict["env_step"] == 0:
                local_response = ""
            else:
                local_response = data_dict["dialogue_history_list"][-1]
                
            user_prompt_1 = rplh_prompt_agent_func(
                state_update_prompt,
                data_dict,
                dialogue_history_method,
                HCA_agent_location,
                local_agent_location="",
                local_responses=local_response,
                judging_mode=False,
            )
            
            # partial function
            partial_rplh_prompt_func = partial(
                rplh_prompt_agent_func,
                state_update_prompt=state_update_prompt,
                data=data_dict,
                dialogue_history_method=dialogue_history_method,
                HCA_agent_location=HCA_agent_location,
                local_agent_location="",
                local_responses=local_response,
                judging_mode=False,
            )

            data_dict["user_prompt_list"].append(user_prompt_1)
            messages = message_construct_func(
                [user_prompt_1], [], dialogue_history_method
            )

            raw_response, token_num_count = LLaMA_response_json(
                messages, model_name, HCA_AgentModel
            )
            # print(f'RAW: {raw_response}')
            raw_response = json.loads(raw_response)
            raw_response = process_response(raw_response)
            response_str = "\n".join([f"{k}: {v}" for k, v, in raw_response.items()])
            response = raw_response["actions_plan"]

            data_dict['agent_model'].append(raw_response["agent_model"])
            data_dict['spy_model'].append(raw_response["spy_model"])

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

            # print(f"HCA Raw Response: {raw_response}")

            # REDO HCA
            response, token_num_count_list_add = with_action_syntactic_check_func(
                data_dict["pg_dict"],
                response,
                [user_prompt_1],
                [response],
                model_name,
                dialogue_history_method,
                partial_rplh_prompt_func,
                state_update_prompt,
                agent_action, 
                False,
            )
            data_dict["token_num_count_list"] = (
                data_dict["token_num_count_list"] + token_num_count_list_add
            )
            print(f"AGENT ACTION RESPONSE: {response}")
            # else:
            #    raise ValueError(f"No action format found in raw response: {raw_response}")

            data_dict["hca_agent_response_list"].append(response)
            data_dict["hca_conversation_list"].append(response_str)
            
            with open(
                Saving_path_result
                + "/agent_model"
                + "/agent_model"
                + str(data_dict["env_step"])
                + "_"
                + str(HCA_agent_location)
                + ".json",
                "w",
            ) as f:
                print("SAVE AGENT MODEL \n")
                json.dump(data_dict['agent_model'][-1], f)
            
            with open(
                Saving_path_result
                + "/spy_model"
                + "/spy_model"
                + str(data_dict["env_step"])
                + "_"
                + str(HCA_agent_location)
                + ".json",
                "w",
            ) as f:
                print("SAVE SPY MODEL \n")
                json.dump(data_dict['spy_model'][-1], f)

            with open(
                Saving_path_result
                + "/hca_agent_response"
                + "/hca_agent_response"
                + str(index_query_times)
                + ".txt",
                "w",
            ) as f:
                print("SAVE HCA RESPONSE \n")
                json.dump(data_dict["hca_conversation_list"], f)

            # write after syntactic check
            with open("conversation.txt", "a") as f:
                message = f"------###------###------HCA_{a}------###------###------: \n {response_str} \n \n"
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
                "last_agent_response": "",
            }

            data_local["local_agent_response_list_dir"]["feedback1"] = ""

            local_agent_iter = 0
            data_dict["agree_num"][f"HCA_{a}"] = 0
            for local_agent_row_i in range(pg_row_num):

                for local_agent_column_j in range(pg_column_num):

                    local_agent_iter += 1
                    
                    # two version, identify needed
                    region_key = f"{local_agent_row_i+0.5}_{local_agent_column_j+0.5}"
                    if len(data_dict["pg_dict"][region_key]) == 0:
                        if (
                        f"Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]"
                        not in data_local["agent_dict"]
                        ):
                            print(
                                f"SKIPPING Agent[{local_agent_row_i+0.5},{local_agent_column_j+0.5}] as no blocks are present in its region and no action assigned by HCA."
                            )
                            continue

                    # need to relapse responses to each agents
                    data_local["agent_dict"] = response

                    if (
                        f"Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]"
                        not in data_local["agent_dict"]
                    ):
                        # fixe bugs here
                        print("Agent not in HCA plan \n")
                        continue

                    if (
                        f"Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]"
                        in data_local["agent_dict"]
                    ):
                        local_agent_location = (
                            f"Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]"
                        )
                        
                        print(f"CURRENT AGENT IS {local_agent_location}")
                        print(f'AGENT ACTION DICT UPDATING:{data_local["agent_dict"]}')
                        
                        # decide attitude
                        if local_agent_location in att_config["spy_agent"]:
                            assigned_attitude = "SPY"
                            #check if spy is out
                            print(f'SPY HAS {data_dict["pg_dict"][region_key]} BOXES/TARGETS')
                            if len(data_dict["pg_dict"][region_key]) == 0:
                                print(f"SPY AT {local_agent_location} HAS NO BOXES?TARGETS, OUT")
                                data_dict['spy_detect'] += 1
                                system_message = {"System message": f"{local_agent_location} is a spy, do not move boxes to that grid"}
                                data_dict['spy_model'].append(system_message)
                                
                                if data_dict['spy_detect'] == len(att_config["spy_agent"]):
                                    print("ALL SPY OUT, END")
                                    system_message = {"System message": "All spys does not have boxes and targets in their region, focus on moving boxes to their location, leave spy_model empty."}
                                    data_dict['spy_model'].append(system_message)
                                    
                        elif local_agent_location in att_config["nice_agent"]:
                            assigned_attitude = "NICE"
                        elif local_agent_location in att_config["critic_agent"]:
                            assigned_attitude = "CRITIC"
                        else:
                            assigned_attitude = "NEUTRAL"
                            
                        print(
                            f"-------###-------###-------###-------{assigned_attitude}_LOCAL_ROW_{local_agent_row_i+0.5}_COL_{local_agent_column_j+0.5}-------###-------###-------###-------"
                        )

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
                            agent_action # this need update as well
                        ) = state_update_func_local_agent(
                            pg_row_num,
                            pg_column_num,
                            local_agent_row_i,
                            local_agent_column_j,
                            data_dict["pg_dict"],
                        )

                        # take in original HCA plan and give local prompt
                        local_reprompt = dialogue_agent_func(
                            state_update_prompt_local_agent,
                            state_update_prompt_other_agent,
                            response,
                            data_dict,
                            dialogue_history_method,
                            local_agent_location,
                            assigned_attitude=assigned_attitude,
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

                        # print(f"LOCAL AGENT RESPONSE: {response_local_agent}")

                        data_dict["token_num_count_list"].append(token_num_count)

                        if ("I Agree" not in response_local_agent) or ("I Disagree" in response_local_agent) or ("However" in response_local_agent):
                            print("I Don't Agree")
                            data_local["local_agent_response_list_dir"][
                                "feedback1"
                            ] += f"Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]: {response_local_agent}\n"

                            dialogue_history += f"Agent[{local_agent_row_i+0.5}, {local_agent_column_j+0.5}]: {response_local_agent}\n"

                            with open("conversation.txt", "a") as f:
                                message = f"------###------###------{assigned_attitude}_DISAGREEING_LOCAL_{a}_ROW_{local_agent_row_i+0.5}_COL_{local_agent_column_j+0.5}------###------###------: \n {response_local_agent} \n \n"
                                f.write(message)

                        else:
                            print("I Agree")
                            data_dict["agree_num"][f"HCA_{a}"] += 1
                            # agree no judge, use HCA response diretcly, avoid error.
                            with open("conversation.txt", "a") as f:
                                message = f"------###------###------{assigned_attitude}_AGREEING_LOCAL_{a}_ROW_{local_agent_row_i+0.5}_COL_{local_agent_column_j+0.5}------###------###------: \n {response_local_agent} \n \n"
                                f.write(message)

                            continue

                    # -----------------------------------------RECONSTRUCT MESSAGES-----------------------------------------#
                    if (
                        data_local["local_agent_response_list_dir"]["feedback1"] != ""
                    ):  # if not I agree
                        # once not agree, set to zero to re-discuss lat plan
                        data_local["local_agent_response_list_dir"][
                            "feedback1"
                        ] += FEEDBACK_LCOAL1

                    # -----------------------------------------JUDGE IF NO AGREEMENT MET, SEND MESSAGE IF AGREE-----------------------------------------#
                    # This message should be constructed for teh judge, include both central and local response, agree on global plan
                    print(
                        f"-------###-------###-------###-------{assigned_attitude}_HCA_JUDGE_ON_ROW_{local_agent_row_i}_COL_{local_agent_column_j}-------###-------###-------###-------"
                    )
                    local_response = data_local["local_agent_response_list_dir"][
                        "feedback1"
                    ]
                    cen_response = data_dict["hca_agent_response_list"][-1]

                    judge_prompt = rplh_prompt_agent_func(
                        state_update_prompt,
                        data_dict,
                        dialogue_history_method,
                        HCA_agent_location,
                        local_agent_location=local_agent_location,
                        local_responses=[local_response],
                        judging_mode=True,
                    )

                    # partial function
                    partial_judge_prompt_func = partial(
                        rplh_prompt_agent_func,
                        state_update_prompt=state_update_prompt,
                        data=data_dict,
                        dialogue_history_method=dialogue_history_method,
                        HCA_agent_location=HCA_agent_location,
                        local_agent_location=local_agent_location,
                        local_responses=[local_response],
                        judging_mode=True,
                    )

                    messages = message_construct_func(
                        [judge_prompt],
                        [],
                        dialogue_history_method,
                    )

                    raw_response_judge, token_num_count = LLaMA_response_json(
                        messages, model_name, HCA_Judge
                    )
                    raw_response_judge = json.loads(raw_response_judge)
                    raw_response_judge = process_response(raw_response_judge)
                    response_str_judge = "\n".join(
                        [f"{k}: {v}" for k, v, in raw_response_judge.items()]
                    )
                    response_judge = raw_response_judge["actions_plan"]

                    data_dict['agent_model'].append(raw_response["agent_model"])
                    data_dict['spy_model'].append(raw_response["spy_model"])
                    
                    with open(
                        Saving_path_result
                        + "/agent_model"
                        + "/agent_model"
                        + str(data_dict["env_step"])
                        + "_"
                        + str(local_agent_location)
                        + ".json",
                        "w",
                    ) as f:
                        print("SAVE AGENT MODEL \n")
                        json.dump(data_dict['agent_model'][-1], f)
                    
                    with open(
                        Saving_path_result
                        + "/spy_model"
                        + "/spy_model"
                        + str(data_dict["env_step"])
                        + "_"
                        + str(local_agent_location)
                        + ".json",
                        "w",
                    ) as f:
                        print("SAVE SPY MODEL \n")
                        json.dump(data_dict['spy_model'][-1], f)
                    
                    if len(response_judge) == 0:
                        print("JUDGE NO RESPONSE: NO APPEND, HCA IS TEH LAST ONE")
                        continue

                    if local_agent_iter != 1:
                        print("RELAPSING RESPONSE IS LAST LOCAL RESPONSE")
                        relapse_response = data_local["last_agent_response"]

                    else:
                        print("RELAPSING RESPONSE IS CENTRAL RESPONSE")
                        relapse_response = cen_response

                    # -----------------------------------------SYNTACTIC CHECK FOR JUDGE-----------------------------------------#
                    data_dict["token_num_count_list"].append(token_num_count)

                    response, token_num_count_list_add = (
                        with_action_syntactic_check_func(
                            data_dict["pg_dict"],
                            response_judge,
                            [judge_prompt, relapse_response],
                            [response_judge],
                            model_name,
                            dialogue_history_method,
                            partial_judge_prompt_func,
                            state_update_prompt,
                            agent_action,
                            is_judge=True,
                        )
                    )
                    data_dict["token_num_count_list"] = (
                        data_dict["token_num_count_list"] + token_num_count_list_add
                    )

                    with open("conversation.txt", "a") as f:
                        messages = f"------###------###------HCA_JUDGE_{a}_ROW_{local_agent_row_i}_COL_{local_agent_column_j}------###------###------: \n {response_str_judge} \n \n"
                        f.write(messages)

                    print(f"JUDGE MODIFIED:\n {response}")

                else:
                    print(f"ORIGINAL PLAN:\n {response}")
                    pass

                data_local["last_agent_response"] = response
                data_dict["dialogue_history_list"].append(dialogue_history)
                
            # response come from HCA if no in judgement
            data_dict["response_total_list"].append(
                response
            )
            
            # -----------------------------------------EXECUTION OF ACTION AT EACH HCA AGENT LEVEL-----------------------------------------#
            print(
                "-------###-------###-------###-------EXECUTION-------###-------###-------###-------"
            )

            # problem here, a week of debug [index_query_time] uses outer most, always get one
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

                # render_graph_terminal_popup(
                #     data_dict["pg_dict"],
                #     pg_row_num=pg_row_num,
                #     pg_column_num=pg_column_num,
                # )
                # render_animate_terminal_popup(data_dict["pg_dict"], [original_response_dict])
                
                success_failure = "Success Update"

            except:
                success_failure = "Hallucination of wrong plan"
                pass

            # need to append new states to state list
            data_dict["pg_state_list"].append(data_dict["pg_dict"])

            # data_dict["agree_num"] = 0

            print(f'AGREEMENT NUMBER IS: {data_dict["agree_num"]}')

            # -----------------------------------------TASK SUCCESS CHECK-----------------------------------------#
            count = 0
            for ky, value in data_dict["pg_dict"].items():
                count += len(value)
                print(f"STILL HAVE {count} LEFT")
            if count == 0:
                # save final result
                with open(
                    Saving_path_result
                    + "/pg_state"
                    + "/pg_state"
                    + str(data_dict["env_step"] + 1)
                    + ".json",
                    "w",
                ) as f:
                    print("SAVE INITIAL STATE \n")
                    json.dump(data_dict["pg_dict"], f)
                break

    # -----------------------------------------TASK SUCCESS OUT-----------------------------------------#

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

    # with open(Saving_path_result + "/token_num_count.txt", "w") as f:
    #     print("SAVE TOKEN NUM \n")
    #     for token_num_num_count in token_num_count_list:
    #         f.write(str(token_num_num_count) + "\n")

    # with open(Saving_path_result + "/success_failure.txt", "w") as f:
    #     print("SAVE RESULT \n")
    #     f.write(success_failure)

    # with open(Saving_path_result + "/env_action_times.txt", "w") as f:
    #     print("SAVE ACTION TIME \n")
    #     f.write(f"{index_query_times+1}")
