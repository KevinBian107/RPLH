import time
import json
import os


def load_experiment_results(Saving_path_result):
    """
    Loads experiment results from the specified saving path.

    Args:
        Saving_path_result (str): The directory where results are stored.

    Returns:
        tuple: Loaded user_prompt_list, response_total_list, pg_state_list, and dialogue_history_list.
    """
    user_prompt_list = []
    response_total_list = []
    pg_state_list = []
    dialogue_history_list = []

    # Load user prompts
    prompt_dir = os.path.join(Saving_path_result, "prompt")
    if os.path.exists(prompt_dir):
        for file_name in sorted(os.listdir(prompt_dir)):
            with open(os.path.join(prompt_dir, file_name), "r") as f:
                user_prompt_list.append(f.read())

    # Load responses
    response_dir = os.path.join(Saving_path_result, "response")
    if os.path.exists(response_dir):
        for file_name in sorted(os.listdir(response_dir)):
            with open(os.path.join(response_dir, file_name), "r") as f:
                response_total_list.append(json.load(f))

    # Load pg states
    pg_state_dir = os.path.join(Saving_path_result, "pg_state")
    if os.path.exists(pg_state_dir):
        for file_name in sorted(os.listdir(pg_state_dir)):
            with open(os.path.join(pg_state_dir, file_name), "r") as f:
                pg_state_list.append(json.load(f))

    # Load dialogue history
    dialogue_history_dir = os.path.join(Saving_path_result, "dialogue_history")
    if os.path.exists(dialogue_history_dir):
        for file_name in sorted(os.listdir(dialogue_history_dir)):
            with open(os.path.join(dialogue_history_dir, file_name), "r") as f:
                dialogue_history_list.append(json.load(f))

    return user_prompt_list, response_total_list, pg_state_list, dialogue_history_list


def print_token_by_token(text, delay=0.05):
    """
    Prints text token by token to simulate LLM-like emerging output.

    Args:
        text (str): The text to print token by token.
        delay (float): Delay between each token (in seconds).
    """
    for token in text:
        print(token, end="", flush=True)
        time.sleep(delay)
    print()  # Newline after the text is complete


def roll_out_conversation(
    user_prompt_list,
    response_total_list,
    pg_state_list,
    dialogue_history_list,
    delay=1.0,
    token_delay=0.05,
):
    """
    Simulates the conversation dynamically in the terminal with dialogue history.

    Args:
        user_prompt_list (list): List of user prompts.
        response_total_list (list): List of responses from the agents.
        pg_state_list (list): List of environment states at each step.
        dialogue_history_list (list): List of dialogue history at each step.
        delay (float): Time delay (in seconds) between each step for better readability.
        token_delay (float): Delay between each token for simulating LLM-like output.
    """
    print("=== Conversation Rollout Start ===\n")

    for step in range(len(user_prompt_list)):
        print(f"--- Step {step + 1} ---\n")

        print("Environment State:")
        print_token_by_token(json.dumps(pg_state_list[step], indent=2), token_delay)

        print("\nUser Prompt:")
        print_token_by_token(user_prompt_list[step], token_delay)

        time.sleep(delay)  # Delay to simulate dynamic interaction

        print("\nAgent Response:")
        print_token_by_token(
            json.dumps(response_total_list[step], indent=2), token_delay
        )

        print("\nDialogue History:")
        if step < len(dialogue_history_list):
            print_token_by_token(
                json.dumps(dialogue_history_list[step], indent=2), token_delay
            )

        print("\n")
        time.sleep(delay)  # Delay to allow readability between steps

    # Display the final state
    print("=== Final Environment State ===\n")
    print_token_by_token(json.dumps(pg_state_list[-1], indent=2), token_delay)
    print("=== Conversation Rollout End ===\n")


# Load experiment data
Saving_path_result = "multi-agent-env/env_pg_state_2_2/pg_state2/_w_all_dialogue_history_qwen2.5:14b-instruct-q3_K_L"

user_prompt_list, response_total_list, pg_state_list, dialogue_history_list = (
    load_experiment_results(Saving_path_result)
)

# Rollout the conversation with dialogue history visualization
roll_out_conversation(
    user_prompt_list, response_total_list, pg_state_list, dialogue_history_list
)
