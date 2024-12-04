import time
import json
import os
import re


def load_experiment_results(Saving_path_result):
    """
    Loads experiment results from the specified saving path.

    Args:
        Saving_path_result (str): The directory where results are stored.

    Returns:
        tuple: Loaded user_prompt_list, response_total_list, pg_state_list, dialogue_history_list, and hca_response_list.
    """
    user_prompt_list = []
    response_total_list = []
    pg_state_list = []
    dialogue_history_list = []
    hca_response_list = []

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

    # Load hca response history
    hca_agent_response_dir = os.path.join(Saving_path_result, "hca_agent_response")
    if os.path.exists(hca_agent_response_dir):
        for file_name in sorted(os.listdir(hca_agent_response_dir)):
            with open(os.path.join(hca_agent_response_dir, file_name), "r") as f:
                hca_response_list.append(json.load(f))

    return (
        user_prompt_list,
        response_total_list,
        pg_state_list,
        dialogue_history_list,
        hca_response_list,
    )


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


def format_dialogue_history(dialogue_history):
    """
    Formats dialogue history to have clear titles for each agent's contributions.

    Args:
        dialogue_history (list): List of dialogue strings.

    Returns:
        str: Formatted dialogue history.
    """
    formatted_history = ""
    for dialogue in dialogue_history:
        # Extract the agent identifier and its conversation
        agent_match = re.match(r"(Agent\[[^\]]+\]):", dialogue)
        if agent_match:
            agent_title = agent_match.group(1)
            conversation = dialogue[
                len(agent_title) + 1 :
            ].strip()  # Remove the agent title from dialogue
            formatted_history += f"\n--- {agent_title} ---\n{conversation}\n"
        else:
            # Add as-is if no specific agent title is found
            formatted_history += f"\n{dialogue}\n"
    return formatted_history


def parse_response(response):
    """
    Parses the response into structured sections.

    Args:
        response (str or list): The response to parse. Can be a single string or a list of strings.

    Returns:
        dict: Parsed sections as a dictionary.
    """
    sections = {}

    # Ensure response is iterable (e.g., handle list of strings)
    if isinstance(response, str):
        response_lines = response.splitlines()
    elif isinstance(response, list):
        response_lines = [line for item in response for line in item.splitlines()]
    else:
        raise ValueError("Response must be a string or a list of strings")

    for line in response_lines:
        if ": " in line:
            key, value = line.split(": ", 1)
            if key in sections:
                sections[key].append(value)
            else:
                sections[key] = [value]

    return sections


def print_response_sections(sections, token_delay=0.01):
    """
    Prints the parsed sections of a response with clear formatting.

    Args:
        sections (dict): Parsed sections of a response.
        token_delay (float): Delay between tokens when printing.
    """
    for key, values in sections.items():
        print(f"--- {key.capitalize()} ---")
        for value in values:
            if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
                # Handle list-like strings
                items = eval(value)  # Convert the string representation of a list into an actual list
                for item in items:
                    print_token_by_token(item.strip(), token_delay)
                    print()  # Blank line between list items
            else:
                print_token_by_token(value.strip(), token_delay)
                print()  # Add a blank line between entries in the same section


def roll_out_conversation(user_prompts, responses, pg_states, dialogue_histories, hca_responses, delay=1.0, token_delay=0.01):
    """
    Simulates a conversation rollout with dynamic dialogue visualization.

    Args:
        user_prompts (list): List of user prompts.
        responses (list): List of agent responses.
        pg_states (list): List of environment states.
        dialogue_histories (list): Dialogue history at each step.
        hca_responses (list): HCA agent response at each step.
        delay (float): Delay between each step (in seconds).
        token_delay (float): Delay between tokens (in seconds).
    """
    print("=== Conversation Rollout Start ===\n")

    for step in range(len(user_prompts)):
        print(f"Step {step + 1}")
        print("=" * 40)  # Separator for steps

        # Print HCA Agent Action
        if step < len(hca_responses):
            print("HCA Agent Action:")
            hca_response = hca_responses[step]
            sections = parse_response(hca_response)
            print_response_sections(sections, token_delay)

        # Print Dialogue History
        print("Dialogue History:")
        if step < len(dialogue_histories):
            formatted_history = format_dialogue_history(dialogue_histories[step])
            print_token_by_token(formatted_history.strip(), token_delay)

        print("\n")  # Add spacing between steps
        time.sleep(delay)

    # Print Final Environment State
    print("=== Final Environment State ===")
    print_token_by_token(json.dumps(pg_states[-1], indent=2), token_delay)
    print("\n=== Conversation Rollout End ===")


# Load experiment data
Saving_path_result = "demos/converging_samples/rplh_efficient_1/"

(
    user_prompt_list,
    response_total_list,
    pg_state_list,
    dialogue_history_list,
    hca_response_list,
) = load_experiment_results(Saving_path_result)

# Rollout the conversation with dialogue history visualization
roll_out_conversation(
    user_prompt_list,
    response_total_list,
    pg_state_list,
    dialogue_history_list,
    hca_response_list,
)
