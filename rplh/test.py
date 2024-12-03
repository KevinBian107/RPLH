import argparse
import importlib
import os
import sys

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Testing function for multi-agent experiments.")
    parser.add_argument(
        "--module_name", 
        type=str, 
        required=True, 
        help="Module name for the inference (e.g., d_efficient or h_vanilla)."
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True, 
        help="Model name to be used in the experiment."
    )
    parser.add_argument(
        "--num_trials", 
        type=int, 
        required=True, 
        help="Number of trials to run."
    )
    parser.add_argument(
        "--box_num_upper_bound", 
        type=int, 
        required=True, 
        help="Upper bound for the number of boxes."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        required=True, 
        help="Random seed to create environment."
    )

    args = parser.parse_args()

    # Dynamically import the specified module
    try:
        # Append ".rplh_inference" to the module name
        inference_loop = args.module_name + ".rplh_inference"
        env_create = args.module_name + ".env"
        inference_module = importlib.import_module(inference_loop)
        env_module = importlib.import_module(env_create)
    except ModuleNotFoundError:
        print(f"Error: Module '{args.module_name}' not found.")
        sys.exit(1)

    # Validate that the run_exp function exists
    if not hasattr(inference_module, "run_exp"):
        print(f"Error: Module '{args.module_name}' does not contain a 'run_exp' function.")
        sys.exit(1)

    # Get the parent directory of the current script (assumes RPLH is one level above)
    Code_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Set up the base saving path
    base_path = os.path.join(Code_dir_path, "multi-agent-env-tests")
    os.makedirs(base_path, exist_ok=True)

    print(f"-------------------CREATING ENVIRONMENT IN {args.module_name}-------------------")

    for trial in range(1, args.num_trials + 1):
        trial_path = os.path.join(base_path, f"trial_{trial}")
        os.makedirs(trial_path, exist_ok=True)

        print(f"Trial {trial}: Saving results in {trial_path}")

        env_module.create_env1(trial_path, repeat_num=1, box_num_upper_bound=args.box_num_upper_bound, box_num_low_bound=1, seed=args.seed)

        print(f"-------------------Trial {trial} | Module: {args.module_name} | Model name: {args.model_name}-------------------")

        if (args.module_name == "h_efficient") or (args.module_name == "d_efficient"):
            dialogue_history_method = "_w_markovian_state_action_history"
        else:
            dialogue_history_method = "_w_only_state_action_history"

        # Experiment parameters
        pg_row_num = 2
        pg_column_num = 2
        iteration_num = 0
        query_time_limit = 20

        # Call the run_exp function from the specified module
        result = inference_module.run_exp(
            trial_path,
            pg_row_num,
            pg_column_num,
            iteration_num,
            query_time_limit,
            dialogue_history_method=dialogue_history_method,
            model_name=args.model_name,
        )

        print(f"Trial {trial} completed successfully.")

    print("All trials completed successfully.")

if __name__ == "__main__":
    main()
