import sys
import os
from pathlib import Path

main_path = Path(__file__).resolve().parent.parent
if str(main_path) not in sys.path:
    sys.path.append(str(main_path))

import argparse
import importlib
from rplh.env.env import create_env1


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Testing function for multi-agent experiments."
    )
    parser.add_argument(
        "--module_name",
        type=str,
        required=True,
        help="Module name for the inference (e.g., d_efficient or h_vanilla).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name to be used in the experiment.",
    )
    parser.add_argument(
        "--num_trials", type=int, required=True, help="Number of trials to run."
    )
    parser.add_argument(
        "--box_num_upper_bound",
        type=int,
        required=True,
        help="Upper bound for the number of boxes.",
    )
    parser.add_argument(
        "--start_iter", type=int, required=True, help="Start iterations in testing."
    )
    parser.add_argument(
        "--row_num",
        type=int,
        required=True,
        help="Rows in environment.",
    )
    parser.add_argument(
        "--column_num",
        type=int,
        required=True,
        help="Columns in environment.",
    )
    parser.add_argument(
        "--reasoning_model",
        type=str,
        required=True,
        help="Reasoning model (agent or standard)",
    )

    args = parser.parse_args()

    # Dynamically import the specified module
    try:
        if args.reasoning_model == "agent":
            inference_loop = "systems." + args.module_name + ".rplh_agent_inference"
        else:
            inference_loop = "systems." + args.module_name + ".rplh_inference"

        inference_module = importlib.import_module(inference_loop)
    except ModuleNotFoundError:
        print(f"Error: Module '{args.module_name}' not found.")
        sys.exit(1)

    # Validate that the run_exp function exists
    if not hasattr(inference_module, "run_exp"):
        print(
            f"Error: Module '{args.module_name}' does not contain a 'run_exp' function."
        )
        sys.exit(1)

    # Get the parent directory of the current script (assumes RPLH is one level above)
    Code_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Set up the base saving path
    base_path = os.path.join(Code_dir_path, f"testing-env-agnt")
    os.makedirs(base_path, exist_ok=True)

    print(
        f"-------------------CREATING ENVIRONMENT IN {args.module_name}-------------------"
    )

    for trial in range(args.start_iter, args.num_trials + 1):
        # start with iter, should be 1 to run full thing
        trial_path = os.path.join(base_path, f"trial_{trial}")
        os.makedirs(trial_path, exist_ok=True)

        print(f"Trial {trial}: Saving results in {trial_path}")

        seed = trial  # seed from trial

        create_env1(
            trial_path,
            repeat_num=1,
            box_num_upper_bound=args.box_num_upper_bound,
            box_num_low_bound=1,
            pg_row_num=args.row_num,
            pg_column_num=args.column_num,
            seed=seed,
        )

        print(
            f"-------------------Trial {trial} | Module: {args.module_name} | Model name: {args.model_name}-------------------"
        )

        if (args.module_name == "h_efficient") or (args.module_name == "d_efficient"):
            dialogue_history_method = (
                "_w_no_history"  # "_w_markovian_state_action_history"
            )
        else:
            dialogue_history_method = "_w_only_state_action_history"

        # Experiment parameters
        iteration_num = 0
        query_time_limit = 50

        # Call the run_exp function from the specified module
        result = inference_module.run_exp(
            trial_path,
            args.row_num,
            args.column_num,
            iteration_num,
            query_time_limit,
            dialogue_history_method=dialogue_history_method,
            model_name=args.model_name,
        )

        print(f"Trial {trial} completed successfully.")

    print("All trials completed successfully.")


if __name__ == "__main__":
    main()
