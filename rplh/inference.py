import argparse
import importlib
import os
import sys

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Central inference function for multi-agent experiments.")
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

    # Set up paths and parameters
    path = "multi-agent-env"
    saving_path = os.path.join(Code_dir_path, path)
    os.makedirs(saving_path, exist_ok=True)
    
    print(f"-------------------CREATING ENVIRONMENT IN {args.module_name}-------------------")
    
    env_module.create_env1(saving_path, repeat_num=1, box_num_upper_bound=1, box_num_low_bound = 1)
    
    print(f"-------------------Module: {args.module_name} | Model name: {args.model_name}-------------------")
    
    if (args.module_name == "h_efficient") or (args.module_name == "d_efficient"):
        dialogue_history_method = "_w_markovian_state_action_history"
    else:
        dialogue_history_method = "_w_no_history"
     
    # Experiment parameters
    pg_row_num = 2
    pg_column_num = 2
    iteration_num = 0
    query_time_limit = 10

    # Call the run_exp function from the specified module
    result = inference_module.run_exp(
        saving_path,
        pg_row_num,
        pg_column_num,
        iteration_num,
        query_time_limit,
        dialogue_history_method=dialogue_history_method,
        model_name=args.model_name,
    )
    
    print("Experiment completed successfully.")
    return result

if __name__ == "__main__":
    main()
