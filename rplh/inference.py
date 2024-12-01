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
        module = args.module_name + ".rplh_inference"
        inference_module = importlib.import_module(module)
    except ModuleNotFoundError:
        print(f"Error: Module '{args.module_name}' not found.")
        sys.exit(1)
    
    # Validate that the run_exp function exists
    if not hasattr(inference_module, "run_exp"):
        print(f"Error: Module '{args.module_name}' does not contain a 'run_exp' function.")
        sys.exit(1)
    
    # Set up paths and parameters
    Code_dir_path = os.getcwd()
    saving_path = os.path.join(Code_dir_path, "multi-agent-env")
    os.makedirs(saving_path, exist_ok=True)

    # Experiment parameters
    pg_row_num = 2
    pg_column_num = 2
    iteration_num = 0
    query_time_limit = 10

    print(f"-------------------Module: {args.module_name} | Model name: {args.model_name}-------------------")

    # Call the run_exp function from the specified module
    result = inference_module.run_exp(
        saving_path,
        pg_row_num,
        pg_column_num,
        iteration_num,
        query_time_limit,
        dialogue_history_method="_w_markovian_state_action_history",
        model_name=args.model_name,
    )
    
    print("Experiment completed successfully.")
    return result

if __name__ == "__main__":
    main()
