import pandas as pd 
from rplh.evaluation.embed import get_spy_detect_embedding_only
from rplh.evaluation.energy import dist_df_process, calculate_auc, calculate_slope

def get_all_evals(df_main, df_success, df_spy, df_justification, df_dist, system="rplh-agent-spy"):
    '''Take in 4 df for the same system:
    
    1. Distance df, including all information of convergence trials norm1 and norm2 data.
    2. Main df, include all standard data.
    3. Success df, including whether converge or not
    4. Spy df, including all spy model info, only agent reasoning has.
    5. Justification df, include all justifications.
    
    3,4,5 should give -> Embedding df, including all standard data + embedding similarity for each agent compared to spy sentence (N.A. for standard).
    
    Out:
    - One column with name of system
    - Rows include
        - Average-AUC,
        - Average-Slope,
        - Average-Embedding-Similarity for each spy agent,
        - Average-Justifcation-Similarity
        - Average-Box-To-Targets,
        - Average-Responses,
        - Average-Convergence-Rate (If converge)
    '''
    # merge main
    main_merged = df_main.merge(df_success, on="Trial")
    avg_standard_data = pd.DataFrame(main_merged.drop(columns=["Trial", "Convergence"]).mean()).T
    
    # auc & slope
    dist_merged = df_dist.merge(df_success, on="Trial")
    dist_converged = dist_merged[dist_merged["Convergence"] == "Converged"].drop(columns=["Trial"]).reset_index(drop=True)
    dist_processed = dist_df_process(dist_converged)
    auc_norm1, auc_norm2 = calculate_auc(dist_processed)
    avg_slope1, avg_slope2 = calculate_slope(dist_processed)
    
    # embedding
    spy_embedding = get_spy_detect_embedding_only(df_spy, df_justification,
                                         spy_sentence="Seems to be the spy agent, its goal is to prevent match targets.",
                                         just_sentence="I suspect that this agent is the spy agent, thus, I would not listen to this agent.",
                                         spy_agents=["Agent[0.5, 0.5]", "Agent[1.5, 1.5]", "Agent[2.5, 2.5]"],
                                         only_spy=True)
    avg_embedding = pd.DataFrame(spy_embedding.mean()).T
    
    # Dynamically collect embedding columns
    embedding_columns = avg_embedding.columns
    avg_embedding_metrics = {f"Average-Embedding-Similarity-{col}": avg_embedding[col].values[0] for col in embedding_columns}

    metrics = {
        "Average-AUC-Norm1": auc_norm1,
        "Average-AUC-Norm2": auc_norm2,
        "Average-Slope1": avg_slope1,
        "Average-Slope2": avg_slope2,
        "Average-Box-To-Targets-Per-Response": avg_standard_data["Avg_Boxes_To_Targets_Per_Response"].values[0],
        "Average-Responses": avg_standard_data["Num_Responses"].values[0],
        "Average-Convergence-Rate": (main_merged["Convergence"] == "Converged").mean(),
        **avg_embedding_metrics,
    }

    result_df = pd.DataFrame([metrics], index=[system]).T
    return result_df