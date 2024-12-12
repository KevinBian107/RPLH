import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
from sklearn.model_selection import train_test_split

from rplh.evaluation.get_data import get_data
from rplh.evaluation.embed import get_spy_detect_embedding_only, get_spy_detect_embedding_for_feature
from rplh.evaluation.energy import dist_df_process, calculate_auc, calculate_slope, calculate_auc_boot
from rplh.evaluation.ab_testing import *


def get_all_boots_evals(df_main, df_success, df_dist, num_samples = 10000,system="rplh-agent-spy"):
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
    bootstrap_mean = []
    standard_data = main_merged.drop(columns=["Trial"])
    # avg_standard_data = pd.DataFrame(standard_data.mean()).T
    
    for _ in range(num_samples):

        bootstrap_sample = standard_data.sample(replace = True, n = main_merged.shape[0])
        bootstrap_mean.append(dict(bootstrap_sample.drop(columns = ['Convergence']).mean()) |
                              {'Convergence' : 
                               bootstrap_sample.query("Convergence == 'Converged'").shape[0] / 
                               bootstrap_sample.shape[0]})

    mean_boot_standard_mean = pd.DataFrame(bootstrap_mean).mean()
    
    
    # auc & slope
    dist_merged = df_dist.merge(df_success, on="Trial")
    dist_converged = dist_merged[dist_merged["Convergence"] == "Converged"].drop(columns=["Trial"]).reset_index(drop=True)
    dist_processed = dist_df_process(dist_converged)
    auc_boot = calculate_auc_boot(dist_processed, num_samples=num_samples)
    auc_norm1 = auc_boot['norm1']
    auc_norm2 = auc_boot['norm2']
    
    
    metrics = {
        "Average-AUC-Norm1": auc_norm1,
        "Average-AUC-Norm2": auc_norm2,
        "Average-avg-Box-To-Targets-Per-Response": mean_boot_standard_mean["Avg_Boxes_To_Targets_Per_Response"],
        "Average-Responses": mean_boot_standard_mean["Num_Responses"],
        "Average-Convergence-Rate": mean_boot_standard_mean['Convergence'],
    }

    result_df = pd.DataFrame([metrics], index=[system]).T
    return result_df

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

def eval_all_feature_importance():
    '''get all feature importance for all data'''
    
    data_list = [ "3up/gpt-agent-testing-env-3up",
                 "3up/gpt-standard-testing-env-3up",
                 "3up/gpt-standard-testing-nospy-env-3up",
                 "2up/gpt-agent-testing-env-2up",
                 "2up/gpt-standard-testing-env-2up",
                 "2up/gpt-standard-testing-nospy-env-2up"]
    
    full_embed_df = pd.DataFrame()
    
    current_folder = Path.cwd()
    parent_folder = current_folder.parent.parent
    print(parent_folder)
    
    for d in data_list:
        print(f"LOADING {d} \n")
        base_dir = parent_folder / "data" / d
        df, success_df, spy_count_df, spy_df, att_df, justification_df, dist_df = get_data(base_dir, 20)
        merged = df.merge(success_df, on="Trial")
    
        embedding = get_spy_detect_embedding_for_feature(merged, spy_df, justification_df,
                                                spy_sentence="Seems to be the spy agent, its goal is to prevent match targets.",
                                                just_sentence="I suspect that this agent is the spy agent, thus, I would not listen to this agent.",
                                                spy_agents=["Agent[0.5, 0.5]", "Agent[1.5, 1.5]", "Agent[2.5, 2.5]"],
                                                only_spy=False).drop(columns=["Justification_Embed"])

        embedding["Have_spy"] = 0 if "nospy" in d else 1
        full_embed_df = pd.concat([full_embed_df, embedding], axis=0, ignore_index=True).fillna(0)
    
    # train on three different types
    X = full_embed_df.drop(columns=["Avg_Boxes_To_Targets_Per_Response", "Num_Responses", "Num_Boxes"])
    y1 = full_embed_df["Avg_Boxes_To_Targets_Per_Response"]
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y1, test_size=0.2, random_state=42)
    model1 = RandomForestRegressor(random_state=42)
    model1.fit(X_train1, y_train1)

    y2 = full_embed_df["Num_Responses"]
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2, test_size=0.2, random_state=42)
    model2 = RandomForestRegressor(random_state=42)
    model2.fit(X_train2, y_train2)
    
    y3 = full_embed_df[["Num_Responses", "Avg_Boxes_To_Targets_Per_Response"]]
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y3, test_size=0.2, random_state=42)
    model3 = RandomForestRegressor(random_state=42)
    model3.fit(X_train3, y_train3)

    importances1 = pd.Series(model1.feature_importances_, index=X.columns).sort_values()
    importances2 = pd.Series(model2.feature_importances_, index=X.columns).sort_values()
    importances3 = pd.Series(model3.feature_importances_, index=X.columns).sort_values()
    
    importances3.plot(kind="barh", title="Feature Importance for Vectorized Targets (All Data)")

    plt.figure(figsize=(10, 8))
    y = range(len(importances1))

    plt.barh(y, importances1.sort_index(), height=0.4, label="Avg_Boxes_To_Targets_Per_Response", alpha=0.7)
    plt.barh([i + 0.4 for i in y], importances2.sort_index(), height=0.4, label="Num_Responses", alpha=0.7)
    plt.yticks([i + 0.2 for i in y], importances1.sort_index().index)
    plt.title("Comparison of Feature Importance Based on Different Target Variables (All Data)")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # explained_variance_model1 = model1.score(X_test1, y_test1)
    # explained_variance_model2 = model2.score(X_test2, y_test2)
    # print(f"Explained Variance (R^2) for Avg_Boxes_To_Targets_Per_Response: {explained_variance_model1:.3f}")
    # print(f"Explained Variance (R^2) for Num_Responses: {explained_variance_model2:.3f}")
    
    plt.show()
    
    # H Test
    n_bootstrap = 15000
    group_with_spy = full_embed_df[full_embed_df["Have_spy"] == 1]["Num_Responses"]
    group_without_spy = full_embed_df[full_embed_df["Have_spy"] == 0]["Num_Responses"]
    observed_diff = group_with_spy.mean() - group_without_spy.mean()
    combined_data = np.concatenate([group_with_spy, group_without_spy])

    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        resampled_group_with_spy = np.random.choice(combined_data, size=len(group_with_spy), replace=True)
        resampled_group_without_spy = np.random.choice(combined_data, size=len(group_without_spy), replace=True)
        bootstrap_diffs.append(resampled_group_with_spy.mean() - resampled_group_without_spy.mean())

    bootstrap_diffs = np.array(bootstrap_diffs)
    p_bootstrap = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
    print(f"Observed difference in means: {observed_diff:.3f}")
    print(f"Bootstrap p-value: {p_bootstrap:.3f}")

    alpha = 0.1
    if p_bootstrap < alpha:
        print("Reject the null hypothesis: Having a spy significantly slows down convergence.")
    else:
        print("Fail to reject the null hypothesis: No evidence that having a spy affects convergence.")

    plt.figure(figsize=(10, 6))
    plt.hist(bootstrap_diffs, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(observed_diff, color='red', linestyle='--', label=f'Observed Diff: {observed_diff:.3f}')
    plt.axvline(-observed_diff, color='red', linestyle='--')
    plt.title('Bootstrap Sampling Distribution of Mean Difference')
    plt.xlabel('Difference in Means')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()