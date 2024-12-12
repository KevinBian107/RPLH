import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.linear_model import LinearRegression
import numpy as np

def plot_env_progression(dist_a_spy, dist_s_spy, dist_s_nospy, get_trial=0):
    '''Plot teh environment progression per trial based on distance'''
    
    # All trials plot
    all_nospy = pd.DataFrame(dist_s_nospy['Distance'].explode())
    all_nospy[['Norm1', 'Norm2']] = pd.DataFrame(all_nospy['Distance'].tolist(), index=all_nospy.index)
    all_nospy.reset_index(inplace=True)

    all_spy = pd.DataFrame(dist_s_spy['Distance'].explode())
    all_spy[['Norm1', 'Norm2']] = pd.DataFrame(all_spy['Distance'].tolist(), index=all_spy.index)
    all_spy.reset_index(inplace=True)
    
    all_a_spy = pd.DataFrame(dist_a_spy['Distance'].explode())
    all_a_spy[['Norm1', 'Norm2']] = pd.DataFrame(all_a_spy['Distance'].tolist(), index=all_a_spy.index)
    all_a_spy.reset_index(inplace=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(all_nospy['index'], all_nospy['Norm1'], label='Norm1 - NoSpy', alpha=0.7)
    plt.plot(all_spy['index'], all_spy['Norm1'], label='Norm1 - Spy', alpha=0.7)
    plt.plot(all_a_spy['index'], all_a_spy['Norm1'], label='Norm1 - Agent Spy', alpha=0.7)
    plt.xlabel('Env Steps')
    plt.ylabel('Distance Norms')
    plt.title('Overall Plot for Distance Norms (Spy vs. NoSpy)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot by single trial
    df_nospy = pd.DataFrame(dist_s_nospy['Distance'].explode())
    df_nospy[['Norm1', 'Norm2']] = pd.DataFrame(df_nospy['Distance'].tolist(), index=df_nospy.index)
    df_nospy.reset_index(inplace=True)
    df_nospy_new = df_nospy.groupby('index').get_group(get_trial)
    df_nospy_new = df_nospy_new.reset_index(names='env_steps')
    df_nospy_new['env_steps'] = df_nospy_new['env_steps'] - min(df_nospy_new['env_steps'])

    df_spy = pd.DataFrame(dist_s_spy['Distance'].explode())
    df_spy[['Norm1', 'Norm2']] = pd.DataFrame(df_spy['Distance'].tolist(), index=df_spy.index)
    df_spy.reset_index(inplace=True)
    df_spy_new = df_spy.groupby('index').get_group(get_trial)
    df_spy_new = df_spy_new.reset_index(names='env_steps')
    df_spy_new['env_steps'] = df_spy_new['env_steps'] - min(df_spy_new['env_steps'])

    df_a_spy = pd.DataFrame(dist_a_spy['Distance'].explode())
    df_a_spy[['Norm1', 'Norm2']] = pd.DataFrame(df_a_spy['Distance'].tolist(), index=df_a_spy.index)
    df_a_spy.reset_index(inplace=True)
    df_a_spy_new = df_a_spy.groupby('index').get_group(get_trial)
    df_a_spy_new = df_a_spy_new.reset_index(names='env_steps')
    df_a_spy_new['env_steps'] = df_a_spy_new['env_steps'] - min(df_a_spy_new['env_steps'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_nospy_new['env_steps'], df_nospy_new['Norm1'], label='Norm1 - NoSpy', alpha=0.7)
    plt.plot(df_spy_new['env_steps'], df_spy_new['Norm1'], label='Norm1 - Spy', alpha=0.7)
    plt.plot(df_a_spy_new['env_steps'], df_a_spy_new['Norm1'], label='Norm1 - Agent Spy', alpha=0.7)
    plt.xlabel('Env Steps')
    plt.ylabel('Distance Norms')
    plt.title('Overlap Plot for Distance Norms 1 (Spy vs. NoSpy)')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(df_nospy_new['env_steps'], df_nospy_new['Norm2'], label='Norm2 - NoSpy', alpha=0.7)
    plt.plot(df_spy_new['env_steps'], df_spy_new['Norm2'], label='Norm2 - Spy', alpha=0.7)
    plt.plot(df_a_spy_new['env_steps'], df_a_spy_new['Norm2'], label='Norm2 - Agent Spy', alpha=0.7)
    plt.xlabel('Env Steps')
    plt.ylabel('Distance Norms')
    plt.title('Overlap Plot for Distance Norms 2 (Spy vs. NoSpy)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def dist_df_process(dist_df):
    '''process df for distance measurement'''
    
    df = pd.DataFrame(dist_df['Distance'].explode())
    df[['Norm1', 'Norm2']] = pd.DataFrame(df['Distance'].tolist(), index=df.index)
    df.reset_index(inplace=True, names="trial")
    df.reset_index(inplace=True, names="env_step")

    min_value = []
    grouped = df.groupby('trial').groups
    for key in (grouped).keys():
        min_value.append(grouped[key][0])
    
    df['env_step'] = df.apply(lambda x: x['env_step'] - min_value[x['trial']], axis=1)
    return df.drop(columns=['Distance'])

def calculate_auc_boot(df, num_samples):
    '''Calculate AUC'''
    
    auc_norm1 = []
    auc_norm2 = []

    for trial, group in df.groupby('trial'):
        group = group.sort_values(by='env_step')
        auc_norm1.append(auc(group['env_step'], group['Norm1']))
        auc_norm2.append(auc(group['env_step'], group['Norm2']))

    # Create a DataFrame with the calculated AUCs
    auc_df = pd.DataFrame({'norm1': auc_norm1, 'norm2': auc_norm2})

    bootstrap_mean = []
    for _ in range(num_samples):
        bootstrap_sample = auc_df.sample(replace=True, n=auc_df.shape[0])
        bootstrap_mean.append(bootstrap_sample.mean().to_dict())

    # Compute the mean of bootstrap samples
    mean_boot = pd.DataFrame(bootstrap_mean).mean()

    # Log or return results
    print(f"Overall AUC for Norm1: {mean_boot['norm1']}")
    print(f"Overall AUC for Norm2: {mean_boot['norm2']}")
    
    return mean_boot


def calculate_auc(df):
    '''Calculate AUC'''
    
    auc_norm1 = 0
    auc_norm2 = 0

    for trial, group in df.groupby('trial'):
        group = group.sort_values(by='env_step')
        
        auc_norm1 += auc(group['env_step'], group['Norm1'])
        auc_norm2 += auc(group['env_step'], group['Norm2'])

    # Average AUC across all trials
    auc_norm1 /= df['trial'].nunique()
    auc_norm2 /= df['trial'].nunique()

    print(f"Overall AUC for Norm1: {auc_norm1}")
    print(f"Overall AUC for Norm2: {auc_norm2}")
    
    return auc_norm1, auc_norm2

def calculate_slope(df):
    '''Calculate average slope with linear regression'''
    
    slopes_norm1 = []
    slopes_norm2 = []

    for trial, group in df.groupby('trial'):
        X = group['env_step'].values.reshape(-1, 1)  # Predictor (env_step)
        y_norm1 = group['Norm1'].values  # Response (Norm1)
        y_norm2 = group['Norm2'].values  # Response (Norm2)

        # Fit linear regression for Norm1
        model_norm1 = LinearRegression().fit(X, y_norm1)
        slopes_norm1.append(model_norm1.coef_[0])

        # Fit linear regression for Norm2
        model_norm2 = LinearRegression().fit(X, y_norm2)
        slopes_norm2.append(model_norm2.coef_[0])

    avg_slope_norm1 = np.mean(slopes_norm1)
    avg_slope_norm2 = np.mean(slopes_norm2)

    print(f"Average slope for Norm1: {avg_slope_norm1}")
    print(f"Average slope for Norm2: {avg_slope_norm2}")
    
    return avg_slope_norm1, avg_slope_norm2
    