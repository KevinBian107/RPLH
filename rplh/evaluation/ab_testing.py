import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm


def bootstrap_proportions_t_technique(group1, group2, alternative='greater', num_samples=10000, population_mean=0):
    """
    Perform a bootstrap test using the bootstrap-t technique for the mean difference in proportions. The difference is group2 - group1
    
    Parameters:
        group1 (DataFrame): DataFrame for group 1 with 'Trial' and 'Num_Responses'.
        group2 (DataFrame): DataFrame for group 2 with 'Trial' and 'Num_Responses'.
        num_samples (int): Number of bootstrap samples to generate.
        alternative (str): Type of test ('greater', 'less', 'two-sided').
        population_mean (float): Null hypothesis value (typically 0).
    
    Returns:
        t_statistics (list): List of bootstrap t-statistics.
        obs_diff (float): Observed mean difference in proportions.
        p_value (float): P-value for the test.
    """
    # Calculate the differences in proportions for trials present in both groups
    feature = group1.columns[1]
    observed_diffs = [
        group2.query(f'Trial == "{trial}"')[feature].iloc[0] -
        group1.query(f'Trial == "{trial}"')[feature].iloc[0]
        for trial in group1['Trial'].unique()
        if trial in group2['Trial'].unique()
    ]
    
    # Calculate the observed mean difference and standard error
    obs_diff = np.mean(observed_diffs)
    obs_std = np.std(observed_diffs, ddof=1) / np.sqrt(len(observed_diffs))  # Standard error of the observed data
    observed_t = (obs_diff - population_mean) / obs_std
    # Generate bootstrap t-statistics
    t_statistics = []
    for _ in range(num_samples):
        bootstrap_sample = np.random.choice(observed_diffs, size=len(observed_diffs), replace=True)
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_std = np.std(bootstrap_sample, ddof=1) / np.sqrt(len(bootstrap_sample))  # Bootstrap standard error
        t_stat = (bootstrap_mean - obs_diff) / bootstrap_std  # Bootstrap t-statistic
        t_statistics.append(t_stat)
    
    # Convert to numpy array for analysis
    t_statistics = np.array(t_statistics)
    
    # Calculate the p-value based on the observed t-statistic
    
    if alternative == 'two-sided':
        p_value = 2 * np.mean(np.abs(t_statistics) >= np.abs(observed_t))  # Two-tailed p-value
    elif alternative == 'greater':
        p_value = np.mean(t_statistics >= observed_t)  # One-tailed p-value (greater)
    elif alternative == 'less':
        p_value = np.mean(t_statistics <= observed_t)  # One-tailed p-value (less)
    else:
        raise ValueError("Invalid alternative hypothesis. Choose 'two-sided', 'greater', or 'less'.")
    
    return t_statistics, obs_diff, p_value, observed_t


from scipy.stats import ttest_1samp
def t_test_proportions_one_tailed(group1, group2):
    """
    Perform a one-tailed t-test for the mean difference in proportions between two groups.
    
    Null Hypothesis (H0): The mean difference in proportions between the two groups is greater than 0.
    Alternative Hypothesis (H1): The mean difference in proportions between the two groups is less than or equal to 0.
    Make sure group2 is expected greater than group1
    
    Parameters:
        group1 (DataFrame): DataFrame for group 1 with 'Trial' and 'Num_Responses'.
        group2 (DataFrame): DataFrame for group 2 with 'Trial' and 'Num_Responses'.
    
    Returns:
        obs_diff (float): Observed mean difference in proportions.
        t_stat (float): T-statistic for the test.
        p_value (float): One-tailed p-value for the test.
    """
    # Calculate the differences in proportions for trials present in both groups
    observed_diffs = [
        group2.query(f'Trial == "{trial}"')['Num_Responses'].iloc[0] - 
        group1.query(f'Trial == "{trial}"')['Num_Responses'].iloc[0]
        for trial in group1['Trial'].unique()
        if trial in group2['Trial'].unique()
    ]
    
    # Calculate the observed mean difference
    obs_diff = np.mean(observed_diffs)
    
    # Perform a one-sample t-test
    t_stat, p_two_tailed = ttest_1samp(observed_diffs, popmean=0)
    
    # Convert two-tailed p-value to one-tailed p-value (for H0: mean > 0)
    p_value = p_two_tailed / 2 if t_stat > 0 else 1.0  # Only consider the positive tail
    
    return obs_diff, t_stat, p_value, observed_diffs

