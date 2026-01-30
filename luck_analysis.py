"""
Luck factor analysis for sports prediction.

This module implements methods to quantify the role of luck in sports outcomes,
as described in "Luck is Hard to Beat: The Difficulty of Sports Prediction" 
by Aoki et al. (2017).
"""

import numpy as np
from typing import List, Tuple, Optional
from bradley_terry import BradleyTerryModel


def simulate_season(model: BradleyTerryModel, schedule: List[Tuple[int, int]], 
                    n_simulations: int = 1000) -> np.ndarray:
    """
    Simulate multiple seasons using the Bradley-Terry model.
    
    This helps quantify uncertainty and the role of luck by showing the distribution
    of possible outcomes given the same team strengths.
    
    Args:
        model: Fitted Bradley-Terry model
        schedule: List of (team_i, team_j) matchups
        n_simulations: Number of simulations to run
        
    Returns:
        Array of shape (n_simulations, n_games) with simulated outcomes
    """
    n_games = len(schedule)
    results = np.zeros((n_simulations, n_games))
    
    for sim in range(n_simulations):
        for game_idx, (team_i, team_j) in enumerate(schedule):
            prob_i_wins = model.predict_probability(team_i, team_j)
            results[sim, game_idx] = np.random.binomial(1, prob_i_wins)
    
    return results


def calculate_win_distribution(model: BradleyTerryModel, schedule: List[Tuple[int, int]],
                               n_simulations: int = 1000) -> np.ndarray:
    """
    Calculate the distribution of wins for each team across simulated seasons.
    
    Args:
        model: Fitted Bradley-Terry model
        schedule: List of (team_i, team_j) matchups
        n_simulations: Number of simulations
        
    Returns:
        Array of shape (n_simulations, n_teams) with win counts for each simulation
    """
    simulated_results = simulate_season(model, schedule, n_simulations)
    n_teams = model.n_teams
    win_counts = np.zeros((n_simulations, n_teams))
    
    for sim in range(n_simulations):
        for game_idx, (team_i, team_j) in enumerate(schedule):
            if simulated_results[sim, game_idx] == 1:
                win_counts[sim, team_i] += 1
            else:
                win_counts[sim, team_j] += 1
    
    return win_counts


def calculate_luck_factor(actual_wins: np.ndarray, expected_wins: np.ndarray,
                         win_distribution: np.ndarray) -> np.ndarray:
    """
    Calculate a luck factor for each team.
    
    The luck factor measures how much a team's actual performance deviates from
    expectation, normalized by the natural variation in outcomes.
    
    Args:
        actual_wins: Actual number of wins for each team
        expected_wins: Expected number of wins based on model
        win_distribution: Distribution of wins from simulations (n_sims x n_teams)
        
    Returns:
        Luck factor for each team (positive = lucky, negative = unlucky)
    """
    # Standard deviation of wins from simulations
    std_wins = np.std(win_distribution, axis=0)
    
    # Avoid division by zero
    std_wins[std_wins == 0] = 1.0
    
    # Z-score: how many standard deviations from expected
    luck_factor = (actual_wins - expected_wins) / std_wins
    
    return luck_factor


def calculate_entropy(probabilities: np.ndarray) -> float:
    """
    Calculate the entropy of a probability distribution.
    
    Higher entropy indicates more uncertainty/randomness (more luck).
    
    Args:
        probabilities: Array of probabilities
        
    Returns:
        Entropy value
    """
    # Remove zeros to avoid log(0)
    probs = probabilities[probabilities > 0]
    return -np.sum(probs * np.log2(probs))


def analyze_predictability(predictions: np.ndarray, actuals: np.ndarray) -> dict:
    """
    Analyze the predictability of game outcomes.
    
    Args:
        predictions: Predicted win probabilities
        actuals: Actual outcomes (0 or 1)
        
    Returns:
        Dictionary with various predictability metrics
    """
    # Average prediction confidence
    avg_confidence = np.mean(np.abs(predictions - 0.5))
    
    # Distribution of predictions
    close_games = np.sum((predictions > 0.4) & (predictions < 0.6))
    total_games = len(predictions)
    
    # Calibration: group predictions into bins and check actual win rate
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    calibration = []
    
    for i in range(n_bins):
        bin_mask = (predictions >= bins[i]) & (predictions < bins[i + 1])
        if bin_mask.sum() > 0:
            predicted_prob = predictions[bin_mask].mean()
            actual_prob = actuals[bin_mask].mean()
            calibration.append((predicted_prob, actual_prob, bin_mask.sum()))
    
    return {
        'avg_confidence': avg_confidence,
        'close_games_ratio': close_games / total_games,
        'calibration': calibration,
        'mean_prediction': np.mean(predictions),
        'std_prediction': np.std(predictions)
    }


def estimate_skill_vs_luck_ratio(model: BradleyTerryModel, 
                                 actual_results: List[Tuple[int, int, int]],
                                 schedule: List[Tuple[int, int]],
                                 n_simulations: int = 1000) -> float:
    """
    Estimate the ratio of skill to luck in determining outcomes.
    
    This uses variance decomposition: total variance = skill variance + luck variance
    
    Args:
        model: Fitted Bradley-Terry model
        actual_results: Actual game results
        schedule: Game schedule
        n_simulations: Number of simulations for estimating luck variance
        
    Returns:
        Ratio of skill variance to total variance (0 = all luck, 1 = all skill)
    """
    # Get actual win counts
    n_teams = model.n_teams
    actual_wins = np.zeros(n_teams)
    for team_i, team_j, result in actual_results:
        if result == 1:
            actual_wins[team_i] += 1
        else:
            actual_wins[team_j] += 1
    
    # Total variance in actual outcomes
    total_variance = np.var(actual_wins)
    
    # Simulate to estimate variance due to luck
    win_distribution = calculate_win_distribution(model, schedule, n_simulations)
    
    # Average variance across simulations (this is the luck component)
    luck_variance = np.mean(np.var(win_distribution, axis=1))
    
    # Skill variance is the remainder
    skill_variance = max(0, total_variance - luck_variance)
    
    # Ratio
    if total_variance == 0:
        return 0.0
    
    return skill_variance / total_variance
