"""
Visualization utilities for luck analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from bradley_terry import BradleyTerryModel


def plot_team_strengths(model: BradleyTerryModel, team_names: Optional[List[str]] = None,
                       save_path: Optional[str] = None):
    """
    Plot team strengths from Bradley-Terry model.
    
    Args:
        model: Fitted Bradley-Terry model
        team_names: Optional team names
        save_path: Optional path to save figure
    """
    rankings = model.get_rankings()
    
    if team_names is None:
        team_names = [f"Team {i+1}" for i in range(model.n_teams)]
    
    sorted_indices = [idx for idx, _ in rankings]
    sorted_strengths = [strength for _, strength in rankings]
    sorted_names = [team_names[idx] for idx in sorted_indices]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_names)), sorted_strengths)
    plt.yticks(range(len(sorted_names)), sorted_names)
    plt.xlabel('Team Strength')
    plt.title('Bradley-Terry Model: Team Strengths')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def plot_luck_factors(luck_factors: np.ndarray, team_names: Optional[List[str]] = None,
                     save_path: Optional[str] = None):
    """
    Plot luck factors for each team.
    
    Args:
        luck_factors: Luck factor for each team
        team_names: Optional team names
        save_path: Optional path to save figure
    """
    n_teams = len(luck_factors)
    
    if team_names is None:
        team_names = [f"Team {i+1}" for i in range(n_teams)]
    
    # Sort by luck factor
    sorted_indices = np.argsort(luck_factors)[::-1]
    sorted_factors = luck_factors[sorted_indices]
    sorted_names = [team_names[i] for i in sorted_indices]
    
    # Color bars: green for lucky, red for unlucky
    colors = ['green' if f > 0 else 'red' for f in sorted_factors]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_names)), sorted_factors, color=colors, alpha=0.7)
    plt.yticks(range(len(sorted_names)), sorted_names)
    plt.xlabel('Luck Factor (standard deviations from expected)')
    plt.title('Team Luck Factors')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def plot_win_distribution(win_distribution: np.ndarray, actual_wins: np.ndarray,
                         team_idx: int, team_name: Optional[str] = None,
                         save_path: Optional[str] = None):
    """
    Plot the distribution of wins for a specific team across simulations.
    
    Args:
        win_distribution: Win counts from simulations (n_sims x n_teams)
        actual_wins: Actual win counts
        team_idx: Index of team to plot
        team_name: Optional team name
        save_path: Optional path to save figure
    """
    if team_name is None:
        team_name = f"Team {team_idx + 1}"
    
    team_wins = win_distribution[:, team_idx]
    
    plt.figure(figsize=(10, 6))
    plt.hist(team_wins, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(x=actual_wins[team_idx], color='red', linestyle='--', 
                linewidth=2, label='Actual wins')
    plt.axvline(x=np.mean(team_wins), color='blue', linestyle='--',
                linewidth=2, label='Expected wins')
    plt.xlabel('Number of Wins')
    plt.ylabel('Frequency')
    plt.title(f'Win Distribution for {team_name}')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def plot_calibration(calibration_data: List[Tuple[float, float, int]],
                    save_path: Optional[str] = None):
    """
    Plot calibration curve showing predicted vs actual probabilities.
    
    Args:
        calibration_data: List of (predicted_prob, actual_prob, count) tuples
        save_path: Optional path to save figure
    """
    if not calibration_data:
        return
    
    predicted = [p for p, _, _ in calibration_data]
    actual = [a for _, a, _ in calibration_data]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(predicted, actual, s=100, alpha=0.7)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Win Rate')
    plt.title('Model Calibration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def plot_prediction_confidence(predictions: np.ndarray, save_path: Optional[str] = None):
    """
    Plot histogram of prediction confidence levels.
    
    Args:
        predictions: Array of predicted probabilities
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6))
    plt.hist(predictions, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='50-50 (coin flip)')
    plt.xlabel('Predicted Win Probability')
    plt.ylabel('Number of Games')
    plt.title('Distribution of Prediction Confidence')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()
