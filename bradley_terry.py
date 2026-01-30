"""
Core implementation of the Bradley-Terry model for sports prediction.

This module implements the Bradley-Terry model used in the paper
"Luck is Hard to Beat: The Difficulty of Sports Prediction" by Aoki et al. (2017).

The Bradley-Terry model estimates team strengths from game outcomes and can be used
to predict win probabilities.
"""

import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple, Dict, Optional


class BradleyTerryModel:
    """
    Bradley-Terry model for paired comparisons (game outcomes).
    
    The model assumes that the probability of team i beating team j is:
    P(i beats j) = strength_i / (strength_i + strength_j)
    
    where strength values are positive real numbers representing team quality.
    """
    
    def __init__(self, n_teams: int):
        """
        Initialize the Bradley-Terry model.
        
        Args:
            n_teams: Number of teams in the competition
        """
        self.n_teams = n_teams
        self.strengths = np.ones(n_teams)  # Initialize with equal strengths
        self.team_names = None
        
    def set_team_names(self, names: List[str]):
        """Set team names for easier interpretation."""
        if len(names) != self.n_teams:
            raise ValueError(f"Expected {self.n_teams} team names, got {len(names)}")
        self.team_names = names
        
    def fit(self, games: List[Tuple[int, int, int]], max_iter: int = 100, tol: float = 1e-6):
        """
        Fit the Bradley-Terry model using Maximum Likelihood Estimation.
        
        Args:
            games: List of (team_i, team_j, result) where result is 1 if i wins, 0 if j wins
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
        """
        # Count wins and total games for each pair
        wins = np.zeros((self.n_teams, self.n_teams))
        games_count = np.zeros((self.n_teams, self.n_teams))
        
        for i, j, result in games:
            if result == 1:
                wins[i, j] += 1
            else:
                wins[j, i] += 1
            games_count[i, j] += 1
            games_count[j, i] += 1
        
        # Iterative algorithm (MM algorithm) for Bradley-Terry
        for iteration in range(max_iter):
            old_strengths = self.strengths.copy()
            
            # Update each team's strength
            total_wins = wins.sum(axis=1)
            denominators = np.zeros(self.n_teams)
            
            for i in range(self.n_teams):
                for j in range(self.n_teams):
                    if i != j and games_count[i, j] > 0:
                        denominators[i] += games_count[i, j] / (self.strengths[i] + self.strengths[j])
            
            # Avoid division by zero
            denominators[denominators == 0] = 1.0
            self.strengths = total_wins / denominators
            
            # Normalize to prevent numerical issues
            self.strengths = self.strengths / self.strengths.sum() * self.n_teams
            
            # Check convergence
            if np.abs(self.strengths - old_strengths).max() < tol:
                break
                
    def predict_probability(self, team_i: int, team_j: int) -> float:
        """
        Predict the probability that team_i beats team_j.
        
        Args:
            team_i: Index of the first team
            team_j: Index of the second team
            
        Returns:
            Probability that team_i wins
        """
        return self.strengths[team_i] / (self.strengths[team_i] + self.strengths[team_j])
    
    def get_rankings(self) -> List[Tuple[int, float]]:
        """
        Get team rankings based on estimated strengths.
        
        Returns:
            List of (team_index, strength) sorted by strength (descending)
        """
        rankings = [(i, self.strengths[i]) for i in range(self.n_teams)]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def get_team_strength(self, team_idx: int) -> float:
        """Get the estimated strength of a team."""
        return self.strengths[team_idx]


def calculate_log_loss(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculate log loss (cross-entropy) for predictions.
    
    Args:
        predictions: Predicted probabilities (0 to 1)
        actuals: Actual outcomes (0 or 1)
        
    Returns:
        Log loss value
    """
    # Clip predictions to avoid log(0)
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    
    return -np.mean(actuals * np.log(predictions) + (1 - actuals) * np.log(1 - predictions))


def calculate_accuracy(predictions: np.ndarray, actuals: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calculate prediction accuracy.
    
    Args:
        predictions: Predicted probabilities
        actuals: Actual outcomes (0 or 1)
        threshold: Decision threshold
        
    Returns:
        Accuracy (fraction of correct predictions)
    """
    predicted_outcomes = (predictions >= threshold).astype(int)
    return np.mean(predicted_outcomes == actuals)
