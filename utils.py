"""
Utility functions for data generation and handling.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


def generate_synthetic_season(n_teams: int, n_games_per_pair: int = 1, 
                              strength_variance: float = 1.0,
                              seed: Optional[int] = None) -> Tuple[List[Tuple[int, int, int]], 
                                                                   List[Tuple[int, int]], 
                                                                   np.ndarray]:
    """
    Generate synthetic season data for testing.
    
    Creates a round-robin schedule where teams have different strength levels,
    and games are simulated probabilistically.
    
    Args:
        n_teams: Number of teams
        n_games_per_pair: Number of games between each pair of teams
        strength_variance: Variance in true team strengths (higher = more skill-based)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (games, schedule, true_strengths) where:
        - games: List of (team_i, team_j, result)
        - schedule: List of (team_i, team_j) matchups
        - true_strengths: True underlying team strengths
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate true team strengths
    true_strengths = np.random.lognormal(mean=0, sigma=strength_variance, size=n_teams)
    true_strengths = true_strengths / true_strengths.sum() * n_teams  # Normalize
    
    # Generate round-robin schedule
    schedule = []
    for i in range(n_teams):
        for j in range(i + 1, n_teams):
            for _ in range(n_games_per_pair):
                schedule.append((i, j))
    
    # Simulate games
    games = []
    for team_i, team_j in schedule:
        # Probability that team_i wins using Bradley-Terry formula
        prob_i_wins = true_strengths[team_i] / (true_strengths[team_i] + true_strengths[team_j])
        result = np.random.binomial(1, prob_i_wins)
        games.append((team_i, team_j, result))
    
    return games, schedule, true_strengths


def create_team_names(n_teams: int, prefix: str = "Team") -> List[str]:
    """Create default team names."""
    return [f"{prefix} {i+1}" for i in range(n_teams)]


def calculate_win_loss_records(games: List[Tuple[int, int, int]], 
                               n_teams: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate win-loss records for all teams.
    
    Args:
        games: List of (team_i, team_j, result)
        n_teams: Number of teams
        
    Returns:
        Tuple of (wins, losses) arrays
    """
    wins = np.zeros(n_teams)
    losses = np.zeros(n_teams)
    
    for team_i, team_j, result in games:
        if result == 1:
            wins[team_i] += 1
            losses[team_j] += 1
        else:
            wins[team_j] += 1
            losses[team_i] += 1
    
    return wins, losses


def split_train_test(games: List[Tuple[int, int, int]], 
                     train_ratio: float = 0.8,
                     seed: Optional[int] = None) -> Tuple[List[Tuple[int, int, int]], 
                                                          List[Tuple[int, int, int]]]:
    """
    Split games into training and test sets.
    
    Args:
        games: List of games
        train_ratio: Fraction of games for training
        seed: Random seed
        
    Returns:
        Tuple of (train_games, test_games)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_games = len(games)
    indices = np.random.permutation(n_games)
    n_train = int(n_games * train_ratio)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_games = [games[i] for i in train_indices]
    test_games = [games[i] for i in test_indices]
    
    return train_games, test_games


def format_probability(prob: float) -> str:
    """Format probability as percentage string."""
    return f"{prob * 100:.1f}%"


def format_record(wins: int, losses: int) -> str:
    """Format win-loss record."""
    total = wins + losses
    if total == 0:
        return "0-0 (-.---)"
    pct = wins / total
    return f"{int(wins)}-{int(losses)} ({pct:.3f})"
