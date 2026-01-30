"""
Simple tests for the luck analysis implementation.
"""

import numpy as np
from bradley_terry import BradleyTerryModel, calculate_log_loss, calculate_accuracy
from luck_analysis import simulate_season, calculate_win_distribution, calculate_luck_factor
from utils import generate_synthetic_season, calculate_win_loss_records


def test_bradley_terry_basic():
    """Test basic Bradley-Terry model functionality."""
    print("Testing Bradley-Terry model...")
    
    # Simple 3-team scenario
    # Team 0 is strongest, Team 2 is weakest
    games = [
        (0, 1, 1),  # Team 0 beats Team 1
        (0, 2, 1),  # Team 0 beats Team 2
        (1, 2, 1),  # Team 1 beats Team 2
        (0, 1, 1),  # Team 0 beats Team 1
        (0, 2, 1),  # Team 0 beats Team 2
        (1, 2, 1),  # Team 1 beats Team 2
    ]
    
    model = BradleyTerryModel(n_teams=3)
    model.fit(games)
    
    # Check that Team 0 is ranked highest
    rankings = model.get_rankings()
    assert rankings[0][0] == 0, "Team 0 should be ranked highest"
    assert rankings[2][0] == 2, "Team 2 should be ranked lowest"
    
    # Check probabilities are reasonable
    prob_0_beats_1 = model.predict_probability(0, 1)
    prob_0_beats_2 = model.predict_probability(0, 2)
    assert prob_0_beats_1 > 0.5, "Team 0 should be favored over Team 1"
    assert prob_0_beats_2 > 0.5, "Team 0 should be favored over Team 2"
    
    print("  ✓ Bradley-Terry model basic tests passed")


def test_prediction_metrics():
    """Test prediction metric calculations."""
    print("Testing prediction metrics...")
    
    predictions = np.array([0.8, 0.6, 0.4, 0.3, 0.9])
    actuals = np.array([1, 1, 0, 0, 1])
    
    accuracy = calculate_accuracy(predictions, actuals)
    log_loss = calculate_log_loss(predictions, actuals)
    
    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"
    assert log_loss >= 0, "Log loss should be non-negative"
    assert accuracy == 1.0, "All predictions should be correct with threshold 0.5"
    
    print("  ✓ Prediction metrics tests passed")


def test_synthetic_data_generation():
    """Test synthetic data generation."""
    print("Testing synthetic data generation...")
    
    n_teams = 6
    games, schedule, true_strengths = generate_synthetic_season(
        n_teams=n_teams,
        n_games_per_pair=1,
        seed=123
    )
    
    assert len(true_strengths) == n_teams, f"Should have {n_teams} strengths"
    assert len(schedule) == n_teams * (n_teams - 1) // 2, "Should have correct number of games"
    assert len(games) == len(schedule), "Games and schedule should match"
    
    # Check that all results are 0 or 1
    for _, _, result in games:
        assert result in [0, 1], "Results should be 0 or 1"
    
    print("  ✓ Synthetic data generation tests passed")


def test_win_loss_records():
    """Test win-loss record calculation."""
    print("Testing win-loss record calculation...")
    
    games = [
        (0, 1, 1),  # Team 0 wins
        (0, 2, 1),  # Team 0 wins
        (1, 2, 0),  # Team 2 wins
    ]
    
    wins, losses = calculate_win_loss_records(games, n_teams=3)
    
    assert wins[0] == 2, "Team 0 should have 2 wins"
    assert wins[1] == 0, "Team 1 should have 0 wins"
    assert wins[2] == 1, "Team 2 should have 1 win"
    
    assert losses[0] == 0, "Team 0 should have 0 losses"
    assert losses[1] == 2, "Team 1 should have 2 losses"
    assert losses[2] == 1, "Team 2 should have 1 loss"
    
    print("  ✓ Win-loss record tests passed")


def test_simulation():
    """Test season simulation."""
    print("Testing season simulation...")
    
    # Create a simple model
    model = BradleyTerryModel(n_teams=3)
    model.strengths = np.array([2.0, 1.0, 0.5])
    
    schedule = [(0, 1), (0, 2), (1, 2)]
    
    # Simulate multiple seasons
    results = simulate_season(model, schedule, n_simulations=100)
    
    assert results.shape == (100, 3), "Should have correct shape"
    assert np.all((results == 0) | (results == 1)), "All results should be 0 or 1"
    
    # Team 0 should win most games on average
    avg_results = results.mean(axis=0)
    assert avg_results[0] > 0.5, "Team 0 should win first matchup most of the time"
    
    print("  ✓ Simulation tests passed")


def test_luck_factor():
    """Test luck factor calculation."""
    print("Testing luck factor calculation...")
    
    actual_wins = np.array([10, 5, 3])
    expected_wins = np.array([8, 6, 4])
    
    # Create synthetic distribution
    np.random.seed(42)
    win_distribution = np.random.normal(
        loc=expected_wins.reshape(1, -1),
        scale=2.0,
        size=(100, 3)
    )
    
    luck_factors = calculate_luck_factor(actual_wins, expected_wins, win_distribution)
    
    assert len(luck_factors) == 3, "Should have luck factor for each team"
    assert luck_factors[0] > 0, "Team 0 should be lucky (won more than expected)"
    assert luck_factors[1] < 0, "Team 1 should be unlucky (won less than expected)"
    
    print("  ✓ Luck factor tests passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Tests")
    print("=" * 60)
    print()
    
    test_bradley_terry_basic()
    test_prediction_metrics()
    test_synthetic_data_generation()
    test_win_loss_records()
    test_simulation()
    test_luck_factor()
    
    print()
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
