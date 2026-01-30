"""
Example script demonstrating the recreation of results from 
"Luck is Hard to Beat: The Difficulty of Sports Prediction" by Aoki et al. (2017).

This script:
1. Generates synthetic sports data
2. Fits a Bradley-Terry model
3. Analyzes the role of luck in outcomes
4. Generates visualizations
"""

import numpy as np
from bradley_terry import BradleyTerryModel, calculate_log_loss, calculate_accuracy
from luck_analysis import (
    simulate_season, calculate_win_distribution, calculate_luck_factor,
    analyze_predictability, estimate_skill_vs_luck_ratio
)
from utils import (
    generate_synthetic_season, create_team_names, calculate_win_loss_records,
    split_train_test, format_probability, format_record
)
from visualization import (
    plot_team_strengths, plot_luck_factors, plot_win_distribution,
    plot_calibration, plot_prediction_confidence
)


def main():
    """Run the main analysis."""
    print("=" * 80)
    print("Recreating Results from 'Luck is Hard to Beat'")
    print("Paper by Aoki et al. (2017)")
    print("=" * 80)
    print()
    
    # Configuration
    N_TEAMS = 12
    N_GAMES_PER_PAIR = 2  # Each pair plays twice (home and away)
    STRENGTH_VARIANCE = 0.5  # Lower = more equal teams = more luck
    N_SIMULATIONS = 1000
    SEED = 42
    
    print(f"Configuration:")
    print(f"  Number of teams: {N_TEAMS}")
    print(f"  Games per team pair: {N_GAMES_PER_PAIR}")
    print(f"  Strength variance: {STRENGTH_VARIANCE}")
    print(f"  Simulations: {N_SIMULATIONS}")
    print()
    
    # Generate synthetic season data
    print("Generating synthetic season data...")
    games, schedule, true_strengths = generate_synthetic_season(
        n_teams=N_TEAMS,
        n_games_per_pair=N_GAMES_PER_PAIR,
        strength_variance=STRENGTH_VARIANCE,
        seed=SEED
    )
    team_names = create_team_names(N_TEAMS)
    print(f"  Generated {len(games)} games")
    print()
    
    # Calculate actual win-loss records
    wins, losses = calculate_win_loss_records(games, N_TEAMS)
    
    print("Actual Season Results:")
    print("-" * 50)
    for i in range(N_TEAMS):
        print(f"  {team_names[i]:12s}: {format_record(wins[i], losses[i])}")
    print()
    
    # Fit Bradley-Terry model
    print("Fitting Bradley-Terry model...")
    model = BradleyTerryModel(n_teams=N_TEAMS)
    model.set_team_names(team_names)
    model.fit(games)
    print("  Model fitted successfully")
    print()
    
    # Display estimated team strengths
    print("Estimated Team Strengths (Bradley-Terry):")
    print("-" * 50)
    rankings = model.get_rankings()
    for rank, (team_idx, strength) in enumerate(rankings, 1):
        true_str = true_strengths[team_idx]
        print(f"  {rank:2d}. {team_names[team_idx]:12s}: {strength:6.3f} (true: {true_str:6.3f})")
    print()
    
    # Evaluate predictions on the training data
    predictions = []
    actuals = []
    for team_i, team_j, result in games:
        prob = model.predict_probability(team_i, team_j)
        predictions.append(prob)
        actuals.append(result)
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    log_loss = calculate_log_loss(predictions, actuals)
    accuracy = calculate_accuracy(predictions, actuals)
    
    print("Prediction Performance:")
    print("-" * 50)
    print(f"  Log Loss: {log_loss:.4f}")
    print(f"  Accuracy: {format_probability(accuracy)}")
    print()
    
    # Analyze predictability
    print("Predictability Analysis:")
    print("-" * 50)
    predictability = analyze_predictability(predictions, actuals)
    print(f"  Average confidence: {predictability['avg_confidence']:.3f}")
    print(f"  Close games ratio: {format_probability(predictability['close_games_ratio'])}")
    print(f"  Mean prediction: {format_probability(predictability['mean_prediction'])}")
    print()
    
    # Simulate multiple seasons to analyze luck
    print("Simulating seasons to analyze luck...")
    win_distribution = calculate_win_distribution(model, schedule, N_SIMULATIONS)
    print(f"  Completed {N_SIMULATIONS} simulations")
    print()
    
    # Calculate expected wins from simulations
    expected_wins = np.mean(win_distribution, axis=0)
    
    # Calculate luck factors
    luck_factors = calculate_luck_factor(wins, expected_wins, win_distribution)
    
    print("Luck Factor Analysis:")
    print("-" * 50)
    print("  Luck factor measures how many standard deviations a team's")
    print("  actual performance differs from expectation.")
    print()
    
    # Sort teams by luck factor
    sorted_indices = np.argsort(luck_factors)[::-1]
    for team_idx in sorted_indices:
        factor = luck_factors[team_idx]
        descriptor = "lucky" if factor > 0 else "unlucky"
        print(f"  {team_names[team_idx]:12s}: {factor:+6.2f} σ ({descriptor})")
    print()
    
    # Estimate skill vs luck ratio
    print("Skill vs Luck Decomposition:")
    print("-" * 50)
    skill_ratio = estimate_skill_vs_luck_ratio(model, games, schedule, N_SIMULATIONS)
    luck_ratio = 1 - skill_ratio
    print(f"  Skill component: {format_probability(skill_ratio)}")
    print(f"  Luck component:  {format_probability(luck_ratio)}")
    print()
    print("  This shows that even in sports, a significant portion of")
    print("  outcomes can be attributed to random chance (luck).")
    print()
    
    # Key insight from the paper
    print("=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print("The paper 'Luck is Hard to Beat' demonstrates that:")
    print()
    print("  1. Even with accurate team strength estimates, prediction is limited")
    print("  2. Luck plays a substantial role in individual game outcomes")
    print("  3. Close matchups are particularly unpredictable")
    print("  4. Over many games, skill dominates, but individual results vary")
    print()
    print("This implementation recreates these findings through:")
    print("  - Bradley-Terry model for team strength estimation")
    print("  - Monte Carlo simulation for uncertainty quantification")
    print("  - Luck factor analysis to measure deviation from expectation")
    print("=" * 80)


if __name__ == "__main__":
    main()
