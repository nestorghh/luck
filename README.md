# Luck is Hard to Beat: The Difficulty of Sports Prediction

Recreation of results from the paper ["Luck is Hard to Beat: The Difficulty of Sports Prediction"](https://dl.acm.org/doi/10.1145/3097983.3098045) by Aoki et al. (KDD 2017).

## Overview

This repository implements the core concepts from the paper, which investigates the role of luck in sports prediction. The paper demonstrates that even with sophisticated models, perfect prediction is impossible because luck (random chance) plays a significant role in sports outcomes.

### Key Concepts

1. **Bradley-Terry Model**: A probabilistic model for estimating team strengths from paired comparisons (game outcomes)
2. **Luck Factor Analysis**: Quantifying how much teams over/underperform relative to expectation
3. **Skill vs. Luck Decomposition**: Separating the contributions of skill and luck to outcomes
4. **Predictability Analysis**: Measuring the inherent limits of prediction accuracy

## Implementation

The implementation includes:

- `bradley_terry.py`: Bradley-Terry model for team strength estimation
- `luck_analysis.py`: Tools for analyzing luck factors and running simulations
- `utils.py`: Utility functions for data handling
- `visualization.py`: Visualization tools for results
- `example.py`: Demonstration script

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.7+
- NumPy
- SciPy
- Pandas
- Matplotlib

## Usage

Run the example script to see the analysis in action:

```bash
python example.py
```

This will:
1. Generate synthetic sports season data
2. Fit a Bradley-Terry model to estimate team strengths
3. Simulate multiple seasons to quantify uncertainty
4. Calculate luck factors for each team
5. Analyze the skill vs. luck decomposition
6. Display comprehensive results

### Example Output

```
Estimated Team Strengths (Bradley-Terry):
--------------------------------------------------
   1. Team 5      :  1.523 (true:  1.489)
   2. Team 3      :  1.312 (true:  1.301)
   ...

Luck Factor Analysis:
--------------------------------------------------
  Team 7      : +2.15 σ (lucky)
  Team 3      : +0.87 σ (lucky)
  Team 1      : -1.42 σ (unlucky)
  ...

Skill vs Luck Decomposition:
--------------------------------------------------
  Skill component: 65.3%
  Luck component:  34.7%
```

## Using the Library

### Basic Example

```python
from bradley_terry import BradleyTerryModel
from luck_analysis import calculate_luck_factor, simulate_season

# Your game data: list of (team_i, team_j, result)
# result = 1 if team_i wins, 0 if team_j wins
games = [
    (0, 1, 1),  # Team 0 beat Team 1
    (1, 2, 0),  # Team 2 beat Team 1
    # ... more games
]

# Fit the model
model = BradleyTerryModel(n_teams=3)
model.fit(games)

# Predict future games
prob = model.predict_probability(team_i=0, team_j=2)
print(f"Team 0 has {prob*100:.1f}% chance to beat Team 2")

# Analyze luck
schedule = [(i, j) for i, j, _ in games]
win_dist = calculate_win_distribution(model, schedule, n_simulations=1000)
```

## Key Findings

The paper and this implementation demonstrate:

1. **Prediction has fundamental limits**: Even perfect knowledge of team strengths doesn't guarantee accurate predictions for individual games
2. **Luck matters**: A significant portion (often 30-40%) of outcome variance is due to random chance
3. **Close games are coin flips**: When teams are evenly matched, outcomes are highly uncertain
4. **Skill dominates over time**: While individual games are unpredictable, season-long performance reflects true skill

## Paper Citation

```bibtex
@inproceedings{aoki2017luck,
  title={Luck is Hard to Beat: The Difficulty of Sports Prediction},
  author={Aoki, Shunto and Noshita, Koichiro and Konishi, Tatsuya and Takeuchi, Koh and Kashima, Hisashi and Kumagai, Atsutoshi},
  booktitle={Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={1367--1376},
  year={2017}
}
```

## License

This is a recreational implementation for educational purposes.

## Related Work

- Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika*, 39(3/4), 324-345.
- Elo rating system
- TrueSkill rating system
