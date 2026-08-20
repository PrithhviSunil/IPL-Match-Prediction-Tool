# IPL Match Prediction Tool

A machine learning model that predicts the winners of Indian Premier League cricket matches using historical data and team form analysis. Built with Python, scikit-learn, and pandas.

## Results

- **60.9% accuracy** on 2026 IPL season matches
- Trained on **1,200+ matches** from 2008–2025
- Baseline accuracy (random guessing): 50%

## How It Works

The model is a Random Forest Classifier trained on engineered features including:
- Team encodings
- Venue
- Each team's win rate in the current season
- Each team's last 5 matches form
- Head-to-head record between the two teams

All features are calculated using only data available *before* each match to prevent data leakage.

## Feature Importance

Through feature importance analysis, the model identified:
- **Venue** (17%) and **head-to-head record** (17%) as the strongest predictors
- **Toss outcome** as a weak predictor (<2%), contradicting common cricket assumptions

## Tech Stack

- Python
- pandas, numpy
- scikit-learn (Random Forest, Label Encoding, train/test split)
- matplotlib

## Dataset

Uses ball-by-ball IPL data from Kaggle (283K+ rows), aggregated to match-level records.

https://www.kaggle.com/datasets/chaitu20/ipl-dataset2008-2025 — download separately as the file exceeds GitHub's size 

