# FIFA World Cup 2026 – High-Performer Classification (ML Assignment 2)

## a. Problem statement
The goal is to predict whether a player had a **high-performance match** at the FIFA World Cup 2026, using that player's match statistics (attacking, passing, defending, discipline, goalkeeping and physical data). This is a **binary classification** problem:

- `high_performer = 1` if the player's match `player_rating` ≥ **5.50** (the dataset median), otherwise `0`.

Five classifiers (Logistic Regression, Decision Tree, KNN, Naive Bayes and SVM) are trained on the same data. They are compared on Accuracy, AUC, Precision, Recall, F1 and MCC, and deployed in an interactive Streamlit web app.

## b. Dataset description
- **Source:** Kaggle – [FIFA World Cup 2026 Player Performance Dataset](https://www.kaggle.com/datasets/rauffauzanrambe/fifa-world-cup-2026-player-performance-dataset) (file `fifa_world_cup_2026_player_performance.csv`). The dataset author notes that the data is synthetically generated.
- **Size:** 54,600 instances (one row per player per match) × 75 columns. This meets the minimum of 12 features and 500 instances.
- **Feature groups:**
  - Player profile: age, height, weight, position, preferred foot, market value
  - Match context: stage, result, team and opponent goals
  - Attacking: goals, assists, shots, xG, xA, key passes, dribbles
  - Passing: passes, pass accuracy, crosses
  - Defending: tackles, interceptions, clearances, blocks, aerial duels, recoveries
  - Discipline: fouls, cards, offsides
  - Goalkeeping: saves, save %, clean sheet, goals conceded
  - Physical: distance, sprint distance, accelerations, stamina
- **Target:** `high_performer`. Class balance: 1 = 50.59%, 0 = 49.41%.
- **Features used (56):** age, jersey_number, height_cm, weight_kg, market_value_eur, goals_team, goals_opponent, minutes_played, goals, assists, shots, shots_on_target, expected_goals_xg, expected_assists_xa, key_passes, successful_passes, total_passes, pass_accuracy, dribbles_attempted, successful_dribbles, crosses, successful_crosses, tackles, interceptions, clearances, blocks, aerial_duels_won, aerial_duels_lost, recoveries, defensive_actions, fouls_committed, fouls_suffered, yellow_cards, red_cards, offsides, saves, save_percentage, punches, clean_sheet, goals_conceded, penalty_saves, distance_covered_km, sprint_distance_km, accelerations, decelerations, stamina_score, player_of_match_awards, nationality, team, position, preferred_foot, stadium, city, opponent_team, tournament_stage, match_result.
- **Columns removed:**
  - *Target source and rating-derived scores (to avoid leakage):* player_rating, performance_score, offensive_contribution, defensive_contribution, possession_impact, pressure_resistance, creativity_score, consistency_score, clutch_performance_score, tournament_rating.
  - *Whole-tournament totals (not known at match time):* total_goals_tournament, total_assists_tournament, total_minutes_tournament.
  - *Identifiers:* player_id, player_name, club_name, match_id, match_date.
  - *Correlation check:* top_speed_kmh was removed automatically because its absolute correlation with player_rating was ≥ 0.90.
- **Pre-processing:** missing values are filled with the median (numeric) or mode (categorical). Numeric features are scaled with StandardScaler and categorical features are One-Hot encoded. The preprocessing is fitted on the training split only and saved inside each model pipeline.
- **Split:** stratified 80/20 train/test split (random_state = 42), giving 43,680 training rows and 10,920 test rows. The test split is provided as `test_data.csv`.

## c. GitHub Repository Link
- **GitHub repository:** <PASTE-YOUR-GITHUB-REPO-URL-HERE>
- **Live Streamlit app:** <PASTE-YOUR-STREAMLIT-APP-URL-HERE>

```
project-folder/
│-- app.py                  # Streamlit web application
│-- requirements.txt
│-- README.md
│-- test_data.csv           # 20% hold-out test split (upload this in the app)
│-- model/
│   │-- train_models.py     # source code: data prep, training, evaluation
│   │-- logistic_regression.pkl, decision_tree.pkl, knn.pkl, naive_bayes.pkl, svm.pkl
│   │-- preprocessor.pkl    # fitted preprocessing artifact
│   │-- metadata.json       # features, target threshold, hyper-parameters, versions
│   │-- metrics.csv         # evaluation metrics of all 5 models
```

## d. Models used
Hyper-parameters were tuned with 5-fold stratified GridSearchCV (F1 score) on the training set. The SVM uses an RBF kernel with C = 1 and gamma = 'scale', trained on a stratified 20,000-row subsample of the training set.

### Comparison Table (evaluation metrics on the hold-out test set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8817 | 0.9432 | 0.8856 | 0.8798 | 0.8827 | 0.7634 |
| Decision Tree | 0.9302 | 0.9559 | 0.8788 | 1.0000 | 0.9355 | 0.8687 |
| KNN | 0.8853 | 0.9381 | 0.8799 | 0.8955 | 0.8877 | 0.7707 |
| Naive Bayes | 0.8344 | 0.9313 | 0.8806 | 0.7782 | 0.8263 | 0.6739 |
| Support Vector Machine | 0.9301 | 0.9411 | 0.8788 | 0.9998 | 0.9354 | 0.8685 |

### Observations on the performance of each model

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Ranks 4th of 5 by MCC (0.7634). Train accuracy (0.8834) and test accuracy (0.8817) are almost equal, so it does not overfit. Precision (0.886) and recall (0.880) are balanced, so its errors are split evenly between false positives and false negatives. It scores about 0.05 below the Decision Tree and SVM in accuracy and about 0.10 in MCC, so a single linear boundary misses part of the pattern in the data. |
| Decision Tree | Ranks 1st by MCC (0.8687) and has the highest AUC (0.9559), even though the tuned tree is shallow (max_depth = 4). Train and test accuracy are very close (0.9273 vs 0.9302), so there is no overfitting. Recall is 1.0000, so it catches every high performer. All of its errors are false positives: low performers predicted as high (precision 0.8788). A few threshold splits are therefore enough to separate the classes well. |
| KNN | Ranks 3rd by MCC (0.7707), with k = 21 neighbours in the scaled feature space. Train and test accuracy are close (0.8875 vs 0.8853). Recall (0.896) is slightly higher than precision (0.880). It does better than Logistic Regression but clearly worse than the Decision Tree and SVM. It is also the slowest model at prediction time, because it compares every test row with the stored training data. |
| Naive Bayes | Ranks last by MCC (0.6739) and has the lowest accuracy (0.8344). Train and test accuracy are almost identical, so the weak score is due to bias, not overfitting. It has the lowest recall (0.778) of all the models, so it misses the most high performers, while its precision (0.881) is similar to the others. This fits its assumption that features are independent and normally distributed. Related statistics such as shots, shots on target and xG break that assumption. |
| Support Vector Machine | Ranks 2nd by MCC (0.8685), almost tied with the Decision Tree, and has a very similar error pattern: recall 0.9998 and precision 0.8788. Train and test accuracy are close (0.9273 vs 0.9301). Its AUC (0.9411) is lower than both the Decision Tree's and Logistic Regression's, so its probability ranking is weaker even though its final predictions are as good. It was trained on 20,000 rows because RBF-SVM training time grows roughly with the square of the number of rows. |
| Overall, Winner for your dataset? | **Decision Tree** is the winner, with the best MCC (0.8687), F1 (0.9355), accuracy (0.9302) and AUC (0.9559). The SVM is a very close second (MCC 0.8685). MCC is used as the deciding metric because it uses all four cells of the confusion matrix. The Decision Tree is also smaller and faster than the SVM, which makes it the better choice for this dataset. |

## How to run
```bash
pip install -r requirements.txt
python model/train_models.py      # downloads data, trains and saves all models, rewrites this README
streamlit run app.py
```
