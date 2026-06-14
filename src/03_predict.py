import pickle
import pandas as pd
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, "models", "xgboost_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "data", "processed", "team_stats.json")) as f:
    team_stats = json.load(f)

def predict_match(home_team, away_team, is_neutral=1):
    h = team_stats.get(home_team, {})
    a = team_stats.get(away_team, {})

    home_elo = h.get("elo", 1500)
    away_elo = a.get("elo", 1500)
    home_wr  = h.get("win_rate", 0.5)
    away_wr  = a.get("win_rate", 0.5)

    features = pd.DataFrame([{
        "home_elo":          home_elo,
        "away_elo":          away_elo,
        "elo_diff":          home_elo - away_elo,        # ✅ new
        "home_win_rate":     home_wr,
        "away_win_rate":     away_wr,
        "win_rate_diff":     home_wr - away_wr,
        "is_neutral":        is_neutral,
        "home_avg_scored":   h.get("avg_scored", 1.5),
        "away_avg_scored":   a.get("avg_scored", 1.5),
        "home_avg_conceded": h.get("avg_conceded", 1.2),
    }])

    prob = model.predict_proba(features)[0]
    print(f"\n{home_team} vs {away_team}")
    print(f"  {home_team} win: {prob[1]:.1%}")
    print(f"  {away_team} win: {prob[0]:.1%}")

# Test it
predict_match("Brazil", "France")
predict_match("England", "Argentina")


