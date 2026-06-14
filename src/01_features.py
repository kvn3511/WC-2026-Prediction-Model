import pandas as pd
import numpy as np
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

results = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "results.csv"), parse_dates=["date"])
results = results[results["date"] >= "2014-01-01"].copy()

competitive = results[results["tournament"] != "Friendly"].copy()
competitive["home_win"] = (competitive["home_score"] > competitive["away_score"]).astype(int)

# ── Elo ratings ──────────────────────────────────────────────────────────────
def compute_elo(df, k=30, base=1500):
    elo = {}
    df = df.sort_values("date")
    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        elo.setdefault(h, base)
        elo.setdefault(a, base)
        exp_h = 1 / (1 + 10 ** ((elo[a] - elo[h]) / 400))  # ✅ Bug 3 fixed
        exp_a = 1 - exp_h
        if row["home_score"] > row["away_score"]:
            act_h, act_a = 1, 0
        elif row["home_score"] == row["away_score"]:
            act_h, act_a = 0.5, 0.5
        else:
            act_h, act_a = 0, 1
        elo[h] += k * (act_h - exp_h)
        elo[a] += k * (act_a - exp_a)
    return elo

# ── Win rates ────────────────────────────────────────────────────────────────
def compute_win_rates(df):                                   # ✅ Bug 2 fixed
    records = {}
    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        for team in [h, a]:
            if team not in records:
                records[team] = {"wins": 0, "games": 0}
        records[h]["games"] += 1
        records[a]["games"] += 1
        if row["home_score"] > row["away_score"]:
            records[h]["wins"] += 1
        elif row["away_score"] > row["home_score"]:
            records[a]["wins"] += 1
    return {
        team: info["wins"] / info["games"] if info["games"] > 0 else 0.5
        for team, info in records.items()
    }

elo_ratings = compute_elo(competitive)
win_rates = compute_win_rates(competitive)

# ── Attach features to each match ────────────────────────────────────────────
competitive["home_elo"] = competitive["home_team"].map(elo_ratings).fillna(1500)
competitive["away_elo"] = competitive["away_team"].map(elo_ratings).fillna(1500)
competitive["elo_diff"] = competitive["home_elo"] - competitive["away_elo"]
competitive["is_neutral"] = competitive["neutral"].astype(int)

competitive["home_win_rate"] = competitive["home_team"].map(win_rates).fillna(0.5)  # ✅ Bug 4 fixed
competitive["away_win_rate"] = competitive["away_team"].map(win_rates).fillna(0.5)
competitive["win_rate_diff"] = competitive["home_win_rate"] - competitive["away_win_rate"]

tournament_weights = {
    "FIFA World Cup": 3,
    "UEFA Euro": 2,
    "Copa América": 2,
    "FIFA World Cup qualification": 2,
    "Friendly": 0
}
competitive["tournament_weight"] = competitive["tournament"].map(tournament_weights).fillna(1)

home_goals_scored = competitive.groupby("home_team")["home_score"].mean()
home_goals_conceded = competitive.groupby("home_team")["away_score"].mean()
away_goals_scored = competitive.groupby("away_team")["away_score"].mean()

competitive["home_avg_scored"] = competitive["home_team"].map(home_goals_scored).fillna(1.5)
competitive["away_avg_scored"] = competitive["away_team"].map(away_goals_scored).fillna(1.5)
competitive["home_avg_conceded"] = competitive["home_team"].map(home_goals_conceded).fillna(1.5)

# ── Save features.csv ────────────────────────────────────────────────────────
features = [
    "home_elo", "away_elo", "elo_diff",
    "home_win_rate", "away_win_rate", "win_rate_diff",
    "is_neutral", "home_avg_scored", "away_avg_scored",
    "home_avg_conceded", "home_win"
]
competitive[features].dropna().to_csv(
    os.path.join(BASE_DIR, "data", "processed", "features.csv"), index=False
)
print("Features saved!")

# ── Export team_stats.json for app.py ────────────────────────────────────────
world_cup_2026_teams = [
    "Algeria", "Argentina", "Australia", "Austria", "Belgium",
    "Bosnia and Herzegovina", "Brazil", "Cabo Verde", "Colombia",
    "Congo DR", "Côte d'Ivoire", "Croatia", "Curaçao", "Czechia",
    "Ecuador", "Egypt", "England", "France", "Germany", "Ghana",
    "Haiti", "Iran", "Iraq", "Japan", "Jordan", "Korea Republic",
    "Morocco", "Netherlands", "New Zealand", "Norway", "Panama",
    "Paraguay", "Portugal", "Qatar", "Saudi Arabia", "Scotland",
    "Senegal", "South Africa", "Spain", "Sweden", "Switzerland",
    "Tunisia", "Türkiye", "Uruguay", "Uzbekistan"
]

team_stats = {}
for team in world_cup_2026_teams:
    home_scored = home_goals_scored.get(team, 1.5)
    away_scored = away_goals_scored.get(team, 1.5)
    team_stats[team] = {
        "win_rate": round(win_rates.get(team, 0.5), 3),
        "avg_scored": round((home_scored + away_scored) / 2, 3),
        "avg_conceded": round(home_goals_conceded.get(team, 1.2), 3),
        "elo": round(elo_ratings.get(team, 1500), 1)
    }

output_path = os.path.join(BASE_DIR, "data", "processed", "team_stats.json")
with open(output_path, "w") as f:
    json.dump(team_stats, f, indent=2)

print("team_stats.json saved!")