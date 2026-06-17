import pandas as pd
import numpy as np
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

results = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "results.csv"), parse_dates=["date"])
results = results[results["date"] >= "2014-01-01"].copy()
results = results[results["tournament"] != "Friendly"].copy()
results = results[results["neutral"] == True].copy()
results = results.dropna(subset=["home_score", "away_score"]).copy()
results = results.sort_values("date").reset_index(drop=True)

tournament_weights = {
    "FIFA World Cup": 3,
    "UEFA Euro": 2,
    "Copa América": 2,
    "FIFA World Cup qualification": 2,
}


# for each match we record elo before updating, then update after
def compute_elo_sequential(df, k=30, base=1500):
    elo = {}
    pre_elo_h, pre_elo_a = [], []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        elo.setdefault(h, base)
        elo.setdefault(a, base)

        pre_elo_h.append(elo[h])
        pre_elo_a.append(elo[a])

        exp_h = 1 / (1 + 10 ** ((elo[a] - elo[h]) / 400))
        exp_a = 1 - exp_h
        if row["home_score"] > row["away_score"]:
            act_h, act_a = 1, 0
        elif row["home_score"] == row["away_score"]:
            act_h, act_a = 0.5, 0.5
        else:
            act_h, act_a = 0, 1
        elo[h] += k * (act_h - exp_h)
        elo[a] += k * (act_a - exp_a)

    return pre_elo_h, pre_elo_a, elo

pre_elo_h, pre_elo_a, final_elo = compute_elo_sequential(results)
results["home_elo_pre"] = pre_elo_h
results["away_elo_pre"] = pre_elo_a

def compute_rolling_win_rates(df):
    records = {}
    pre_wr_h, pre_wr_a = [], []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        for team in [h, a]:
            if team not in records:
                records[team] = {"wins": 0, "games": 0}


        h_wr = records[h]["wins"] / records[h]["games"] if records[h]["games"] > 0 else 0.5
        a_wr = records[a]["wins"] / records[a]["games"] if records[a]["games"] > 0 else 0.5
        pre_wr_h.append(h_wr)
        pre_wr_a.append(a_wr)


        records[h]["games"] += 1
        records[a]["games"] += 1
        if row["home_score"] > row["away_score"]:
            records[h]["wins"] += 1
        elif row["away_score"] > row["home_score"]:
            records[a]["wins"] += 1

    return pre_wr_h, pre_wr_a, records

pre_wr_h, pre_wr_a, final_win_records = compute_rolling_win_rates(results)
results["home_wr_pre"] = pre_wr_h
results["away_wr_pre"] = pre_wr_a

def compute_rolling_goals(df):
    scored = {}
    conceded = {}
    pre_scored_h, pre_scored_a, pre_conceded_h, pre_conceded_a = [], [], [], []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        for team in [h, a]:
            if team not in scored:
                scored[team] = []
                conceded[team] = []

        # store averages before this match
        pre_scored_h.append(np.mean(scored[h][-10:]) if scored[h] else 1.5)
        pre_scored_a.append(np.mean(scored[a][-10:]) if scored[a] else 1.5)
        pre_conceded_h.append(np.mean(conceded[h][-10:]) if conceded[h] else 1.2)
        pre_conceded_a.append(np.mean(conceded[a][-10:]) if conceded[a] else 1.2)

        # update after
        scored[h].append(row["home_score"])
        scored[a].append(row["away_score"])
        conceded[h].append(row["away_score"])
        conceded[a].append(row["home_score"])

    return pre_scored_h, pre_scored_a, pre_conceded_h, pre_conceded_a, scored, conceded

pre_scored_h, pre_scored_a, pre_conceded_h, pre_conceded_a, final_scored, final_conceded = compute_rolling_goals(results)
results["home_avg_scored_pre"] = pre_scored_h
results["away_avg_scored_pre"] = pre_scored_a
results["home_avg_conceded_pre"] = pre_conceded_h
results["away_avg_conceded_pre"] = pre_conceded_a

rows = []
for _, row in results.iterrows():
    w = tournament_weights.get(row["tournament"], 1)

    if row["home_score"] > row["away_score"]:
        match_result = 2
    elif row["home_score"] == row["away_score"]:
        match_result = 1
    else:
        match_result = 0

    rows.append({
        "elo_diff": row["home_elo_pre"] - row["away_elo_pre"],
        "win_rate_diff": row["home_wr_pre"] - row["away_wr_pre"],
        "avg_scored": row["home_avg_scored_pre"],
        "avg_conceded": row["home_avg_conceded_pre"],
        "opp_avg_scored": row["away_avg_scored_pre"],
        "opp_avg_conceded": row["away_avg_conceded_pre"],
        "tournament_weight": w,
        "result": match_result
    })

    if row["away_score"] > row["home_score"]:
        flipped_result = 2
    elif row["away_score"] == row["home_score"]:
        flipped_result = 1
    else:
        flipped_result = 0

    rows.append({
        "elo_diff": row["away_elo_pre"] - row["home_elo_pre"],
        "win_rate_diff": row["away_wr_pre"] - row["home_wr_pre"],
        "avg_scored": row["away_avg_scored_pre"],
        "avg_conceded": row["away_avg_conceded_pre"],
        "opp_avg_scored": row["home_avg_scored_pre"],
        "opp_avg_conceded": row["home_avg_conceded_pre"],
        "tournament_weight": w,
        "result": flipped_result
    })

competitive = pd.DataFrame(rows)
competitive.to_csv(os.path.join(BASE_DIR, "data", "processed", "features.csv"), index=False)
print("Features saved!")

final_win_rates = {
    team: info["wins"] / info["games"] if info["games"] > 0 else 0.5
    for team, info in final_win_records.items()
}
final_avg_scored = {team: np.mean(goals[-10:]) if goals else 1.5 for team, goals in final_scored.items()}
final_avg_conceded = {team: np.mean(goals[-10:]) if goals else 1.2 for team, goals in final_conceded.items()}

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
    team_stats[team] = {
        "win_rate": round(final_win_rates.get(team, 0.5), 3),
        "avg_scored": round(final_avg_scored.get(team, 1.5), 3),
        "avg_conceded": round(final_avg_conceded.get(team, 1.2), 3),
        "elo": round(final_elo.get(team, 1500), 1)
    }

output_path = os.path.join(BASE_DIR, "data", "processed", "team_stats.json")
with open(output_path, "w") as f:
    json.dump(team_stats, f, indent=2)
print("team_stats.json saved!")