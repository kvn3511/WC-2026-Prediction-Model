from flask import request, jsonify, render_template, Flask
import pickle
import pandas as pd
import os
import json

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "models", "xgboost_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "data", "processed", "team_stats.json")) as f:
    team_stats = json.load(f)

@app.route("/")
def index():
    teams = sorted(team_stats.keys())
    return render_template("index.html", teams=teams)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    home = data["home_team"]
    away = data["away_team"]

    h = team_stats.get(home, {})
    a = team_stats.get(away, {})

    home_elo = h.get("elo", 1500)
    away_elo = a.get("elo", 1500)
    home_wr  = h.get("win_rate", 0.5)
    away_wr  = a.get("win_rate", 0.5)

    features = pd.DataFrame([{
        "elo_diff": home_elo - away_elo,
        "win_rate_diff": home_wr - away_wr,
        "avg_scored": h.get("avg_scored", 1.5),
        "avg_conceded": h.get("avg_conceded", 1.2),
        "opp_avg_scored": a.get("avg_scored", 1.5),
        "opp_avg_conceded": a.get("avg_conceded", 1.2),
    }])

    prob = model.predict_proba(features)[0]
    return jsonify({
        "home_team": home,
        "away_team": away,
        "home_win_prob": round(float(prob[2]) * 100, 1),
        "draw_prob": round(float(prob[1]) * 100, 1),
        "away_win_prob": round(float(prob[0]) * 100, 1)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)