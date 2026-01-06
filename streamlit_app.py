#!/usr/bin/env python3
"""
Streamlit App - Previsão de partidas (Poisson + ML opcional)

Inclui:
- 1 ou 2 temporadas por URL + ponderação por recência (ex.: 70/30)
- Poisson: matriz de placares, top 5 mais/menos prováveis, 1X2, Over/Under, BTTS
- ML (RandomForest): estima lambdas e compara mercados (opcional)
- Presets (4 modelos) via botão
- Score de confiança (0-100) baseado na divergência Poisson x ML
- Alertas visuais por mercado (verde/amarelo/vermelho)
- Mercado recomendado automático (Top 1 + alternativas)
- Texto automático de análise do jogo
✅ NOVO:
- Perfil de risco (Conservador/Equilibrado/Agressivo) impacta recomendações
- Alerta de extremos de λ (Poisson/ML) para evitar previsões “explodidas”

Rodar:
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import poisson
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# =========================
# UI Config
# =========================

st.set_page_config(page_title="Futebol Predictor", page_icon="⚽", layout="wide")
st.title("⚽ Futebol Predictor — Poisson + ML (RandomForest)")
st.caption("2 temporadas por URL + ponderação por recência + presets + score de confiança. Sem previsão em lote.")


# =========================
# Presets (4 modelos)
# =========================

PRESETS = {
    "Liga grande (equilíbrio)": {
        "use_recency": True,
        "w_current": 70,
        "n_last": 8,
        "max_goals": 5,
        "use_ml": False,
        "desc": "Padrão para ligas grandes: estável e realista (70/30, forma=8, matriz=5)."
    },
    "Liga pequena (conservador)": {
        "use_recency": True,
        "w_current": 60,
        "n_last": 10,
        "max_goals": 5,
        "use_ml": False,
        "desc": "Mais estável contra ruído: forma=10 e 60/40. ML desligado."
    },
    "Copas / mata-mata (tático)": {
        "use_recency": False,
        "w_current": 100,
        "n_last": 6,
        "max_goals": 4,
        "use_ml": False,
        "desc": "Mais cauteloso (menos gols e mais controle). Sem recência e sem ML."
    },
    "Explosivo / valor (agressivo)": {
        "use_recency": True,
        "w_current": 80,
        "n_last": 6,
        "max_goals": 6,
        "use_ml": True,
        "desc": "Reage rápido ao momento (80/20), matriz maior e ML ligado para auditoria."
    }
}

def apply_preset(name: str):
    p = PRESETS[name]
    st.session_state["use_recency"] = p["use_recency"]
    st.session_state["w_current"] = int(p["w_current"])
    st.session_state["n_last"] = int(p["n_last"])
    st.session_state["max_goals"] = int(p["max_goals"])
    st.session_state["use_ml"] = bool(p["use_ml"])


# =========================
# Dataset fictício (opcional)
# =========================

def make_sample_dataset(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    league = "Premier League"
    teams = ["Arsenal", "Liverpool", "Man City", "Chelsea", "Tottenham", "Newcastle"]
    base_elos = {t: float(rng.integers(1450, 1850)) for t in teams}

    rows = []
    for i in range(220):
        home, away = rng.choice(teams, size=2, replace=False)
        home_elo = base_elos[home] + float(rng.normal(0, 20))
        away_elo = base_elos[away] + float(rng.normal(0, 20))

        elo_diff = (home_elo - away_elo) / 400.0
        lam_home = max(0.2, 1.35 + 0.25 * elo_diff + 0.20)
        lam_away = max(0.2, 1.10 - 0.20 * elo_diff)

        hg = int(rng.poisson(lam_home))
        ag = int(rng.poisson(lam_away))

        rows.append({
            "date": f"2025-{(i%12)+1:02d}-{(i%28)+1:02d}",
            "league": league,
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
            "home_elo": round(home_elo, 1),
            "away_elo": round(away_elo, 1),
            "home_odds": np.nan,
            "draw_odds": np.nan,
            "away_odds": np.nan,
            "season_tag": "SAMPLE",
        })

    df = pd.DataFrame(rows)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_dt"]).sort_values("date_dt").drop(columns=["date_dt"]).reset_index(drop=True)
    return df


# =========================
# Normalização tolerante (football-data e similares)
# =========================

DIV_TO_LEAGUE = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
    "SC0": "Scottish Premiership",
    "CL": "Champions League",
}

ODDS_CANDIDATES = [
    ("B365H", "B365D", "B365A"),
    ("PSH", "PSD", "PSA"),
    ("WHH", "WHD", "WHA"),
    ("VCH", "VCD", "VCA"),
    ("AvgH", "AvgD", "AvgA"),
    ("MaxH", "MaxD", "MaxA"),
]

def _first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None

def _pick_odds_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    cols = set(df.columns)
    for h, d, a in ODDS_CANDIDATES:
        if h in cols and d in cols and a in cols:
            return h, d, a
    return None, None, None

def _parse_date_series(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if dt.isna().mean() > 0.5:
        dt = pd.to_datetime(s, errors="coerce")
    return dt

def normalize_matches_dataframe(raw: pd.DataFrame, default_league_label: str, season_tag: str) -> pd.DataFrame:
    df = raw.copy()

    home_col = _first_existing(df, ["HomeTeam", "Home", "Home Team", "HT"])
    away_col = _first_existing(df, ["AwayTeam", "Away", "Away Team", "AT"])
    if home_col is None or away_col is None:
        raise ValueError("Não achei colunas de times. Esperado HomeTeam e AwayTeam.")

    hg_col = _first_existing(df, ["FTHG", "HG", "HomeGoals", "Home Goals"])
    ag_col = _first_existing(df, ["FTAG", "AG", "AwayGoals", "Away Goals"])
    if hg_col is None or ag_col is None:
        raise ValueError("Não achei colunas de gols FT. Esperado FTHG/FTAG.")

    date_col = _first_existing(df, ["Date", "date", "MatchDate"])
    if date_col is not None:
        df["date_dt"] = _parse_date_series(df[date_col])
        df["date"] = df["date_dt"].dt.strftime("%Y-%m-%d")
    else:
        df["date"] = pd.date_range("2000-01-01", periods=len(df), freq="D").strftime("%Y-%m-%d")
        df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")

    div_col = _first_existing(df, ["Div", "League", "Competition"])
    if div_col is not None:
        df["league"] = df[div_col].astype(str).map(lambda x: DIV_TO_LEAGUE.get(x, x))
    else:
        df["league"] = default_league_label

    df["home_team"] = df[home_col].astype(str).str.strip()
    df["away_team"] = df[away_col].astype(str).str.strip()

    df["home_goals"] = pd.to_numeric(df[hg_col], errors="coerce")
    df["away_goals"] = pd.to_numeric(df[ag_col], errors="coerce")

    oh, od, oa = _pick_odds_columns(df)
    if oh and od and oa:
        df["home_odds"] = pd.to_numeric(df[oh], errors="coerce")
        df["draw_odds"] = pd.to_numeric(df[od], errors="coerce")
        df["away_odds"] = pd.to_numeric(df[oa], errors="coerce")
    else:
        df["home_odds"] = np.nan
        df["draw_odds"] = np.nan
        df["away_odds"] = np.nan

    df["home_elo"] = 1600.0
    df["away_elo"] = 1600.0

    df_played = df.dropna(subset=["home_goals", "away_goals", "date_dt"]).copy()
    df_played["home_goals"] = df_played["home_goals"].astype(int)
    df_played["away_goals"] = df_played["away_goals"].astype(int)

    keep = [
        "date", "date_dt", "league", "home_team", "away_team", "home_goals", "away_goals",
        "home_elo", "away_elo", "home_odds", "draw_odds", "away_odds"
    ]
    df_played = df_played[keep].copy()
    df_played["season_tag"] = season_tag or "DATA"

    df_played = df_played.dropna(subset=["date", "league", "home_team", "away_team"])
    df_played = df_played.sort_values(["date_dt", "league", "home_team", "away_team"]).reset_index(drop=True)
    return df_played

def combine_histories(df_current: pd.DataFrame, df_prev: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df_prev is None:
        df = df_current.copy()
    else:
        df = pd.concat([df_current, df_prev], ignore_index=True)

    df["_key"] = (
        df["date"].astype(str) + "|" +
        df["league"].astype(str) + "|" +
        df["home_team"].astype(str) + "|" +
        df["away_team"].astype(str) + "|" +
        df["home_goals"].astype(str) + "|" +
        df["away_goals"].astype(str)
    )
    df = df.drop_duplicates(subset=["_key"]).drop(columns=["_key"])
    df = df.sort_values(["date_dt", "league", "home_team", "away_team"]).reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def load_url_normalized(url: str, league_label_override: str, season_tag: str) -> pd.DataFrame:
    raw = pd.read_csv(url)
    label = league_label_override.strip() if league_label_override.strip() else "URL Dataset"
    return normalize_matches_dataframe(raw, default_league_label=label, season_tag=season_tag)


# =========================
# Forma / stats com ponderação
# =========================

@dataclass
class TeamForm:
    team: str
    n_games: int
    gf_avg: float
    ga_avg: float
    gf_home_avg: float
    ga_home_avg: float
    gf_away_avg: float
    ga_away_avg: float
    points_per_game: float
    elo: float

def _points_from_score(gf: int, ga: int) -> int:
    if gf > ga:
        return 3
    if gf == ga:
        return 1
    return 0

def _wavg(values: List[float], weights: List[float], fallback: float) -> float:
    if len(values) == 0:
        return fallback
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    if np.all(w <= 0):
        return float(np.mean(v))
    return float(np.average(v, weights=w))

def compute_team_form(matches: pd.DataFrame, team: str, n_last: int, weights_by_season: Dict[str, float]) -> TeamForm:
    mask = (matches["home_team"] == team) | (matches["away_team"] == team)
    tm = matches.loc[mask].copy()
    if tm.empty:
        return TeamForm(team, 0, 1.0, 1.0, 1.1, 1.0, 0.9, 1.1, 1.0, 1600.0)

    tm = tm.sort_values("date_dt").tail(n_last)

    gf_list, ga_list, pts_list, w_list = [], [], [], []
    gf_home, ga_home, w_home = [], [], []
    gf_away, ga_away, w_away = [], [], []
    elo_values, elo_w = [], []

    for _, r in tm.iterrows():
        season = str(r.get("season_tag", "DATA"))
        w = float(weights_by_season.get(season, 1.0))

        if r["home_team"] == team:
            gf, ga = int(r["home_goals"]), int(r["away_goals"])
            gf_home.append(gf); ga_home.append(ga); w_home.append(w)
            elo_values.append(float(r.get("home_elo", 1600.0))); elo_w.append(w)
        else:
            gf, ga = int(r["away_goals"]), int(r["home_goals"])
            gf_away.append(gf); ga_away.append(ga); w_away.append(w)
            elo_values.append(float(r.get("away_elo", 1600.0))); elo_w.append(w)

        gf_list.append(gf)
        ga_list.append(ga)
        pts_list.append(_points_from_score(gf, ga))
        w_list.append(w)

    return TeamForm(
        team=team,
        n_games=len(tm),
        gf_avg=_wavg(gf_list, w_list, 1.0),
        ga_avg=_wavg(ga_list, w_list, 1.0),
        gf_home_avg=_wavg(gf_home, w_home, 1.1),
        ga_home_avg=_wavg(ga_home, w_home, 1.0),
        gf_away_avg=_wavg(gf_away, w_away, 0.9),
        ga_away_avg=_wavg(ga_away, w_away, 1.1),
        points_per_game=_wavg(pts_list, w_list, 1.0),
        elo=_wavg(elo_values, elo_w, 1600.0),
    )

def league_goal_averages(matches: pd.DataFrame, league: str, weights_by_season: Dict[str, float]) -> Dict[str, float]:
    df = matches[matches["league"] == league].copy()
    if df.empty:
        return {"avg_home_goals": 1.35, "avg_away_goals": 1.15, "avg_total_goals": 2.50}

    w = df["season_tag"].astype(str).map(lambda s: float(weights_by_season.get(s, 1.0))).values
    hg = df["home_goals"].astype(float).values
    ag = df["away_goals"].astype(float).values

    if np.all(w <= 0):
        avg_h = float(np.mean(hg))
        avg_a = float(np.mean(ag))
    else:
        avg_h = float(np.average(hg, weights=w))
        avg_a = float(np.average(ag, weights=w))

    return {"avg_home_goals": avg_h, "avg_away_goals": avg_a, "avg_total_goals": avg_h + avg_a}


# =========================
# Poisson
# =========================

def estimate_expected_goals_poisson(
    matches: pd.DataFrame,
    league: str,
    home_team: str,
    away_team: str,
    n_last: int,
    weights_by_season: Dict[str, float],
    home_advantage: float = 0.12,
    elo_k: float = 0.10,
) -> Tuple[float, float, Dict[str, float]]:
    lg = league_goal_averages(matches, league, weights_by_season)
    avg_home, avg_away = lg["avg_home_goals"], lg["avg_away_goals"]

    hf = compute_team_form(matches, home_team, n_last=n_last, weights_by_season=weights_by_season)
    af = compute_team_form(matches, away_team, n_last=n_last, weights_by_season=weights_by_season)

    eps = 1e-6
    attack_home = (hf.gf_home_avg + eps) / (avg_home + eps)
    defense_away = (af.ga_away_avg + eps) / (avg_home + eps)

    attack_away = (af.gf_away_avg + eps) / (avg_away + eps)
    defense_home = (hf.ga_home_avg + eps) / (avg_away + eps)

    form_home = 1.0 + 0.05 * (hf.points_per_game - 1.0)
    form_away = 1.0 + 0.05 * (af.points_per_game - 1.0)

    elo_diff = (hf.elo - af.elo) / 400.0
    elo_adj_home = 1.0 + elo_k * elo_diff
    elo_adj_away = 1.0 - elo_k * elo_diff

    lam_home = avg_home * attack_home * defense_away * (1.0 + home_advantage) * form_home * elo_adj_home
    lam_away = avg_away * attack_away * defense_home * (1.0 - home_advantage / 2.0) * form_away * elo_adj_away

    lam_home = float(np.clip(lam_home, 0.05, 4.50))
    lam_away = float(np.clip(lam_away, 0.05, 4.50))

    dbg = {
        "lambda_home": lam_home,
        "lambda_away": lam_away,
        "avg_home_goals_league": avg_home,
        "avg_away_goals_league": avg_away,
        "attack_home": float(attack_home),
        "defense_away": float(defense_away),
        "attack_away": float(attack_away),
        "defense_home": float(defense_home),
        "form_home": float(form_home),
        "form_away": float(form_away),
        "elo_diff_scaled": float(elo_diff),
        "weights_by_season": weights_by_season,
    }
    return lam_home, lam_away, dbg

def score_matrix_poisson(lambda_home: float, lambda_away: float, max_goals: int) -> pd.DataFrame:
    hs = np.arange(0, max_goals + 1)
    as_ = np.arange(0, max_goals + 1)
    p_home = poisson.pmf(hs, mu=lambda_home)
    p_away = poisson.pmf(as_, mu=lambda_away)
    mat = np.outer(p_home, p_away)
    return pd.DataFrame(mat, index=hs, columns=as_)

def list_top_bottom_scores(mat: pd.DataFrame, k: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    flat = []
    for hg in mat.index:
        for ag in mat.columns:
            flat.append((int(hg), int(ag), float(mat.loc[hg, ag])))

    top = sorted(flat, key=lambda x: x[2], reverse=True)[:k]
    bottom = sorted(flat, key=lambda x: x[2])[:k]

    top_df = pd.DataFrame([{"placar": f"{hg}x{ag}", "prob": p, "prob_%": 100.0*p} for hg, ag, p in top])
    bot_df = pd.DataFrame([{"placar": f"{hg}x{ag}", "prob": p, "prob_%": 100.0*p} for hg, ag, p in bottom])
    return top_df, bot_df

def probs_1x2_over_btts(mat: pd.DataFrame) -> Dict[str, float]:
    p_home = p_draw = p_away = 0.0
    p_over25 = 0.0
    p_btts = 0.0

    for hg in mat.index:
        for ag in mat.columns:
            p = float(mat.loc[hg, ag])
            if hg > ag:
                p_home += p
            elif hg == ag:
                p_draw += p
            else:
                p_away += p

            if (hg + ag) >= 3:
                p_over25 += p
            if (hg >= 1) and (ag >= 1):
                p_btts += p

    return {
        "home_win": p_home,
        "draw": p_draw,
        "away_win": p_away,
        "over_2_5": p_over25,
        "under_2_5": 1.0 - p_over25,
        "btts_yes": p_btts,
        "btts_no": 1.0 - p_btts,
    }


# =========================
# ML (RandomForest) — estável
# =========================

def build_ml_dataset(matches: pd.DataFrame, n_last: int, weights_by_season: Dict[str, float]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    rows = []
    ms = matches.sort_values("date_dt").reset_index(drop=True)

    for idx, r in ms.iterrows():
        hist = ms.iloc[:idx]
        if len(hist) < 10:
            continue

        league = r["league"]
        home = r["home_team"]
        away = r["away_team"]

        lg = league_goal_averages(hist, league, weights_by_season)
        hf = compute_team_form(hist, home, n_last=n_last, weights_by_season=weights_by_season)
        af = compute_team_form(hist, away, n_last=n_last, weights_by_season=weights_by_season)

        feat = {
            "avg_home_goals_league": lg["avg_home_goals"],
            "avg_away_goals_league": lg["avg_away_goals"],
            "home_gf_avg": hf.gf_avg,
            "home_ga_avg": hf.ga_avg,
            "home_gf_home_avg": hf.gf_home_avg,
            "home_ga_home_avg": hf.ga_home_avg,
            "home_ppg": hf.points_per_game,
            "away_gf_avg": af.gf_avg,
            "away_ga_avg": af.ga_avg,
            "away_gf_away_avg": af.gf_away_avg,
            "away_ga_away_avg": af.ga_away_avg,
            "away_ppg": af.points_per_game,
            "home_elo": hf.elo,
            "away_elo": af.elo,
            "elo_diff": hf.elo - af.elo,
            "target_season_weight": float(weights_by_season.get(str(r.get("season_tag", "DATA")), 1.0)),
        }

        rows.append({**feat, "y_home_goals": int(r["home_goals"]), "y_away_goals": int(r["away_goals"])})

    df = pd.DataFrame(rows).dropna()
    X = df.drop(columns=["y_home_goals", "y_away_goals"])
    y_home = df["y_home_goals"]
    y_away = df["y_away_goals"]
    return X, y_home, y_away

@st.cache_resource(show_spinner=False)
def train_ml_models_cached(matches: pd.DataFrame, n_last: int, weights_by_season: Dict[str, float], random_state: int = 42) -> Dict[str, object]:
    X, y_h, y_a = build_ml_dataset(matches, n_last=n_last, weights_by_season=weights_by_season)
    if len(X) < 80:
        raise ValueError(f"Dataset pequeno para ML ({len(X)} linhas). Use 2 temporadas ou mais histórico.")

    idx_all = np.arange(len(X))
    idx_train, idx_val = train_test_split(idx_all, test_size=0.2, random_state=random_state, shuffle=True)

    X_train, X_val = X.iloc[idx_train], X.iloc[idx_val]
    yh_train, yh_val = y_h.iloc[idx_train], y_h.iloc[idx_val]
    ya_train, ya_val = y_a.iloc[idx_train], y_a.iloc[idx_val]

    model_h = RandomForestRegressor(n_estimators=350, max_depth=12, random_state=random_state, n_jobs=-1)
    model_a = RandomForestRegressor(n_estimators=350, max_depth=12, random_state=random_state, n_jobs=-1)

    model_h.fit(X_train, yh_train)
    model_a.fit(X_train, ya_train)

    pred_h = model_h.predict(X_val)
    pred_a = model_a.predict(X_val)

    mae_h = mean_absolute_error(yh_val, pred_h)
    mae_a = mean_absolute_error(ya_val, pred_a)

    return {
        "model_home": model_h,
        "model_away": model_a,
        "mae_home": float(mae_h),
        "mae_away": float(mae_a),
        "feature_columns": list(X.columns),
        "n_rows": int(len(X)),
    }

def predict_expected_goals_ml(matches: pd.DataFrame, league: str, home_team: str, away_team: str, trained: Dict[str, object],
                              n_last: int, weights_by_season: Dict[str, float]) -> Tuple[float, float, Dict[str, float]]:
    cols = trained["feature_columns"]
    lg = league_goal_averages(matches, league, weights_by_season)

    hf = compute_team_form(matches, home_team, n_last=n_last, weights_by_season=weights_by_season)
    af = compute_team_form(matches, away_team, n_last=n_last, weights_by_season=weights_by_season)

    tsw = float(weights_by_season.get("CURRENT", np.mean(list(weights_by_season.values())) if weights_by_season else 1.0))

    row = {
        "avg_home_goals_league": lg["avg_home_goals"],
        "avg_away_goals_league": lg["avg_away_goals"],
        "home_gf_avg": hf.gf_avg,
        "home_ga_avg": hf.ga_avg,
        "home_gf_home_avg": hf.gf_home_avg,
        "home_ga_home_avg": hf.ga_home_avg,
        "home_ppg": hf.points_per_game,
        "away_gf_avg": af.gf_avg,
        "away_ga_avg": af.ga_avg,
        "away_gf_away_avg": af.gf_away_avg,
        "away_ga_away_avg": af.ga_away_avg,
        "away_ppg": af.points_per_game,
        "home_elo": hf.elo,
        "away_elo": af.elo,
        "elo_diff": hf.elo - af.elo,
        "target_season_weight": tsw,
    }

    X = pd.DataFrame([row])[cols]
    lam_home = float(trained["model_home"].predict(X)[0])
    lam_away = float(trained["model_away"].predict(X)[0])

    lam_home = float(np.clip(lam_home, 0.05, 4.50))
    lam_away = float(np.clip(lam_away, 0.05, 4.50))

    dbg = {
        "lambda_home_ml": lam_home,
        "lambda_away_ml": lam_away,
        "mae_home": trained["mae_home"],
        "mae_away": trained["mae_away"],
        "ml_rows_used": trained["n_rows"],
    }
    return lam_home, lam_away, dbg


# =========================
# Confiança + Alertas + Recomendação + Texto
# =========================

MARKET_LABELS = {
    "home_win": "1 (Mandante)",
    "draw": "X (Empate)",
    "away_win": "2 (Visitante)",
    "over_2_5": "Over 2.5",
    "under_2_5": "Under 2.5",
    "btts_yes": "BTTS Sim",
    "btts_no": "BTTS Não",
}

def confidence_score_from_models(probsP: Dict[str, float], probsM: Dict[str, float]) -> Tuple[int, Dict[str, float]]:
    main_keys = ["home_win", "draw", "away_win", "over_2_5", "btts_yes"]
    diffs_pp_all = {k: abs(probsP[k] - probsM[k]) * 100.0 for k in MARKET_LABELS.keys()}
    avg_diff = float(np.mean([diffs_pp_all[k] for k in main_keys]))
    score = int(np.clip(100 - (avg_diff * 2.0), 0, 100))
    return score, diffs_pp_all

def confidence_label(score: int) -> str:
    if score >= 80:
        return "Alta"
    if score >= 60:
        return "Média"
    return "Baixa"

def market_level(diff_pp: float) -> str:
    if diff_pp <= 8.0:
        return "verde"
    if diff_pp <= 15.0:
        return "amarelo"
    return "vermelho"

def lambda_extremes(lh: float, la: float) -> Tuple[bool, List[str]]:
    msgs = []
    extreme = False
    # limites práticos para futebol (heurística)
    if lh >= 3.50:
        extreme = True
        msgs.append(f"λ mandante muito alto ({lh:.2f}) → risco de superestimar goleada.")
    if la >= 3.50:
        extreme = True
        msgs.append(f"λ visitante muito alto ({la:.2f}) → risco de superestimar goleada.")
    if lh <= 0.30:
        extreme = True
        msgs.append(f"λ mandante muito baixo ({lh:.2f}) → risco de subestimar gols do mandante.")
    if la <= 0.30:
        extreme = True
        msgs.append(f"λ visitante muito baixo ({la:.2f}) → risco de subestimar gols do visitante.")
    return extreme, msgs

def risk_profile_params(profile: str) -> Dict[str, object]:
    """
    Ajusta recomendações:
    - Conservador: prioriza prob alta, evita mercados voláteis (1X2).
    - Equilibrado: padrão.
    - Agressivo: permite mais mercados e aceita prob menor se divergência for baixa.
    """
    if profile == "Conservador":
        return {
            "allowed": {"over_2_5", "under_2_5", "btts_yes", "btts_no"},  # evita 1X2
            "w_diff": 2.0,
            "w_prob": 1.6,
            "min_prob": 0.55,
        }
    if profile == "Agressivo":
        return {
            "allowed": set(MARKET_LABELS.keys()),
            "w_diff": 2.2,
            "w_prob": 0.9,
            "min_prob": 0.40,
        }
    # Equilibrado
    return {
        "allowed": set(MARKET_LABELS.keys()),
        "w_diff": 2.0,
        "w_prob": 1.2,
        "min_prob": 0.50,
    }

def recommended_markets(
    probsP: Dict[str, float],
    probsM: Dict[str, float],
    diffs_pp: Dict[str, float],
    profile: str
) -> List[Dict[str, object]]:
    params = risk_profile_params(profile)
    allowed = params["allowed"]
    w_diff = float(params["w_diff"])
    w_prob = float(params["w_prob"])
    min_prob = float(params["min_prob"])

    items = []
    for k, label in MARKET_LABELS.items():
        if k not in allowed:
            continue

        p_avg = 0.5 * (probsP[k] + probsM[k])
        diff = float(diffs_pp[k])
        lvl = market_level(diff)

        # penalidade por divergência e prob baixa
        prob_penalty = 0.0
        if p_avg < min_prob:
            prob_penalty = (min_prob - p_avg) * 100.0  # em pontos

        # score menor é melhor
        lvl_score = {"verde": 0.0, "amarelo": 7.0, "vermelho": 18.0}[lvl]
        score = (w_diff * diff) + (w_prob * (100.0 * (1.0 - p_avg))) + lvl_score + prob_penalty

        items.append({
            "key": k,
            "mercado": label,
            "diff_pp": diff,
            "prob_avg": float(p_avg),
            "nivel": lvl,
            "rank_score": float(score),
        })

    items_sorted = sorted(items, key=lambda x: x["rank_score"])
    return items_sorted[:3]

def auto_analysis_text(
    league: str,
    home_team: str,
    away_team: str,
    probsP: Dict[str, float],
    probsM: Dict[str, float],
    diffs_pp: Dict[str, float],
    score: int,
    label: str,
    lam_h: float,
    lam_a: float,
    lam_h_ml: float,
    lam_a_ml: float,
    recs: List[Dict[str, object]],
    risk_profile: str,
    extremes_msgs: List[str],
) -> str:
    avg_1 = 0.5 * (probsP["home_win"] + probsM["home_win"])
    avg_x = 0.5 * (probsP["draw"] + probsM["draw"])
    avg_2 = 0.5 * (probsP["away_win"] + probsM["away_win"])
    fav = "Mandante" if avg_1 >= max(avg_x, avg_2) else ("Visitante" if avg_2 >= max(avg_1, avg_x) else "Empate")
    fav_pct = max(avg_1, avg_x, avg_2) * 100.0

    over_avg = 0.5 * (probsP["over_2_5"] + probsM["over_2_5"]) * 100.0
    btts_avg = 0.5 * (probsP["btts_yes"] + probsM["btts_yes"]) * 100.0

    worst = sorted([(MARKET_LABELS[k], diffs_pp[k]) for k in ["home_win","draw","away_win","btts_yes","over_2_5"]], key=lambda x: x[1], reverse=True)[:2]
    w1, w2 = worst[0], worst[1]

    top1 = recs[0]
    alt = [r["mercado"] for r in recs[1:]]

    texto = []
    texto.append(f"**{home_team} x {away_team} ({league})**")
    texto.append(f"- **Perfil de risco:** **{risk_profile}** (isso muda o ranking de recomendações).")
    texto.append(f"- **Confiança:** {score}/100 (**{label}**) — quanto mais alta, mais Poisson e ML concordam.")
    texto.append(f"- **Gols esperados (Poisson):** {lam_h:.2f} x {lam_a:.2f} | **Gols esperados (ML):** {lam_h_ml:.2f} x {lam_a_ml:.2f}")
    texto.append(f"- **Tendência de resultado:** leve viés para **{fav}** (~{fav_pct:.1f}%).")
    texto.append(f"- **Tendência de gols:** Over 2.5 ~**{over_avg:.1f}%** | BTTS (Sim) ~**{btts_avg:.1f}%** (médias Poisson+ML).")

    if extremes_msgs:
        texto.append("")
        texto.append("**⚠️ Alerta de extremos (λ):**")
        for m in extremes_msgs:
            texto.append(f"- {m}")
        texto.append("Sugestão: aumente o histórico (2 temporadas) e/ou use perfil Conservador para recomendações.")

    texto.append("")
    texto.append("**✅ Mercado recomendado (pela consistência entre modelos + perfil de risco):**")
    texto.append(f"- **{top1['mercado']}** — divergência **{top1['diff_pp']:.2f} p.p.**, prob. média **{top1['prob_avg']*100:.1f}%** ({top1['nivel']}).")
    if alt:
        texto.append(f"- Alternativas: {', '.join(alt)}")

    texto.append("")
    texto.append("**⚠️ Pontos de atenção (onde os modelos mais discordam):**")
    texto.append(f"- {w1[0]}: **{w1[1]:.2f} p.p.**")
    texto.append(f"- {w2[0]}: **{w2[1]:.2f} p.p.**")
    texto.append("")
    texto.append("**Leitura rápida:** divergência alta em 1X2/BTTS = evite esses mercados e priorize os mercados com alerta verde.")
    return "\n".join(texto)


# =========================
# Plot helper
# =========================

def heatmap_figure(mat: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots()
    im = ax.imshow(mat.values, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Gols Visitante")
    ax.set_ylabel("Gols Mandante")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_yticks(range(len(mat.index)))
    ax.set_xticklabels(mat.columns)
    ax.set_yticklabels(mat.index)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig

def pct(x: float) -> float:
    return 100.0 * float(x)


# =========================
# Sidebar — Fonte + Presets + Modelo + Perfil de risco
# =========================

with st.sidebar:
    st.header("Presets (4 modelos)")
    preset_name = st.selectbox("Escolha um preset", list(PRESETS.keys()), index=0, key="preset_name")
    st.caption(PRESETS[preset_name]["desc"])
    if st.button("Aplicar preset agora"):
        apply_preset(preset_name)
        st.rerun()

    st.divider()
    st.header("Fonte de dados")
    source = st.radio("Escolha a fonte", ["URL (1 ou 2 temporadas)", "Dataset fictício"], index=0)

    url1 = ""
    url2 = ""
    league_override = ""

    if source == "URL (1 ou 2 temporadas)":
        st.caption("URL 1 = CURRENT. URL 2 = PREV (opcional).")
        url1 = st.text_input("URL CSV (Temporada atual)", value="https://www.football-data.co.uk/mmz4281/2526/E0.csv")
        url2 = st.text_input("URL CSV (Temporada anterior - opcional)", value="https://www.football-data.co.uk/mmz4281/2425/E0.csv")
        league_override = st.text_input("Opcional: nome da liga (override)", value="")

    st.divider()
    st.header("Modelo")

    use_recency = st.checkbox(
        "Usar ponderação por recência entre temporadas",
        value=st.session_state.get("use_recency", True),
        key="use_recency"
    )

    w_current = st.slider(
        "Peso temporada atual (%)",
        50, 95,
        value=int(st.session_state.get("w_current", 70)),
        step=1,
        disabled=not use_recency,
        key="w_current"
    )
    st.caption(f"Peso anterior = {100 - int(w_current)}%")

    n_last = st.slider(
        "Últimos N jogos (forma)",
        5, 10,
        value=int(st.session_state.get("n_last", 8)),
        key="n_last"
    )

    max_goals = st.slider(
        "Máximo de gols na matriz",
        3, 7,
        value=int(st.session_state.get("max_goals", 5)),
        key="max_goals"
    )

    use_ml = st.checkbox(
        "Comparar com ML (RandomForest)",
        value=bool(st.session_state.get("use_ml", False)),
        key="use_ml"
    )

    st.divider()
    st.header("Recomendação")
    risk_profile = st.selectbox(
        "Perfil de risco",
        ["Conservador", "Equilibrado", "Agressivo"],
        index=1,
        help="Controla o ranking do 'mercado recomendado'. Conservador evita 1X2 e prioriza probabilidade mais alta."
    )


# =========================
# Carregar dados
# =========================

try:
    if source == "Dataset fictício":
        played = make_sample_dataset()
        has_two_seasons = False
    else:
        if not url1.strip():
            st.stop()
        df_current = load_url_normalized(url1.strip(), league_override, "CURRENT")

        df_prev = None
        has_two_seasons = bool(url2.strip())
        if has_two_seasons:
            df_prev = load_url_normalized(url2.strip(), league_override, "PREV")

        played = combine_histories(df_current, df_prev)

except Exception as e:
    st.error(f"Erro ao carregar/normalizar dados: {e}")
    st.stop()

if played.empty:
    st.error("Nenhum jogo com placar encontrado. Histórico vazio.")
    st.stop()

if source == "URL (1 ou 2 temporadas)" and has_two_seasons and use_recency:
    weights_by_season = {"CURRENT": int(w_current) / 100.0, "PREV": (100 - int(w_current)) / 100.0}
else:
    seasons = sorted(played["season_tag"].astype(str).unique().tolist())
    weights_by_season = {s: 1.0 for s in seasons}

n_current = int((played["season_tag"] == "CURRENT").sum()) if "season_tag" in played.columns else len(played)
n_prev = int((played["season_tag"] == "PREV").sum()) if "season_tag" in played.columns else 0

st.info(f"Histórico: **{len(played)} jogos** | CURRENT: {n_current} | PREV: {n_prev} | Pesos: {weights_by_season}")

with st.expander("📄 Preview do histórico"):
    st.dataframe(played.tail(60), use_container_width=True)


# =========================
# Filtros: liga / times
# =========================

leagues = sorted(played["league"].dropna().astype(str).unique().tolist())
teams_all = sorted(set(played["home_team"].dropna().astype(str).unique().tolist()) | set(played["away_team"].dropna().astype(str).unique().tolist()))

colA, colB, colC = st.columns([1.2, 1, 1])
with colA:
    league = st.selectbox("Liga", leagues, index=0)
with colB:
    home_team = st.selectbox("Mandante", teams_all, index=0)
with colC:
    away_team = st.selectbox("Visitante", teams_all, index=1 if len(teams_all) > 1 else 0)

if home_team == away_team:
    st.warning("Selecione times diferentes.")
    st.stop()

st.divider()


# =========================
# Poisson — jogo único
# =========================

lam_h, lam_a, dbgP = estimate_expected_goals_poisson(
    played, league, home_team, away_team,
    n_last=n_last, weights_by_season=weights_by_season
)

matP = score_matrix_poisson(lam_h, lam_a, max_goals=max_goals)
topP, botP = list_top_bottom_scores(matP, k=5)
probsP = probs_1x2_over_btts(matP)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("λ Mandante (Poisson)", f"{lam_h:.2f}")
c2.metric("λ Visitante (Poisson)", f"{lam_a:.2f}")
c3.metric("1 (Mandante)", f"{pct(probsP['home_win']):.1f}%")
c4.metric("X (Empate)", f"{pct(probsP['draw']):.1f}%")
c5.metric("2 (Visitante)", f"{pct(probsP['away_win']):.1f}%")
c6.metric("Over 2.5", f"{pct(probsP['over_2_5']):.1f}%")

c7, c8, c9, c10 = st.columns(4)
c7.metric("Under 2.5", f"{pct(probsP['under_2_5']):.1f}%")
c8.metric("BTTS (Sim)", f"{pct(probsP['btts_yes']):.1f}%")
c9.metric("BTTS (Não)", f"{pct(probsP['btts_no']):.1f}%")
c10.metric("Total gols (liga, médio)", f"{league_goal_averages(played, league, weights_by_season)['avg_total_goals']:.2f}")

# ✅ Alerta de extremos (Poisson)
extP, msgsP = lambda_extremes(lam_h, lam_a)
if extP:
    st.warning("⚠️ Alerta: λ extremo no Poisson. Isso pode distorcer placares e mercados.")
    for m in msgsP:
        st.caption(f"- {m}")

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Matriz de placares — Poisson")
    fig = heatmap_figure(matP * 100.0, "Probabilidade (%) por placar (Poisson)")
    st.pyplot(fig, clear_figure=True)

with right:
    st.subheader("Placares mais / menos prováveis — Poisson")
    t = topP.copy()
    t["prob_%"] = t["prob_%"].astype(float).round(3)
    st.markdown("**Top 5 mais prováveis**")
    st.dataframe(
        t[["placar", "prob_%"]].style.format({"prob_%": "{:.3f}%"}).background_gradient(subset=["prob_%"]),
        use_container_width=True
    )

    b = botP.copy()
    b["prob_%"] = b["prob_%"].astype(float).round(6)
    st.markdown("**Top 5 menos prováveis**")
    st.dataframe(b[["placar", "prob_%"]].style.format({"prob_%": "{:.6f}%"}), use_container_width=True)

with st.expander("🔎 Detalhes do cálculo (Poisson)"):
    st.json(dbgP)

st.divider()


# =========================
# ML + Confiança + Alertas + Recomendação + Texto
# =========================

if use_ml:
    try:
        with st.spinner("Treinando ML (RandomForest)..."):
            trained = train_ml_models_cached(played, n_last=n_last, weights_by_season=weights_by_season)

        lam_h_ml, lam_a_ml, dbgM = predict_expected_goals_ml(
            played, league, home_team, away_team,
            trained=trained, n_last=n_last, weights_by_season=weights_by_season
        )

        matM = score_matrix_poisson(lam_h_ml, lam_a_ml, max_goals=max_goals)
        topM, botM = list_top_bottom_scores(matM, k=5)
        probsM = probs_1x2_over_btts(matM)

        score, diffs_pp = confidence_score_from_models(probsP, probsM)
        label = confidence_label(score)

        # ✅ Recomendações respeitando perfil de risco
        recs = recommended_markets(probsP, probsM, diffs_pp, profile=risk_profile)

        st.subheader("Comparação — Poisson vs ML (RandomForest) + Confiança")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("λ Mandante (ML)", f"{lam_h_ml:.2f}")
        c2.metric("λ Visitante (ML)", f"{lam_a_ml:.2f}")
        c3.metric("MAE Home", f"{dbgM['mae_home']:.3f}")
        c4.metric("MAE Away", f"{dbgM['mae_away']:.3f}")
        c5.metric("Confiança", f"{score}/100")
        c6.metric("Nível", label)

        # ✅ Alerta de extremos (ML)
        extM, msgsM = lambda_extremes(lam_h_ml, lam_a_ml)
        extremes_msgs = []
        if extP:
            extremes_msgs.extend([f"(Poisson) {m}" for m in msgsP])
        if extM:
            extremes_msgs.extend([f"(ML) {m}" for m in msgsM])

        if extM:
            st.warning("⚠️ Alerta: λ extremo no ML. Use com cautela (pode ser poucos dados/viés recente).")
            for m in msgsM:
                st.caption(f"- {m}")

        st.markdown("### ✅ Alertas por mercado (consistência Poisson × ML)")
        st.caption("Verde: modelos concordam | Amarelo: divergência moderada | Vermelho: divergência alta (mais risco).")

        cols = st.columns(7)
        keys_order = ["home_win", "draw", "away_win", "over_2_5", "under_2_5", "btts_yes", "btts_no"]
        for i, k in enumerate(keys_order):
            diff = float(diffs_pp[k])
            lvl = market_level(diff)
            text = f"{MARKET_LABELS[k]}\n{diff:.1f} p.p."
            if lvl == "verde":
                cols[i].success(text)
            elif lvl == "amarelo":
                cols[i].warning(text)
            else:
                cols[i].error(text)

        st.markdown("### 🎯 Mercado recomendado (automático)")
        st.caption(f"Perfil aplicado: **{risk_profile}**")
        top1 = recs[0]
        st.info(
            f"**Top 1:** {top1['mercado']} — divergência **{top1['diff_pp']:.2f} p.p.** | "
            f"prob. média (Poisson+ML) **{top1['prob_avg']*100:.1f}%** | nível: **{top1['nivel']}**"
        )
        if len(recs) > 1:
            alt_str = " | ".join([f"{r['mercado']} ({r['diff_pp']:.1f} p.p.)" for r in recs[1:]])
            st.caption(f"Alternativas: {alt_str}")

        st.markdown("### 📝 Análise automática do jogo")
        analysis = auto_analysis_text(
            league=league,
            home_team=home_team,
            away_team=away_team,
            probsP=probsP,
            probsM=probsM,
            diffs_pp=diffs_pp,
            score=score,
            label=label,
            lam_h=lam_h,
            lam_a=lam_a,
            lam_h_ml=lam_h_ml,
            lam_a_ml=lam_a_ml,
            recs=recs,
            risk_profile=risk_profile,
            extremes_msgs=extremes_msgs,
        )
        st.markdown(analysis)

        tab1, tab2, tab3, tab4 = st.tabs(["Heatmap ML", "Top/Bottom ML", "Resumo Mercados", "Divergências"])
        with tab1:
            fig2 = heatmap_figure(matM * 100.0, "Probabilidade (%) por placar (ML -> λ -> Poisson)")
            st.pyplot(fig2, clear_figure=True)

        with tab2:
            l, r = st.columns(2)
            with l:
                tm = topM.copy()
                tm["prob_%"] = tm["prob_%"].astype(float).round(3)
                st.markdown("**Top 5 mais prováveis (ML)**")
                st.dataframe(
                    tm[["placar", "prob_%"]].style.format({"prob_%": "{:.3f}%"}).background_gradient(subset=["prob_%"]),
                    use_container_width=True
                )
            with r:
                bm = botM.copy()
                bm["prob_%"] = bm["prob_%"].astype(float).round(6)
                st.markdown("**Top 5 menos prováveis (ML)**")
                st.dataframe(bm[["placar", "prob_%"]].style.format({"prob_%": "{:.6f}%"}), use_container_width=True)

        with tab3:
            cmp = pd.DataFrame({
                "Mercado": [MARKET_LABELS[k] for k in keys_order],
                "Poisson (%)": [100*probsP[k] for k in keys_order],
                "ML (%)": [100*probsM[k] for k in keys_order],
            })
            st.dataframe(
                cmp.style.format({"Poisson (%)": "{:.2f}", "ML (%)": "{:.2f}"}).background_gradient(subset=["Poisson (%)", "ML (%)"]),
                use_container_width=True
            )

        with tab4:
            dd = pd.DataFrame({
                "Mercado": [MARKET_LABELS[k] for k in keys_order],
                "Diferença (p.p.)": [diffs_pp[k] for k in keys_order],
                "Nível": [market_level(diffs_pp[k]) for k in keys_order],
            })
            st.dataframe(
                dd.style.format({"Diferença (p.p.)": "{:.2f}"}).background_gradient(subset=["Diferença (p.p.)"]),
                use_container_width=True
            )

        with st.expander("🔎 Detalhes do ML"):
            st.json(dbgM)

    except Exception as e:
        st.error(f"Falha ao treinar/rodar ML: {e}")

st.caption("Dica: perfil de risco muda o ranking; alerta de extremos avisa quando λ está fora do padrão esperado.")
