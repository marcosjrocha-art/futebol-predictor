#!/usr/bin/env python3
"""
Futebol Predictor — Poisson + Dixon–Coles + Elo dinâmico + ML + Backtest/Calibração

Inclui:
- Fonte por URL (1 ou 2 temporadas) + ponderação por recência (ex.: 70/30)
- Normalização tolerante (football-data.co.uk e similares)
- Elo dinâmico (jogo-a-jogo) pré-partida (Upgrade 1)
- Poisson + Dixon–Coles (Upgrade 2): toggle + rho + preset por liga + efeito nos low-scores
- Probabilidades: placares (0..N), Top 5 mais e menos prováveis, 1X2, Over/Under 2.5, BTTS
- ML (RandomForest) opcional: estima lambdas e compara com Poisson
- Confiança (0–100) pela divergência Poisson x ML
- EV (Valor Esperado) usando odds do CSV (quando existirem)
- Upgrade 3: Backtest rolling (sem data leakage), Brier, LogLoss, ROI EV+, reliability curve

Rodar:
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from scipy.stats import poisson

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from odds_api import (
    LEAGUE_TO_SPORT_KEY,
    get_odds_api_key,
    get_quota_headers,
    fetch_best_h2h_odds,
    match_event_best_prices,
    implied_probs_from_decimal_odds,
)
from sklearn.calibration import calibration_curve


# =========================
# UI
# =========================

# =========================
# BOOTSTRAP — Produção/Decisão (sempre antes de qualquer uso)
# =========================
def is_production_mode() -> bool:
    # Produção se botão ativo OU perfil conservador
    if st.session_state.get("app_mode") == "PRODUCAO":
        return True
    if st.session_state.get("risk_profile") == "Conservador":
        return True
    return False

def production_rules_snapshot() -> dict:
    return {
        "EV mínimo 1X2": float(st.session_state.get("ev_min_1x2", 0.06)),
        "Prob mínima 1X2": float(st.session_state.get("pmin_1x2", 0.50)),
        "Confiança mínima": int(st.session_state.get("conf_min", 75)),
        "Filtro vermelho": True,
        "Região odds": str(st.session_state.get("odds_region", "eu")),
    }

def production_decision(
    recs: list,
    confidence_score: int | None,
    odds_available: bool,
    ev_table=None,
) -> tuple[str, list]:
    """
    Retorna: ("APOSTAR" ou "NÃO APOSTAR", motivos[])
    Em produção: bloqueia se não houver mercado seguro / odds / EV / confiança.
    """
    if not is_production_mode():
        return "LAB", ["Modo Laboratório: sem bloqueio estrito."]

    rules = production_rules_snapshot()
    ev_min = float(rules["EV mínimo 1X2"])
    pmin = float(rules["Prob mínima 1X2"])
    conf_min = int(rules["Confiança mínima"])

    reasons = []

    if not recs:
        reasons.append("Sem mercado seguro (todos vermelhos ou filtrados).")

    if confidence_score is not None and confidence_score < conf_min:
        reasons.append(f"Confiança abaixo do mínimo ({confidence_score} < {conf_min}).")

    if not odds_available:
        reasons.append("Odds não disponíveis (não dá para validar EV).")

    if ev_table is not None and getattr(ev_table, "__len__", lambda: 0)() > 0:
        ok = False
        for _, r in ev_table.iterrows():
            try:
                ev = float(r.get("EV", None))
                prob = float(r.get("Prob (modelo)", None))
            except Exception:
                continue
            if ev >= ev_min and prob >= pmin:
                ok = True
                break
        if not ok:
            reasons.append(f"Nenhuma opção 1X2 com EV≥{ev_min:.2f} e Prob≥{pmin:.2f}.")

    if reasons:
        return "NÃO APOSTAR", reasons
    return "APOSTAR", ["Passou nos filtros do Modo Produção."]




st.set_page_config(page_title="Futebol Predictor", page_icon="⚽", layout="wide")

# init log
if "decision_log" not in st.session_state:
    st.session_state["decision_log"] = []

st.title("⚽ Futebol Predictor — Poisson + Dixon–Coles + Elo + ML + Backtest")
st.caption("Sem dados pagos. Compatível com football-data.co.uk. Backtest rolling e calibração para EV mais realista.")




# =========================
# Badge: modo ativo
# =========================
mode = st.session_state.get("app_mode", "")

rules = production_rules_snapshot()
if is_production_mode():
    with st.expander("🔒 Regras de Produção ativas", expanded=False):
        st.json(rules)

if mode == "PRODUCAO":
    st.success("🔒 **Modo Produção ativo** — filtros mais rígidos (EV maior, menos overfit).")
elif mode == "LABORATORIO":
    st.info("🧪 **Modo Laboratório ativo** — parâmetros mais reativos para explorar hipóteses.")

# =========================
# Presets do modelo
# =========================

PRESETS = {
    "Liga grande (equilíbrio)": {
        "use_recency": True, "w_current": 70, "n_last": 8, "max_goals": 5,
        "use_ml": False, "use_dc": True, "rho": -0.06,
        "desc": "Padrão realista: 70/30, forma=8, matriz=5, Dixon–Coles ligado."
    },
    "Liga pequena (conservador)": {
        "use_recency": True, "w_current": 60, "n_last": 10, "max_goals": 5,
        "use_ml": False, "use_dc": True, "rho": -0.08,
        "desc": "Mais estável contra ruído: forma=10, 60/40. DC mais forte."
    },
    "Copas / mata-mata (tático)": {
        "use_recency": False, "w_current": 100, "n_last": 6, "max_goals": 4,
        "use_ml": False, "use_dc": True, "rho": -0.07,
        "desc": "Menos gols e mais controle: forma=6, matriz=4. DC ligado."
    },
    "Explosivo / valor (agressivo)": {
        "use_recency": True, "w_current": 80, "n_last": 6, "max_goals": 6,
        "use_ml": True, "use_dc": True, "rho": -0.05,
        "desc": "Reage rápido: 80/20, matriz=6, ML ligado como auditor."
    },
}


def apply_preset(name: str) -> None:
    p = PRESETS[name]
    st.session_state["use_recency"] = bool(p["use_recency"])
    st.session_state["w_current"] = int(p["w_current"])
    st.session_state["n_last"] = int(p["n_last"])
    st.session_state["max_goals"] = int(p["max_goals"])
    st.session_state["use_ml"] = bool(p["use_ml"])
    st.session_state["use_dc"] = bool(p["use_dc"])


# =========================
# Normalização tolerante (football-data e similares)
# =========================

DIV_TO_LEAGUE = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
    "B1": "Jupiler Pro League",
    "N1": "Eredivisie",
    "P1": "Primeira Liga",
    "SC0": "Scottish Premiership",
    "BRA": "Brasileirão",
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

# Presets simples de ρ (heurístico) por liga — ajuste no slider se quiser
DC_RHO_PRESETS = {
    "Premier League": -0.06,
    "La Liga": -0.07,
    "Serie A": -0.08,
    "Bundesliga": -0.04,
    "Ligue 1": -0.07,
    "Brasileirão": -0.08,
    "Champions League": -0.06,
}


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
        raise ValueError("Não encontrei colunas de times (ex.: HomeTeam e AwayTeam).")

    hg_col = _first_existing(df, ["FTHG", "HG", "HomeGoals", "Home Goals"])
    ag_col = _first_existing(df, ["FTAG", "AG", "AwayGoals", "Away Goals"])
    if hg_col is None or ag_col is None:
        raise ValueError("Não encontrei colunas de gols FT (ex.: FTHG e FTAG).")

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

    # placeholders (serão preenchidos pelo Elo dinâmico se habilitado)
    df["home_elo"] = 1500.0
    df["away_elo"] = 1500.0

    df_played = df.dropna(subset=["home_goals", "away_goals", "date_dt"]).copy()
    df_played["home_goals"] = df_played["home_goals"].astype(int)
    df_played["away_goals"] = df_played["away_goals"].astype(int)

    keep = [
        "date", "date_dt", "league", "home_team", "away_team",
        "home_goals", "away_goals", "home_elo", "away_elo",
        "home_odds", "draw_odds", "away_odds"
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
# Elo dinâmico (Upgrade 1)
# =========================

def _elo_expected_score(r_home: float, r_away: float, hfa: float) -> float:
    return 1.0 / (1.0 + 10 ** (((r_away) - (r_home + hfa)) / 400.0))


def _elo_mov_multiplier(goal_diff: int, elo_diff: float) -> float:
    if goal_diff <= 1:
        return 1.0
    return (math.log(goal_diff + 1.0) * (2.2 / ((elo_diff * 0.001) + 2.2)))


@st.cache_data(show_spinner=False)
def add_dynamic_elo_columns(
    matches: pd.DataFrame,
    base_elo: float = 1500.0,
    k: float = 20.0,
    hfa: float = 65.0,
    use_mov: bool = True,
    per_league: bool = True,
) -> pd.DataFrame:
    df = matches.copy()
    df = df.sort_values(["date_dt", "league", "home_team", "away_team"]).reset_index(drop=True)

    ratings: Dict[str, float] = {}

    def key_for(league: str, team: str) -> str:
        return f"{league}||{team}" if per_league else team

    home_elos: List[float] = []
    away_elos: List[float] = []

    for _, r in df.iterrows():
        league = str(r["league"])
        home = str(r["home_team"])
        away = str(r["away_team"])

        kh = key_for(league, home)
        ka = key_for(league, away)

        r_home = float(ratings.get(kh, base_elo))
        r_away = float(ratings.get(ka, base_elo))

        # pré-jogo
        home_elos.append(r_home)
        away_elos.append(r_away)

        hg = int(r["home_goals"])
        ag = int(r["away_goals"])

        if hg > ag:
            s_home = 1.0
        elif hg == ag:
            s_home = 0.5
        else:
            s_home = 0.0

        e_home = _elo_expected_score(r_home, r_away, hfa=hfa)
        e_away = 1.0 - e_home

        goal_diff = abs(hg - ag)
        elo_diff = (r_home + hfa) - r_away
        mult = _elo_mov_multiplier(goal_diff, elo_diff) if use_mov else 1.0

        k_eff = float(k) * float(mult)

        r_home_new = r_home + k_eff * (s_home - e_home)
        r_away_new = r_away + k_eff * ((1.0 - s_home) - e_away)

        ratings[kh] = r_home_new
        ratings[ka] = r_away_new

    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    return df


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
        return TeamForm(team, 0, 1.0, 1.0, 1.1, 1.0, 0.9, 1.1, 1.0, 1500.0)

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
            elo_values.append(float(r.get("home_elo", 1500.0))); elo_w.append(w)
        else:
            gf, ga = int(r["away_goals"]), int(r["home_goals"])
            gf_away.append(gf); ga_away.append(ga); w_away.append(w)
            elo_values.append(float(r.get("away_elo", 1500.0))); elo_w.append(w)

        gf_list.append(gf); ga_list.append(ga)
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
        elo=_wavg(elo_values, elo_w, 1500.0),
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
# Poisson + Dixon–Coles
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
        "weights_by_season": dict(weights_by_season),
    }
    return lam_home, lam_away, dbg


def score_matrix_poisson(lambda_home: float, lambda_away: float, max_goals: int) -> pd.DataFrame:
    hs = np.arange(0, max_goals + 1)
    as_ = np.arange(0, max_goals + 1)
    p_home = poisson.pmf(hs, mu=lambda_home)
    p_away = poisson.pmf(as_, mu=lambda_away)
    mat = np.outer(p_home, p_away)
    s = float(mat.sum())
    if s > 0:
        mat = mat / s
    return pd.DataFrame(mat, index=hs, columns=as_)


def dixon_coles_tau(hg: int, ag: int, lam_h: float, lam_a: float, rho: float) -> float:
    if hg == 0 and ag == 0:
        return 1.0 - (lam_h * lam_a * rho)
    if hg == 0 and ag == 1:
        return 1.0 + (lam_h * rho)
    if hg == 1 and ag == 0:
        return 1.0 + (lam_a * rho)
    if hg == 1 and ag == 1:
        return 1.0 - rho
    return 1.0


def score_matrix_dixon_coles(lambda_home: float, lambda_away: float, max_goals: int, rho: float) -> pd.DataFrame:
    hs = np.arange(0, max_goals + 1)
    as_ = np.arange(0, max_goals + 1)
    p_home = poisson.pmf(hs, mu=lambda_home)
    p_away = poisson.pmf(as_, mu=lambda_away)
    mat = np.outer(p_home, p_away)

    # Ajuste apenas nos low scores
    for hg in [0, 1]:
        for ag in [0, 1]:
            if hg <= max_goals and ag <= max_goals:
                mat[hg, ag] = mat[hg, ag] * dixon_coles_tau(hg, ag, lambda_home, lambda_away, rho)

    s = float(mat.sum())
    if s > 0:
        mat = mat / s

    return pd.DataFrame(mat, index=hs, columns=as_)


def list_top_bottom_scores(mat: pd.DataFrame, k: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    flat: List[Tuple[int, int, float]] = []
    for hg in mat.index:
        for ag in mat.columns:
            flat.append((int(hg), int(ag), float(mat.loc[hg, ag])))

    top = sorted(flat, key=lambda x: x[2], reverse=True)[:k]
    bottom = sorted(flat, key=lambda x: x[2])[:k]

    top_df = pd.DataFrame([{"placar": f"{hg}x{ag}", "prob": p, "prob_%": 100.0 * p} for hg, ag, p in top])
    bot_df = pd.DataFrame([{"placar": f"{hg}x{ag}", "prob": p, "prob_%": 100.0 * p} for hg, ag, p in bottom])
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
# ML (RandomForest) — opcional
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
# Confiança + recomendação + EV
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

        # 🔒 Modo Produção: bloqueia mercados vermelhos SEM EXCEÇÃO
        if is_production_mode() and lvl == "vermelho":
            continue

        # penalidade por divergência e prob baixa
        prob_penalty = 0.0
        if p_avg < min_prob:
            prob_penalty = (min_prob - p_avg) * 100.0  # em pontos

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


def risk_profile_params(profile: str) -> Dict[str, object]:
    if profile == "Conservador":
        return {"allowed": {"over_2_5", "under_2_5", "btts_yes", "btts_no"}, "w_diff": 2.0, "w_prob": 1.6, "min_prob": 0.55}
    if profile == "Agressivo":
        return {"allowed": set(MARKET_LABELS.keys()), "w_diff": 2.2, "w_prob": 0.9, "min_prob": 0.40}
    return {"allowed": set(MARKET_LABELS.keys()), "w_diff": 2.0, "w_prob": 1.2, "min_prob": 0.50}



def ev_from_odds(p: float, odds: float) -> Optional[float]:
    if not (np.isfinite(odds) and odds > 1e-9):
        return None
    return float(p) * float(odds) - 1.0


# =========================
# Upgrade 3 — Backtest + Calibração
# =========================

def _outcome_1x2(hg: int, ag: int) -> str:
    if hg > ag:
        return "H"
    if hg == ag:
        return "D"
    return "A"


def _brier_multiclass(pH: float, pD: float, pA: float, y: str) -> float:
    oH = 1.0 if y == "H" else 0.0
    oD = 1.0 if y == "D" else 0.0
    oA = 1.0 if y == "A" else 0.0
    return ((pH - oH) ** 2 + (pD - oD) ** 2 + (pA - oA) ** 2) / 3.0


def _logloss_1x2(pH: float, pD: float, pA: float, y: str, eps: float = 1e-12) -> float:
    if y == "H":
        p = max(pH, eps)
    elif y == "D":
        p = max(pD, eps)
    else:
        p = max(pA, eps)
    return -float(np.log(p))


def _brier_binary(p: float, y: int) -> float:
    return (float(p) - float(y)) ** 2


def _logloss_binary(p: float, y: int, eps: float = 1e-12) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return - (y * np.log(p) + (1 - y) * np.log(1 - p))


def _pick_matrix(lh: float, la: float, max_goals: int, use_dc: bool, rho: float) -> pd.DataFrame:
    if use_dc:
        return score_matrix_dixon_coles(lh, la, max_goals=max_goals, rho=rho)
    return score_matrix_poisson(lh, la, max_goals=max_goals)


@st.cache_data(show_spinner=False)
def run_rolling_backtest(
    matches: pd.DataFrame,
    league: str,
    n_last: int,
    max_goals: int,
    weights_by_season: Dict[str, float],
    use_dc: bool,
    rho: float,
    home_advantage: float,
    elo_k: float,
    min_hist_matches: int = 120,
    test_last_n: int = 250,
) -> pd.DataFrame:
    df = matches[matches["league"] == league].copy()
    df = df.sort_values("date_dt").reset_index(drop=True)
    if len(df) < (min_hist_matches + 20):
        raise ValueError(f"Poucos jogos na liga para backtest: {len(df)} (mínimo recomendado: {min_hist_matches + 20}).")

    start_idx = max(min_hist_matches, len(df) - test_last_n)
    rows = []

    for i in range(start_idx, len(df)):
        r = df.iloc[i]
        hist = df.iloc[:i].copy()

        home = str(r["home_team"])
        away = str(r["away_team"])

        lh, la, _ = estimate_expected_goals_poisson(
            matches=hist,
            league=league,
            home_team=home,
            away_team=away,
            n_last=n_last,
            weights_by_season=weights_by_season,
            home_advantage=home_advantage,
            elo_k=elo_k,
        )

        mat = _pick_matrix(lh, la, max_goals=max_goals, use_dc=use_dc, rho=rho)
        probs = probs_1x2_over_btts(mat)

        hg = int(r["home_goals"])
        ag = int(r["away_goals"])
        y_1x2 = _outcome_1x2(hg, ag)
        y_over25 = 1 if (hg + ag) >= 3 else 0
        y_btts = 1 if (hg >= 1 and ag >= 1) else 0

        rows.append({
            "date": str(r["date"]),
            "home": home,
            "away": away,
            "hg": hg,
            "ag": ag,
            "y_1x2": y_1x2,
            "y_over25": y_over25,
            "y_btts": y_btts,
            "p_home": float(probs["home_win"]),
            "p_draw": float(probs["draw"]),
            "p_away": float(probs["away_win"]),
            "p_over25": float(probs["over_2_5"]),
            "p_btts": float(probs["btts_yes"]),
            "home_odds": float(r.get("home_odds")) if pd.notna(r.get("home_odds")) else np.nan,
            "draw_odds": float(r.get("draw_odds")) if pd.notna(r.get("draw_odds")) else np.nan,
            "away_odds": float(r.get("away_odds")) if pd.notna(r.get("away_odds")) else np.nan,
        })

    return pd.DataFrame(rows)


def backtest_metrics(df_bt: pd.DataFrame) -> Dict[str, float]:
    if df_bt.empty:
        raise ValueError("Backtest vazio.")

    brier_1x2 = float(np.mean([
        _brier_multiclass(r.p_home, r.p_draw, r.p_away, r.y_1x2) for r in df_bt.itertuples(index=False)
    ]))
    ll_1x2 = float(np.mean([
        _logloss_1x2(r.p_home, r.p_draw, r.p_away, r.y_1x2) for r in df_bt.itertuples(index=False)
    ]))

    brier_over = float(np.mean([_brier_binary(r.p_over25, int(r.y_over25)) for r in df_bt.itertuples(index=False)]))
    ll_over = float(np.mean([_logloss_binary(r.p_over25, int(r.y_over25)) for r in df_bt.itertuples(index=False)]))

    brier_btts = float(np.mean([_brier_binary(r.p_btts, int(r.y_btts)) for r in df_bt.itertuples(index=False)]))
    ll_btts = float(np.mean([_logloss_binary(r.p_btts, int(r.y_btts)) for r in df_bt.itertuples(index=False)]))

    return {
        "brier_1x2": brier_1x2,
        "logloss_1x2": ll_1x2,
        "brier_over25": brier_over,
        "logloss_over25": ll_over,
        "brier_btts": brier_btts,
        "logloss_btts": ll_btts,
        "n_test": float(len(df_bt)),
    }


def roi_evplus_1x2(df_bt: pd.DataFrame, ev_threshold: float = 0.02, min_prob: float = 0.0) -> Dict[str, float]:
    bets = 0
    profit = 0.0

    for r in df_bt.itertuples(index=False):
        odds = {"H": r.home_odds, "D": r.draw_odds, "A": r.away_odds}
        probs = {"H": r.p_home, "D": r.p_draw, "A": r.p_away}

        if not (np.isfinite(odds["H"]) and np.isfinite(odds["D"]) and np.isfinite(odds["A"])):
            continue

        evs = {k: probs[k] * odds[k] - 1.0 for k in ["H", "D", "A"]}
        pick = max(evs, key=lambda k: evs[k])

        if evs[pick] <= ev_threshold:
            continue
        if probs[pick] < min_prob:
            continue

        bets += 1
        if r.y_1x2 == pick:
            profit += (odds[pick] - 1.0)
        else:
            profit -= 1.0

    roi = profit / bets if bets > 0 else 0.0
    return {"bets": float(bets), "profit": float(profit), "roi": float(roi)}


def plot_reliability(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, title: str = "Reliability") -> plt.Figure:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    fig, ax = plt.subplots()
    ax.plot(mean_pred, frac_pos, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("Probabilidade prevista (bin)")
    ax.set_ylabel("Frequência observada")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig


# =========================
# Sidebar
# =========================

with st.sidebar:

    st.divider()
    st.header("🚀 Modos rápidos")

    if st.button("Limpar modo (voltar ao normal)"):
        st.session_state.pop("app_mode", None)
        st.rerun()


    colm1, colm2 = st.columns(2)
    with colm1:
        if st.button("🔒 Modo Produção"):
            st.session_state["app_mode"] = "PRODUCAO"
            # Produção: estabilidade, menos apostas, menos overfit
            st.session_state["use_odds_api"] = True
            st.session_state["odds_region"] = "eu"

            st.session_state["use_recency"] = True
            st.session_state["w_current"] = 75
            st.session_state["n_last"] = 9
            st.session_state["max_goals"] = 5

            # Poisson avançado / DC
            st.session_state["use_dc"] = True
            st.session_state["auto_rho"] = True
            st.session_state["dc_rho"] = -0.06  # usado se auto_rho desligar

            # Elo dinâmico
            st.session_state["use_elo_dynamic"] = True
            st.session_state["elo_base"] = 1500
            st.session_state["elo_k"] = 14
            st.session_state["elo_hfa"] = 55
            st.session_state["elo_use_mov"] = True
            st.session_state["elo_by_league"] = True

            # ML auditor + recomendação
            st.session_state["use_ml"] = True
            st.session_state["risk_profile"] = "Conservador"

            # Backtest + calibração
            st.session_state["enable_backtest"] = True
            st.session_state["bt_test_last_n"] = 300
            st.session_state["bt_min_history"] = 200
            st.session_state["bt_bins"] = 10
            st.session_state["bt_market"] = "1X2"

            # Regras de aposta / EV
            st.session_state["ev_min_1x2"] = 0.06
            st.session_state["pmin_1x2"] = 0.50
            st.session_state["conf_min"] = 75

            st.rerun()

    with colm2:
        if st.button("🧪 Modo Laboratório"):
            st.session_state["app_mode"] = "LABORATORIO"
            # Laboratório: mais reativo (exploração)
            st.session_state["use_odds_api"] = True
            st.session_state["odds_region"] = "eu"

            st.session_state["use_recency"] = True
            st.session_state["w_current"] = 85
            st.session_state["n_last"] = 6
            st.session_state["max_goals"] = 6

            # DC ligado, porém mais “solto”
            st.session_state["use_dc"] = True
            st.session_state["auto_rho"] = True
            st.session_state["dc_rho"] = -0.06

            # Elo mais reativo
            st.session_state["use_elo_dynamic"] = True
            st.session_state["elo_base"] = 1500
            st.session_state["elo_k"] = 20
            st.session_state["elo_hfa"] = 65
            st.session_state["elo_use_mov"] = True
            st.session_state["elo_by_league"] = True

            # ML sempre ligado para auditar
            st.session_state["use_ml"] = True
            st.session_state["risk_profile"] = "Agressivo"

            # Backtest menor pra iterar rápido
            st.session_state["enable_backtest"] = True
            st.session_state["bt_test_last_n"] = 200
            st.session_state["bt_min_history"] = 120
            st.session_state["bt_bins"] = 10
            st.session_state["bt_market"] = "Over 2.5"

            # EV mais permissivo (para testar hipóteses)
            st.session_state["ev_min_1x2"] = 0.03
            st.session_state["pmin_1x2"] = 0.40
            st.session_state["conf_min"] = 65

            st.rerun()
    st.divider()
    st.header("The Odds API (odds)")

    use_odds_api = st.checkbox(
        "Usar odds (1X2) para comparar/EV",
        value=False,
        help="Lê odds do mercado 1X2 (h2h). Usa st.secrets['ODDS_API_KEY'] no Cloud."
    )

    odds_region = st.selectbox(
        "Região (bookmakers)",
        ["eu", "uk", "us", "us2", "au"],
        index=0,
        disabled=not use_odds_api
    )

    # Fallback automático de chave: secrets -> env -> input
    api_key = get_odds_api_key()
    if use_odds_api and not api_key:
        st.text_input(
            "ODDS_API_KEY (fallback local)",
            type="password",
            key="odds_api_key_input",
            help="Se não houver secret/env, você pode inserir aqui (não salva no Git)."
        )
        api_key = get_odds_api_key()

    if use_odds_api and api_key:
        q = get_quota_headers(api_key)
        if q.get("remaining") is not None:
            st.metric("Quota restante", q["remaining"])
        if q.get("used") is not None:
            st.caption(f"Usadas: {q['used']} | Última: {q.get('last')}")
        if q.get("status_code") and q["status_code"] != 200:
            st.warning(f"Quota check retornou HTTP {q['status_code']} (pode ser temporário).")

    st.header("Presets")
    preset_name = st.selectbox("Escolha um preset", list(PRESETS.keys()), index=0, key="preset_name")
    st.caption(PRESETS[preset_name]["desc"])
    if st.button("Aplicar preset agora"):
        apply_preset(preset_name)
        st.rerun()

    st.divider()
    st.header("Fonte de dados")

    url1 = st.text_input("URL CSV (Temporada atual)", value=st.session_state.get("url1", "https://www.football-data.co.uk/mmz4281/2526/E0.csv"))
    url2 = st.text_input("URL CSV (Temporada anterior - opcional)", value=st.session_state.get("url2", "https://www.football-data.co.uk/mmz4281/2425/E0.csv"))
    league_override = st.text_input("Opcional: nome da liga (override)", value=st.session_state.get("league_override", ""))

    st.divider()
    st.header("Modelo")

    use_recency = st.checkbox("Usar ponderação por recência entre temporadas",
                             value=bool(st.session_state.get("use_recency", True)),
                             key="use_recency")

    w_current = st.slider("Peso temporada atual (%)", 50, 95,
                          value=int(st.session_state.get("w_current", 70)),
                          step=1, disabled=not use_recency, key="w_current")
    st.caption(f"Peso anterior = {100 - int(w_current)}%")

    n_last = st.slider("Últimos N jogos (forma)", 5, 10,
                       value=int(st.session_state.get("n_last", 8)),
                       key="n_last")

    max_goals = st.slider("Máximo de gols na matriz", 3, 7,
                          value=int(st.session_state.get("max_goals", 5)),
                          key="max_goals")

    st.divider()
    st.subheader("Poisson avançado (Dixon–Coles)")
    use_dc = st.checkbox(
        "Ativar Dixon–Coles",
        value=bool(st.session_state.get("use_dc", True)),
        key="use_dc",
        help="Corrige o Poisson para placares baixos (0–0, 1–0, 0–1, 1–1) via parâmetro ρ."
    )

    dc_rho = st.slider(
        "ρ (rho) — ajuste low-score",
        -0.25, 0.25,
        value=float(st.session_state.get("dc_rho", -0.06)),
        step=0.01,
        disabled=not use_dc,
        key="dc_rho",
        help="Use valores levemente negativos (ex.: -0.10 a -0.02). Valores extremos podem distorcer empates/unders."
    )

    auto_rho = st.checkbox(
        "Usar preset de ρ por liga (auto)",
        value=bool(st.session_state.get("auto_rho", True)),
        key="auto_rho",
        disabled=not use_dc,
        help="Sugere um ρ padrão por liga. Você ainda pode ajustar no slider."
    )

    st.divider()
    st.header("Elo dinâmico (Upgrade 1)")
    use_dynamic_elo = st.checkbox(
        "Ativar Elo dinâmico (jogo-a-jogo)",
        value=bool(st.session_state.get("use_dynamic_elo", True)),
        key="use_dynamic_elo"
    )

    base_elo = st.slider("Elo inicial (base)", 1200, 1800,
                         value=int(st.session_state.get("base_elo", 1500)),
                         step=10, disabled=not use_dynamic_elo, key="base_elo")

    elo_k_slider = st.slider("K (reação do Elo)", 5, 60,
                             value=int(st.session_state.get("elo_k_slider", 20)),
                             step=1, disabled=not use_dynamic_elo, key="elo_k_slider")

    elo_hfa = st.slider("Vantagem de casa (HFA Elo)", 0, 120,
                        value=int(st.session_state.get("elo_hfa", 65)),
                        step=1, disabled=not use_dynamic_elo, key="elo_hfa")

    elo_use_mov = st.checkbox("Usar margem de vitória (MOV)", value=bool(st.session_state.get("elo_use_mov", True)),
                              disabled=not use_dynamic_elo, key="elo_use_mov")

    elo_per_league = st.checkbox("Elo separado por liga", value=bool(st.session_state.get("elo_per_league", True)),
                                 disabled=not use_dynamic_elo, key="elo_per_league")

    st.divider()
    st.header("ML + recomendação")
    use_ml = st.checkbox("Comparar com ML (RandomForest)", value=bool(st.session_state.get("use_ml", False)), key="use_ml")

    risk_profile = st.selectbox(
        "Perfil de risco",
        ["Conservador", "Equilibrado", "Agressivo"],
        index=1,
        key="risk_profile",
        help="Conservador evita 1X2; Equilibrado padrão; Agressivo aceita prob. menor se houver consistência."
    )

    st.divider()
    st.header("Backtest (Upgrade 3)")
    bt_enable = st.checkbox("Ativar backtest rolling", value=False, key="bt_enable")
    bt_last_n = st.slider("Testar últimos N jogos", 80, 600, value=250, step=10, key="bt_last_n")
    bt_min_hist = st.slider("Mínimo de histórico antes de testar", 80, 300, value=120, step=10, key="bt_min_hist")
    bt_bins = st.slider("Bins do gráfico de calibração", 5, 20, value=10, step=1, key="bt_bins")
    bt_market = st.selectbox("Mercado para calibração", ["Over 2.5", "BTTS Sim"], index=0, key="bt_market")
    ev_th = st.slider("EV mínimo (1X2) p/ apostar", -0.05, 0.20, value=0.02, step=0.01, key="ev_th")
    ev_min_prob = st.slider("Probabilidade mínima (1X2)", 0.0, 0.8, value=0.0, step=0.05, key="ev_min_prob")


# =========================
# Carregar dados
# =========================

if not url1.strip():
    st.error("Informe a URL da temporada atual.")
    st.stop()

try:
    df_current = load_url_normalized(url1.strip(), league_override, "CURRENT")

    df_prev = None
    has_two = bool(url2.strip())
    if has_two:
        df_prev = load_url_normalized(url2.strip(), league_override, "PREV")

    played = combine_histories(df_current, df_prev)
except Exception as e:
    st.error(f"Erro ao carregar/normalizar dados: {e}")
    st.stop()

if played.empty:
    st.error("Histórico vazio.")
    st.stop()

# Ponderação por recência
if has_two and use_recency:
    weights_by_season = {"CURRENT": int(w_current) / 100.0, "PREV": (100 - int(w_current)) / 100.0}
else:
    seasons = sorted(played["season_tag"].astype(str).unique().tolist())
    weights_by_season = {s: 1.0 for s in seasons}

# Aplicar Elo dinâmico (pré-partida)
if use_dynamic_elo:
    try:
        played = add_dynamic_elo_columns(
            played,
            base_elo=float(base_elo),
            k=float(elo_k_slider),
            hfa=float(elo_hfa),
            use_mov=bool(elo_use_mov),
            per_league=bool(elo_per_league),
        )
    except Exception as e:
        st.warning(f"Falha ao calcular Elo dinâmico. Seguindo com Elo padrão: {e}")

# Auto rho por liga
leagues = sorted(played["league"].dropna().astype(str).unique().tolist())
teams_all = sorted(set(played["home_team"].dropna().astype(str).unique().tolist()) | set(played["away_team"].dropna().astype(str).unique().tolist()))

st.info(f"Histórico: **{len(played)} jogos** | Pesos: {weights_by_season} | Elo dinâmico: {use_dynamic_elo} | Dixon–Coles: {use_dc}")

with st.expander("📄 Preview do histórico"):
    st.dataframe(played.tail(80), use_container_width=True)

# =========================
# Seleção jogo
# =========================

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

if use_dc:
    rho_suggested = DC_RHO_PRESETS.get(str(league), float(dc_rho))
    rho_used = float(rho_suggested) if auto_rho else float(dc_rho)
    if auto_rho:
        st.caption(f"ρ sugerido (preset da liga): {rho_suggested:+.2f} | ρ efetivo usado: {rho_used:+.2f}")
    else:
        st.caption(f"ρ efetivo usado (manual): {rho_used:+.2f}")
else:
    rho_used = float(dc_rho)


# Aplicar preset rho sugerido pela liga (se ligado)
if use_dc and auto_rho:
    if ("last_league_for_rho" not in st.session_state) or (st.session_state["last_league_for_rho"] != str(league)):
        st.session_state["last_league_for_rho"] = str(league)
st.divider()

# =========================
# Previsão Poisson/DC
# =========================

lam_h, lam_a, dbgP = estimate_expected_goals_poisson(
    played, league, home_team, away_team,
    n_last=int(n_last),
    weights_by_season=weights_by_season
)

if use_dc:
    matP = score_matrix_dixon_coles(lam_h, lam_a, max_goals=int(max_goals), rho=float(dc_rho))
    mode_label = f"Dixon–Coles (ρ={float(dc_rho):.2f})"
else:
    matP = score_matrix_poisson(lam_h, lam_a, max_goals=int(max_goals))
    mode_label = "Poisson"

topP, botP = list_top_bottom_scores(matP, k=5)
probsP = probs_1x2_over_btts(matP)

# Métricas topo
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("λ Mandante", f"{lam_h:.2f}")
c2.metric("λ Visitante", f"{lam_a:.2f}")
c3.metric("1 (Mandante)", f"{pct(probsP['home_win']):.1f}%")
c4.metric("X (Empate)", f"{pct(probsP['draw']):.1f}%")
c5.metric("2 (Visitante)", f"{pct(probsP['away_win']):.1f}%")
c6.metric("Over 2.5", f"{pct(probsP['over_2_5']):.1f}%")

c7, c8, c9, c10 = st.columns(4)
c7.metric("Under 2.5", f"{pct(probsP['under_2_5']):.1f}%")
c8.metric("BTTS (Sim)", f"{pct(probsP['btts_yes']):.1f}%")
c9.metric("BTTS (Não)", f"{pct(probsP['btts_no']):.1f}%")
c10.metric("Total gols (liga, médio)", f"{league_goal_averages(played, league, weights_by_season)['avg_total_goals']:.2f}")

# Heatmap + placares
left, right = st.columns([1.2, 1])
with left:
    st.subheader(f"Matriz de placares — {mode_label}")
    fig = heatmap_figure(matP * 100.0, f"Probabilidade (%) por placar — {mode_label}")
    st.pyplot(fig, clear_figure=True)

with right:
    st.subheader("Placares mais / menos prováveis")
    st.markdown("**Top 5 mais prováveis**")
    t = topP.copy()
    t["prob_%"] = t["prob_%"].round(3)
    st.dataframe(t[["placar", "prob_%"]].style.format({"prob_%": "{:.3f}%"}).background_gradient(subset=["prob_%"]), use_container_width=True)

    st.markdown("**Top 5 menos prováveis**")
    b = botP.copy()
    b["prob_%"] = b["prob_%"].round(6)
    st.dataframe(b[["placar", "prob_%"]].style.format({"prob_%": "{:.6f}%"}), use_container_width=True)

# Comparação low-scores se Dixon–Coles ligado
if use_dc:
    mat_plain = score_matrix_poisson(lam_h, lam_a, max_goals=int(max_goals))
    rows = []
    for (hg, ag) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        if hg in mat_plain.index and ag in mat_plain.columns:
            rows.append({
                "Placar": f"{hg}x{ag}",
                "Poisson (%)": 100.0 * float(mat_plain.loc[hg, ag]),
                "Dixon–Coles (%)": 100.0 * float(matP.loc[hg, ag]),
                "Delta (p.p.)": 100.0 * float(matP.loc[hg, ag] - mat_plain.loc[hg, ag]),
            })
    with st.expander("📌 Efeito do Dixon–Coles nos placares baixos (0–0/1–0/0–1/1–1)"):
        df_ls = pd.DataFrame(rows)
        st.dataframe(df_ls.style.format({"Poisson (%)": "{:.3f}", "Dixon–Coles (%)": "{:.3f}", "Delta (p.p.)": "{:+.3f}"}), use_container_width=True)
        st.caption("Se o delta ficar exagerado, reduza |ρ| no slider.")

with st.expander("🔎 Detalhes do cálculo (Poisson/DC)"):
    dbg = dict(dbgP)
    dbg["mode"] = mode_label
    st.json(dbg)

st.divider()

# =========================
# Odds/EV (do CSV) — se existirem
# =========================

st.subheader("💰 EV (Valor Esperado) — usando odds do CSV (se houver)")
st.caption("EV = p_modelo * odds - 1. EV > 0 sugere valor (aposta com expectativa positiva).")

# Tenta achar odds do jogo selecionado (mesma liga + times) no histórico mais recente (apenas para demonstrar)
df_match_odds = played[(played["league"] == league) & (played["home_team"] == home_team) & (played["away_team"] == away_team)].copy()
df_match_odds = df_match_odds.sort_values("date_dt").tail(1)
if len(df_match_odds) == 0:
    st.info("Não encontrei uma linha desse confronto no histórico para exibir odds. (Normal em alguns datasets).")
else:
    r = df_match_odds.iloc[0]
    oh, od, oa = r.get("home_odds", np.nan), r.get("draw_odds", np.nan), r.get("away_odds", np.nan)
    ev_h = ev_from_odds(probsP["home_win"], float(oh)) if np.isfinite(oh) else None
    ev_d = ev_from_odds(probsP["draw"], float(od)) if np.isfinite(od) else None
    ev_a = ev_from_odds(probsP["away_win"], float(oa)) if np.isfinite(oa) else None

    ev_table = pd.DataFrame([
        {"Mercado": "1 (Mandante)", "Odds": oh, "Prob (modelo)": probsP["home_win"], "EV": ev_h},
        {"Mercado": "X (Empate)", "Odds": od, "Prob (modelo)": probsP["draw"], "EV": ev_d},
        {"Mercado": "2 (Visitante)", "Odds": oa, "Prob (modelo)": probsP["away_win"], "EV": ev_a},
    ])

    st.dataframe(
        ev_table.style.format({
            "Odds": "{:.2f}",
            "Prob (modelo)": "{:.3f}",
            "EV": "{:+.3f}",
        }),
        use_container_width=True
    )

st.divider()

# =========================
# ML + Confiança + Recomendação
# =========================

if use_ml:
    try:
        with st.spinner("Treinando ML (RandomForest)..."):
            trained = train_ml_models_cached(played, n_last=int(n_last), weights_by_season=weights_by_season)

        lam_h_ml, lam_a_ml, dbgM = predict_expected_goals_ml(
            played, league, home_team, away_team,
            trained=trained, n_last=int(n_last), weights_by_season=weights_by_season
        )

        matM = score_matrix_poisson(lam_h_ml, lam_a_ml, max_goals=int(max_goals))
        probsM = probs_1x2_over_btts(matM)

        score, diffs_pp = confidence_score_from_models(probsP, probsM)
        label = confidence_label(score)
        recs = recommended_markets(probsP, probsM, diffs_pp, profile=str(risk_profile))

        st.subheader("🤖 Comparação — Poisson/DC vs ML + Confiança")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("λ Mandante (ML)", f"{lam_h_ml:.2f}")
        c2.metric("λ Visitante (ML)", f"{lam_a_ml:.2f}")
        c3.metric("MAE Home", f"{dbgM['mae_home']:.3f}")
        c4.metric("MAE Away", f"{dbgM['mae_away']:.3f}")
        c5.metric("Confiança", f"{score}/100")
        c6.metric("Nível", label)

        st.markdown("### 🎯 Mercado recomendado (automático)")
        top1 = recs[0]
        st.info(
            f"**Top 1:** {top1['mercado']} — divergência **{top1['diff_pp']:.2f} p.p.** | "
            f"prob. média (Poisson+ML) **{top1['prob_avg']*100:.1f}%** | nível: **{top1['nivel']}**"
        )
        if len(recs) > 1:
            alt_str = " | ".join([f"{r['mercado']} ({r['diff_pp']:.1f} p.p.)" for r in recs[1:]])
            st.caption(f"Alternativas: {alt_str}")

        with st.expander("🔎 Detalhes do ML"):
            st.json(dbgM)

    except Exception as e:
        st.error(f"Falha ao treinar/rodar ML: {e}")

st.divider()

# =========================
# Upgrade 3 — Backtest + Calibração + ROI EV+
# =========================

if bt_enable:
    st.subheader("🧪 Backtest rolling + Calibração (Upgrade 3)")
    try:
        # parâmetros do modelo
        use_dc_bt = bool(use_dc)
        rho_bt = float(dc_rho) if use_dc_bt else 0.0
        home_adv_bt = 0.12
        elo_k_bt = 0.10

        with st.spinner("Rodando backtest rolling..."):
            df_bt = run_rolling_backtest(
                matches=played,
                league=league,
                n_last=int(n_last),
                max_goals=int(max_goals),
                weights_by_season=weights_by_season,
                use_dc=use_dc_bt,
                rho=rho_bt,
                home_advantage=float(home_adv_bt),
                elo_k=float(elo_k_bt),
                min_hist_matches=int(bt_min_hist),
                test_last_n=int(bt_last_n),
            )

        mets = backtest_metrics(df_bt)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Brier (1X2)", f"{mets['brier_1x2']:.4f}")
        c2.metric("LogLoss (1X2)", f"{mets['logloss_1x2']:.4f}")
        c3.metric("Brier (Over2.5)", f"{mets['brier_over25']:.4f}")
        c4.metric("Brier (BTTS)", f"{mets['brier_btts']:.4f}")

        st.caption("Brier menor = melhor calibração. LogLoss menor = melhor probabilidade (penaliza erros confiantes).")

        roi = roi_evplus_1x2(df_bt, ev_threshold=float(ev_th), min_prob=float(ev_min_prob))
        c5, c6, c7 = st.columns(3)
        c5.metric("Apostas (EV+)", f"{int(roi['bets'])}")
        c6.metric("Lucro (unid.)", f"{roi['profit']:.2f}")
        c7.metric("ROI", f"{roi['roi']*100:.2f}%")

        # Reliability curve
        if bt_market == "Over 2.5":
            y_true = df_bt["y_over25"].astype(int).values
            y_prob = df_bt["p_over25"].astype(float).values
            title = "Calibração — Over 2.5"
        else:
            y_true = df_bt["y_btts"].astype(int).values
            y_prob = df_bt["p_btts"].astype(float).values
            title = "Calibração — BTTS (Sim)"

        st.markdown("### 📈 Gráfico de calibração (Reliability curve)")
        fig_cal = plot_reliability(y_true=y_true, y_prob=y_prob, n_bins=int(bt_bins), title=title)
        st.pyplot(fig_cal, clear_figure=True)

        with st.expander("📄 Ver amostra do backtest"):
            st.dataframe(df_bt.tail(80), use_container_width=True)

    except Exception as e:
        st.error(f"Backtest/Calibração falhou: {e}")

st.caption("Pronto. Se quiser, o próximo passo é colocar sua API key da Odds API em st.secrets para deploy seguro.")


# =========================
# Upgrade 5 — Decisão de Produção (log)
# =========================

def production_decision(
    recs: list,
    is_prod: bool,
    confidence_score: int | None,
    odds_available: bool,
    ev_table: "pd.DataFrame | None" = None,
) -> tuple[str, list]:
    """
    Retorna: ("APOSTAR" ou "NÃO APOSTAR", motivos[])
    - Em produção: recs vazio => NÃO APOSTAR
    - Exige odds para EV (se odds não tiver, não aposta)
    - Exige confiança mínima se disponível
    - Exige EV mínimo e prob mínima no mercado 1X2 se ev_table existir
    """
    reasons = []
    if not is_prod:
        return "LAB", ["Modo Laboratório: sem bloqueio estrito."]

    rules = production_rules_snapshot()
    ev_min = rules["EV mínimo 1X2"]
    pmin = rules["Prob mínima 1X2"]
    conf_min = rules["Confiança mínima"]

    if not recs:
        reasons.append("Sem mercado seguro (todos vermelhos ou filtrados).")

    if confidence_score is not None and confidence_score < conf_min:
        reasons.append(f"Confiança abaixo do mínimo ({confidence_score} < {conf_min}).")

    if not odds_available:
        reasons.append("Odds não disponíveis (não dá para validar EV).")

    # Se existir tabela EV (1X2), checa se há algum EV>=min e prob>=pmin
    if ev_table is not None and len(ev_table) > 0:
        ok = False
        for _, r in ev_table.iterrows():
            try:
                ev = float(r.get("EV", None))
                prob = float(r.get("Prob (modelo)", None))
            except Exception:
                continue
            if ev >= ev_min and prob >= pmin:
                ok = True
                break
        if not ok:
            reasons.append(f"Nenhuma opção 1X2 com EV≥{ev_min:.2f} e Prob≥{pmin:.2f}.")

    if reasons:
        return "NÃO APOSTAR", reasons
    return "APOSTAR", ["Passou nos filtros do Modo Produção."]

