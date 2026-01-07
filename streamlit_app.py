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
✅ NOVO (EV + Odds):
- EV (Valor Esperado) por mercado com odds manual ou automáticas
- Integração The Odds API (região eu) para 1X2, O/U 2.5, BTTS
- Seleção de bookmaker ou “Best price”

Rodar:
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import re
import unicodedata
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import poisson

import requests

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

    # (placeholder) Elo; você pode plugar Elo real depois
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
# Odds API (The Odds API)
# =========================

LEAGUE_TO_ODDSAPI_SPORT = {
    "Premier League": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue 1": "soccer_france_ligue_one",
    "Brasileirão": "soccer_brazil_campeonato",
    "Brazil Serie A": "soccer_brazil_campeonato",
    "Campeonato Brasileiro": "soccer_brazil_campeonato",
    "Champions League": "soccer_uefa_champs_league",
    "UEFA Champions League": "soccer_uefa_champs_league",
}

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def norm_team(s: str) -> str:
    s0 = _strip_accents(str(s).lower()).strip()
    s0 = re.sub(r"[\.\,\-\(\)\[\]\/\&\+]", " ", s0)
    s0 = re.sub(r"\b(fc|sc|cf|ac|afc|cfc|cd|ud|sv|tsv|bk|fk|ec|cr|atletico|athletic)\b", " ", s0)
    s0 = re.sub(r"\b(de|da|do|das|dos|the)\b", " ", s0)
    s0 = re.sub(r"\s+", " ", s0).strip()
    return s0

def teams_match_score(home_a: str, away_a: str, home_b: str, away_b: str) -> float:
    """
    Score de match entre (A=app) e (B=api).
    1.0 = perfeito; maior melhor.
    """
    ha, aa = norm_team(home_a), norm_team(away_a)
    hb, ab = norm_team(home_b), norm_team(away_b)

    # match direto
    if ha == hb and aa == ab:
        return 1.0

    # sets iguais (pouco provável mas ajuda em inversões)
    seta = {ha, aa}
    setb = {hb, ab}
    if seta == setb:
        return 0.75

    # token overlap
    def tok(s: str) -> set:
        return set(s.split())

    inter = len(tok(ha) & tok(hb)) + len(tok(aa) & tok(ab))
    denom = max(1, len(tok(ha)) + len(tok(hb)) + len(tok(aa)) + len(tok(ab)))
    return inter / denom

@st.cache_data(show_spinner=False, ttl=300)
def oddsapi_list_soccer_sports(api_key: str) -> List[Dict[str, str]]:
    url = "https://api.the-odds-api.com/v4/sports"
    r = requests.get(url, params={"apiKey": api_key}, timeout=20)
    r.raise_for_status()
    data = r.json()
    soccer = []
    for item in data:
        if str(item.get("key", "")).startswith("soccer_"):
            soccer.append({"key": item.get("key", ""), "title": item.get("title", "")})
    return sorted(soccer, key=lambda x: x["title"])

@st.cache_data(show_spinner=False, ttl=300)
def oddsapi_fetch_odds(api_key: str, sport_key: str, regions: str, markets: str, odds_format: str = "decimal") -> List[dict]:
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": "iso",
    }
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    return r.json()

def _pick_bookmaker(books: List[dict], preference: str) -> List[dict]:
    """
    preference:
      - "Best price" => usar todas para escolher melhor odd por outcome
      - ou nome exato de bookmaker (ex.: "pinnacle", "bet365"... dependendo do catálogo)
    """
    if preference == "Best price":
        return books
    pref = preference.strip().lower()
    chosen = [b for b in books if str(b.get("key", "")).lower() == pref or str(b.get("title", "")).lower() == pref]
    return chosen if chosen else books

def _extract_market_prices(bookmakers: List[dict], market_key: str) -> List[dict]:
    out = []
    for b in bookmakers:
        for m in b.get("markets", []) or []:
            if str(m.get("key", "")).lower() == market_key.lower():
                out.append({"book": b, "market": m})
    return out

def oddsapi_best_price_h2h(markets_blocks: List[dict], home_name_api: str, away_name_api: str) -> Dict[str, Optional[float]]:
    best = {"home": None, "draw": None, "away": None}
    for blk in markets_blocks:
        outcomes = blk["market"].get("outcomes", []) or []
        for o in outcomes:
            name = str(o.get("name", ""))
            price = o.get("price", None)
            if price is None:
                continue
            price = float(price)

            n = norm_team(name)
            if n == norm_team(home_name_api):
                best["home"] = price if best["home"] is None else max(best["home"], price)
            elif n == norm_team(away_name_api):
                best["away"] = price if best["away"] is None else max(best["away"], price)
            elif n == "draw":
                best["draw"] = price if best["draw"] is None else max(best["draw"], price)
    return best

def oddsapi_best_price_totals_25(markets_blocks: List[dict]) -> Dict[str, Optional[float]]:
    # retorna Over/Under 2.5
    best = {"over_2_5": None, "under_2_5": None}
    for blk in markets_blocks:
        outcomes = blk["market"].get("outcomes", []) or []
        for o in outcomes:
            name = str(o.get("name", "")).lower()
            point = o.get("point", None)
            price = o.get("price", None)
            if point is None or price is None:
                continue
            try:
                point = float(point)
            except Exception:
                continue
            if abs(point - 2.5) > 1e-9:
                continue
            price = float(price)
            if "over" in name:
                best["over_2_5"] = price if best["over_2_5"] is None else max(best["over_2_5"], price)
            if "under" in name:
                best["under_2_5"] = price if best["under_2_5"] is None else max(best["under_2_5"], price)
    return best

def oddsapi_best_price_btts(markets_blocks: List[dict]) -> Dict[str, Optional[float]]:
    best = {"btts_yes": None, "btts_no": None}
    for blk in markets_blocks:
        outcomes = blk["market"].get("outcomes", []) or []
        for o in outcomes:
            name = str(o.get("name", "")).lower()
            price = o.get("price", None)
            if price is None:
                continue
            price = float(price)
            if name in {"yes", "y"}:
                best["btts_yes"] = price if best["btts_yes"] is None else max(best["btts_yes"], price)
            if name in {"no", "n"}:
                best["btts_no"] = price if best["btts_no"] is None else max(best["btts_no"], price)
    return best

def oddsapi_get_odds_for_fixture(
    api_key: str,
    sport_key: str,
    regions: str,
    bookmaker_pref: str,
    home_team_app: str,
    away_team_app: str,
) -> Dict[str, Optional[float]]:
    """
    Busca odds 1X2, O/U 2.5, BTTS para o jogo mais próximo por nomes.
    Retorna dict com chaves:
      home_win, draw, away_win, over_2_5, under_2_5, btts_yes, btts_no
    """
    markets = "h2h,totals,btts"
    events = oddsapi_fetch_odds(api_key, sport_key=sport_key, regions=regions, markets=markets)

    if not isinstance(events, list) or len(events) == 0:
        return {k: None for k in ["home_win","draw","away_win","over_2_5","under_2_5","btts_yes","btts_no"]}

    # achar melhor match por times
    best_event = None
    best_score = -1.0
    for ev in events:
        h_api = ev.get("home_team", "")
        a_api = ev.get("away_team", "")
        sc = teams_match_score(home_team_app, away_team_app, h_api, a_api)
        if sc > best_score:
            best_score = sc
            best_event = ev

    if best_event is None or best_score < 0.25:
        return {k: None for k in ["home_win","draw","away_win","over_2_5","under_2_5","btts_yes","btts_no"]}

    books = best_event.get("bookmakers", []) or []
    books = _pick_bookmaker(books, bookmaker_pref)

    # h2h
    h2h_blocks = _extract_market_prices(books, "h2h")
    best_h2h = oddsapi_best_price_h2h(h2h_blocks, best_event.get("home_team",""), best_event.get("away_team",""))

    # totals
    totals_blocks = _extract_market_prices(books, "totals")
    best_tot = oddsapi_best_price_totals_25(totals_blocks)

    # btts
    btts_blocks = _extract_market_prices(books, "btts")
    best_btts = oddsapi_best_price_btts(btts_blocks)

    return {
        "home_win": best_h2h["home"],
        "draw": best_h2h["draw"],
        "away_win": best_h2h["away"],
        "over_2_5": best_tot["over_2_5"],
        "under_2_5": best_tot["under_2_5"],
        "btts_yes": best_btts["btts_yes"],
        "btts_no": best_btts["btts_no"],
    }

def oddsapi_available_bookmakers(events: List[dict]) -> List[str]:
    keys = set()
    titles = set()
    for ev in events or []:
        for b in (ev.get("bookmakers") or []):
            if b.get("key"):
                keys.add(str(b.get("key")).lower())
            if b.get("title"):
                titles.add(str(b.get("title")).lower())
    out = sorted(keys.union(titles))
    return out



# =========================
# Elo dinâmico (Upgrade 1) — ELO_DINAMICO_V1
# =========================

def _elo_expected_score(r_home: float, r_away: float, hfa: float) -> float:
    """
    Expectativa do mandante (0-1) considerando Home Field Advantage (hfa em pontos Elo).
    """
    return 1.0 / (1.0 + 10 ** (((r_away) - (r_home + hfa)) / 400.0))

def _elo_mov_multiplier(goal_diff: int, elo_diff: float) -> float:
    """
    Multiplicador por margem de vitória (heurística comum).
    goal_diff: |HG-AG|
    elo_diff: (R_home+HFA) - R_away
    """
    if goal_diff <= 1:
        return 1.0
    # fórmula inspirada em variações clássicas de Elo no futebol
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
    """
    Preenche home_elo e away_elo PRE-MATCH, calculando Elo jogo-a-jogo no histórico.

    - base_elo: Elo inicial por time
    - k: fator de atualização (quanto maior, mais reage)
    - hfa: vantagem de casa em pontos Elo
    - use_mov: aplica multiplicador por margem de vitória
    - per_league: Elo separado por liga (recomendado)
    """
    df = matches.copy()
    if "date_dt" not in df.columns:
        raise ValueError("date_dt não encontrado; normalize_matches_dataframe deveria criar essa coluna.")

    df = df.sort_values(["date_dt", "league", "home_team", "away_team"]).reset_index(drop=True)

    # rating store
    ratings: dict = {}  # key -> rating

    def key_for(league: str, team: str) -> str:
        return f"{league}||{team}" if per_league else team

    home_elos = []
    away_elos = []

    for _, r in df.iterrows():
        league = str(r["league"])
        home = str(r["home_team"])
        away = str(r["away_team"])

        kh = key_for(league, home)
        ka = key_for(league, away)

        r_home = float(ratings.get(kh, base_elo))
        r_away = float(ratings.get(ka, base_elo))

        # Elo pré-jogo (o que interessa para features/modelo)
        home_elos.append(r_home)
        away_elos.append(r_away)

        # resultado
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

# =========================
# Dixon–Coles (Upgrade 2) — DIXON_COLES_V1
# =========================

def dixon_coles_tau(hg: int, ag: int, lam_h: float, lam_a: float, rho: float) -> float:
    """
    Fator de correção τ(hg,ag) de Dixon–Coles para low-scores.
    Fórmulas clássicas:
      τ00 = 1 - (λh*λa*rho)
      τ01 = 1 + (λh*rho)
      τ10 = 1 + (λa*rho)
      τ11 = 1 - rho
    Para outros placares, τ = 1.
    """
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
    """
    Matriz de placares usando Poisson * τ (Dixon–Coles) nos low-scores.
    Observação: como truncamos em 0..max_goals, normalizamos a soma para 1.
    """
    hs = np.arange(0, max_goals + 1)
    as_ = np.arange(0, max_goals + 1)

    p_home = poisson.pmf(hs, mu=lambda_home)
    p_away = poisson.pmf(as_, mu=lambda_away)

    mat = np.outer(p_home, p_away)

    # aplica τ nos low scores (0/1)
    for hg in [0, 1]:
        for ag in [0, 1]:
            if hg <= max_goals and ag <= max_goals:
                mat[hg, ag] = mat[hg, ag] * dixon_coles_tau(hg, ag, lambda_home, lambda_away, rho)

    # normaliza (por truncamento)
    s = float(mat.sum())
    if s > 0:
        mat = mat / s

    return pd.DataFrame(mat, index=hs, columns=as_)


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
    if profile == "Conservador":
        return {"allowed": {"over_2_5", "under_2_5", "btts_yes", "btts_no"}, "w_diff": 2.0, "w_prob": 1.6, "min_prob": 0.55}
    if profile == "Agressivo":
        return {"allowed": set(MARKET_LABELS.keys()), "w_diff": 2.2, "w_prob": 0.9, "min_prob": 0.40}
    return {"allowed": set(MARKET_LABELS.keys()), "w_diff": 2.0, "w_prob": 1.2, "min_prob": 0.50}

def recommended_markets(probsP: Dict[str, float], probsM: Dict[str, float], diffs_pp: Dict[str, float], profile: str) -> List[Dict[str, object]]:
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

        prob_penalty = 0.0
        if p_avg < min_prob:
            prob_penalty = (min_prob - p_avg) * 100.0

        lvl_score = {"verde": 0.0, "amarelo": 7.0, "vermelho": 18.0}[lvl]
        score = (w_diff * diff) + (w_prob * (100.0 * (1.0 - p_avg))) + lvl_score + prob_penalty

        items.append({"key": k, "mercado": label, "diff_pp": diff, "prob_avg": float(p_avg), "nivel": lvl, "rank_score": float(score)})

    return sorted(items, key=lambda x: x["rank_score"])[:3]

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
# EV (Valor Esperado)
# =========================

def fair_odds(p: float) -> Optional[float]:
    if p <= 0:
        return None
    return 1.0 / p

def expected_value(p: float, odd: float) -> float:
    return (p * odd) - 1.0

def clamp_prob(p: float) -> float:
    return float(np.clip(p, 1e-9, 1.0 - 1e-9))


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
# Sidebar — Fonte + Presets + Modelo + Perfil de risco + Odds
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

    st.divider()
    st.subheader("Poisson avançado (Dixon–Coles)")
    use_dc = st.checkbox(
        "Ativar Dixon–Coles (corrige placares baixos / empates)",
        value=bool(st.session_state.get("use_dc", True)),
        key="use_dc",
        help="Ajusta 0-0, 1-0, 0-1, 1-1 via parâmetro ρ. Bom para ligas com jogos truncados/unders."
    )

    dc_rho = st.slider(
        "ρ (rho) — intensidade do ajuste low-score",
        -0.25, 0.25,
        value=float(st.session_state.get("dc_rho", -0.06)),
        step=0.01,
        disabled=not use_dc,
        key="dc_rho",
        help="Valores típicos ficam levemente negativos (ex.: -0.10 a -0.02). Se exagerar, distorce empates/unders."
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

    st.divider()
    st.header("Odds (The Odds API)")
    odds_source = st.radio("Fonte de odds", ["Manual", "The Odds API"], index=1, horizontal=True)

    odds_api_key = st.text_input(
        "API Key (não publique em repo)",
        value=st.session_state.get("odds_api_key", ""),
        type="password",
        help="Dica: no Streamlit Cloud use Secrets para guardar a chave."
    )
    st.session_state["odds_api_key"] = odds_api_key

    odds_region = "eu"  # fixo conforme seu pedido

    auto_sport_guess = "soccer_epl"
    # vamos escolher um default mais provável; depois do filtro de liga, o app ajusta
    sport_key_override = st.text_input(
        "Sport key (opcional)",
        value=st.session_state.get("sport_key_override", ""),
        help="Se vazio, tenta mapear pela liga (EPL, LaLiga, etc.)."
    )
    st.session_state["sport_key_override"] = sport_key_override

    bookmaker_pref = st.text_input(
        "Bookmaker (opcional)",
        value=st.session_state.get("bookmaker_pref", "Best price"),
        help="Use 'Best price' para pegar o melhor preço entre bookmakers, ou digite o nome/slug (ex.: pinnacle, bet365 se existir na API/region)."
    )
    st.session_state["bookmaker_pref"] = bookmaker_pref

    st.divider()
    st.header("Elo dinâmico (Upgrade 1)")
    use_dynamic_elo = st.checkbox(
        "Ativar Elo dinâmico (jogo-a-jogo)",
        value=bool(st.session_state.get("use_dynamic_elo", True)),
        key="use_dynamic_elo",
        help="Se ligado, calcula Elo no histórico e usa como força relativa no Poisson e ML."
    )

    base_elo = st.slider(
        "Elo inicial (base)",
        1200, 1800,
        value=int(st.session_state.get("base_elo", 1500)),
        step=10,
        disabled=not use_dynamic_elo,
        key="base_elo"
    )

    elo_k = st.slider(
        "K (reação do Elo)",
        5, 60,
        value=int(st.session_state.get("elo_k", 20)),
        step=1,
        disabled=not use_dynamic_elo,
        key="elo_k"
    )

    elo_hfa = st.slider(
        "Vantagem de casa (HFA em pontos Elo)",
        0, 120,
        value=int(st.session_state.get("elo_hfa", 65)),
        step=1,
        disabled=not use_dynamic_elo,
        key="elo_hfa"
    )

    elo_use_mov = st.checkbox(
        "Usar margem de vitória (MOV)",
        value=bool(st.session_state.get("elo_use_mov", True)),
        disabled=not use_dynamic_elo,
        key="elo_use_mov",
        help="Se ligado, vitórias por mais gols atualizam um pouco mais o Elo."
    )

    elo_per_league = st.checkbox(
        "Elo separado por liga",
        value=bool(st.session_state.get("elo_per_league", True)),
        disabled=not use_dynamic_elo,
        key="elo_per_league",
        help="Recomendado: evita 'misturar' força entre ligas diferentes."
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


# =========================
# Aplicar Elo dinâmico no histórico (antes do modelo)
# =========================

if bool(st.session_state.get("use_dynamic_elo", True)):
    try:
        played = add_dynamic_elo_columns(
            played,
            base_elo=float(st.session_state.get("base_elo", 1500)),
            k=float(st.session_state.get("elo_k", 20)),
            hfa=float(st.session_state.get("elo_hfa", 65)),
            use_mov=bool(st.session_state.get("elo_use_mov", True)),
            per_league=bool(st.session_state.get("elo_per_league", True)),
        )
    except Exception as _e:
        st.warning(f"Falha ao calcular Elo dinâmico (seguindo com Elo padrão do dataset): {_e}")

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


# Matriz de placares (Poisson clássico ou Dixon–Coles)
if bool(st.session_state.get("use_dc", True)):
    rho = float(st.session_state.get("dc_rho", -0.06))
    matP = score_matrix_dixon_coles(lam_h, lam_a, max_goals=max_goals, rho=rho)
else:
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
    
mode_label = "Dixon–Coles" if bool(st.session_state.get("use_dc", True)) else "Poisson"
fig = heatmap_figure(matP * 100.0, f"Probabilidade (%) por placar ({mode_label})")
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

with st.expander("🔎 Detalhes do cálculo (Poisson/Dixon–Coles)"):
    dbgP2 = dict(dbgP)
    dbgP2["poisson_mode"] = "Dixon–Coles" if bool(st.session_state.get("use_dc", True)) else "Poisson"
    dbgP2["dixon_coles_rho"] = float(st.session_state.get("dc_rho", -0.06)) if bool(st.session_state.get("use_dc", True)) else None
    st.json(dbgP2)

st.divider()


# =========================
# Final probabilities (para EV): por padrão Poisson
# =========================

probs_final = probsP.copy()
probs_final_label = "Poisson"
confidence_score_final: Optional[int] = None
confidence_label_final: Optional[str] = None

# odds automáticas (The Odds API)
odds_auto: Dict[str, Optional[float]] = {k: None for k in MARKET_LABELS.keys()}
sport_key_auto = (sport_key_override.strip() if sport_key_override.strip() else LEAGUE_TO_ODDSAPI_SPORT.get(league, "soccer_epl"))

if odds_source == "The Odds API":
    if not odds_api_key.strip():
        st.warning("Odds API selecionada, mas a API Key está vazia. Preencha no sidebar.")
    else:
        try:
            with st.spinner("Buscando odds na The Odds API..."):
                odds_auto = oddsapi_get_odds_for_fixture(
                    api_key=odds_api_key.strip(),
                    sport_key=sport_key_auto,
                    regions=odds_region,
                    bookmaker_pref=bookmaker_pref.strip() if bookmaker_pref.strip() else "Best price",
                    home_team_app=home_team,
                    away_team_app=away_team,
                )
            found_any = any(v is not None for v in odds_auto.values())
            if not found_any:
                st.info("Não encontrei odds automáticas (ou o match do jogo ficou fraco). Use Manual ou ajuste o Sport key/bookmaker.")
        except Exception as e:
            st.warning(f"Falha ao buscar odds na The Odds API: {e}")


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

        # ✅ Atualiza probs finais para EV: média Poisson + ML
        probs_final = {k: 0.5 * (probsP[k] + probsM[k]) for k in MARKET_LABELS.keys()}
        probs_final_label = "Média (Poisson + ML)"
        confidence_score_final = int(score)
        confidence_label_final = str(label)

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


# =========================
# EV (Valor Esperado) — sempre disponível
# =========================

st.divider()
st.subheader("💰 EV — Expectativa de Valor (EV+ / EV-)")

st.caption(
    f"Probabilidade usada: **{probs_final_label}**. "
    + (f"Confiança: **{confidence_score_final}/100 ({confidence_label_final})**." if confidence_score_final is not None else "Confiança: (ML desligado/indisponível).")
)

# Mostrar odds automáticas encontradas (se houver)
with st.expander("📌 Odds automáticas detectadas (The Odds API)"):
    if odds_source != "The Odds API":
        st.info("Fonte de odds está em Manual.")
    else:
        st.write(f"Sport key usado: **{sport_key_auto}** | Região: **{odds_region}** | Bookmaker: **{bookmaker_pref}**")
        df_auto = pd.DataFrame([{"Mercado": MARKET_LABELS[k], "Odd": odds_auto.get(k, None)} for k in MARKET_LABELS.keys()])
        st.dataframe(df_auto, use_container_width=True)

market_key = st.selectbox(
    "Mercado para calcular EV",
    list(MARKET_LABELS.keys()),
    format_func=lambda k: MARKET_LABELS[k],
    index=0
)

p_model = clamp_prob(float(probs_final[market_key]))
fair = fair_odds(p_model)

# Odds: preferir automática se veio da API e se o usuário escolheu API
odd_auto_value = odds_auto.get(market_key, None) if odds_source == "The Odds API" else None

odds_mode = st.radio(
    "Fonte da odd",
    ["Automática (The Odds API)", "Manual"] if odd_auto_value is not None else ["Manual"],
    index=0 if odd_auto_value is not None else 0,
    horizontal=True
)

odd_used: float
if odds_mode.startswith("Automática") and odd_auto_value is not None:
    odd_used = float(odd_auto_value)
    st.info(f"Odd automática: **{odd_used:.2f}**")
else:
    odd_used = float(st.number_input("Odd decimal (ex.: 1.85)", min_value=1.01, max_value=100.0, value=1.85, step=0.01))

ev = expected_value(p_model, odd_used)
ev_pct = ev * 100.0

ev_adj = ev
if confidence_score_final is not None:
    ev_adj = ev * (confidence_score_final / 100.0)

ev_adj_pct = ev_adj * 100.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Prob. do modelo", f"{p_model*100:.2f}%")
c2.metric("Odd justa (1/p)", f"{fair:.2f}" if fair is not None else "—")
c3.metric("EV (%)", f"{ev_pct:+.2f}%")
c4.metric("EV ajustado confiança (%)", f"{ev_adj_pct:+.2f}%" if confidence_score_final is not None else "—")

if ev > 0:
    st.success("✅ **EV+ (Valor Esperado Positivo)** — tendência favorável no longo prazo (se o modelo estiver bem calibrado).")
else:
    st.error("❌ **EV- (Valor Esperado Negativo)** — a odd não compensa a probabilidade estimada pelo modelo.")

with st.expander("ℹ️ Como interpretar o EV"):
    st.markdown(
        """
- Fórmula: **EV = p × odd − 1**
- **EV > 0** → Expectativa Positiva (EV+)  
- **EV < 0** → Expectativa Negativa (EV-)  
- **Odd justa**: **1/p** (se o mercado paga acima disso, tende a ser EV+)
- Quando ML está ligado, mostramos também **EV ajustado**: **EV × (confiança/100)** (heurística de risco).
"""
    )

st.caption("Dica: perfil de risco muda o ranking; alerta de extremos avisa quando λ está fora do padrão esperado.")
