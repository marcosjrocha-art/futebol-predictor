from __future__ import annotations

import os
import requests
import streamlit as st
from typing import Dict, Any, Optional, Tuple

ODDS_API_HOST = "https://api.the-odds-api.com"

# Mapeamento: suas ligas do CSV -> sport_key da The Odds API (soccer)
LEAGUE_TO_SPORT_KEY = {
    "Premier League": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue 1": "soccer_france_ligue_one",
    "Champions League": "soccer_uefa_champs_league",
    "Brasileirão": "soccer_brazil_campeonato",
    "Brazil Série A": "soccer_brazil_campeonato",
    "Campeonato Brasileiro": "soccer_brazil_campeonato",
}

def _get_secret(name: str) -> Optional[str]:
    # st.secrets pode não ter a chave localmente; tratamos com segurança
    try:
        v = st.secrets.get(name, None)
        if v:
            return str(v).strip()
    except Exception:
        pass
    return None

def get_odds_api_key() -> Optional[str]:
    """
    Prioridade:
      1) st.secrets["ODDS_API_KEY"]
      2) env var ODDS_API_KEY
      3) st.session_state["odds_api_key_input"] (input do usuário)
    """
    v = _get_secret("ODDS_API_KEY")
    if v:
        return v

    v = os.getenv("ODDS_API_KEY")
    if v:
        return v.strip()

    v = st.session_state.get("odds_api_key_input", "")
    v = str(v).strip()
    return v if v else None

def odds_api_enabled() -> bool:
    return get_odds_api_key() is not None

def _session() -> requests.Session:
    # sessão reaproveita conexões (melhor performance no Cloud)
    s = requests.Session()
    s.headers.update({"User-Agent": "futebol-predictor/1.0"})
    return s

@st.cache_data(show_spinner=False, ttl=300)  # 5 min
def get_quota_headers(api_key: str) -> Dict[str, Optional[int]]:
    """
    Usa GET /v4/sports (não custa quota) e lê headers:
      x-requests-remaining, x-requests-used, x-requests-last
    Docs: The Odds API v4.
    """
    url = f"{ODDS_API_HOST}/v4/sports/"
    r = _session().get(url, params={"apiKey": api_key}, timeout=12)
    # Mesmo se der erro, tentamos ler headers
    def _to_int(x):
        try:
            return int(x)
        except Exception:
            return None

    return {
        "remaining": _to_int(r.headers.get("x-requests-remaining")),
        "used": _to_int(r.headers.get("x-requests-used")),
        "last": _to_int(r.headers.get("x-requests-last")),
        "status_code": r.status_code,
    }

def _norm_team(s: str) -> str:
    return str(s).strip().lower()

def _pick_best_h2h(outcomes: list) -> Dict[str, float]:
    """
    outcomes: lista com {name, price} em odds decimais.
    Retorna melhor odd por nome.
    """
    best = {}
    for o in outcomes:
        name = str(o.get("name", "")).strip()
        price = o.get("price", None)
        if not name or price is None:
            continue
        try:
            price = float(price)
        except Exception:
            continue
        if price <= 1.0:
            continue
        if name not in best or price > best[name]:
            best[name] = price
    return best

@st.cache_data(show_spinner=False, ttl=600)  # 10 min
def fetch_best_h2h_odds(
    api_key: str,
    sport_key: str,
    region: str = "eu",
    odds_format: str = "decimal",
) -> Tuple[Optional[list], Dict[str, Optional[int]]]:
    """
    Busca odds pré-jogo (endpoint /odds) para 1X2 (h2h).
    Retorna:
      - lista de eventos (JSON) ou None
      - quota headers (remaining/used/last)
    """
    url = f"{ODDS_API_HOST}/v4/sports/{sport_key}/odds/"
    params = {
        "apiKey": api_key,
        "regions": region,
        "markets": "h2h",
        "oddsFormat": odds_format,
    }

    r = _session().get(url, params=params, timeout=18)
    q = {
        "remaining": _try_int(r.headers.get("x-requests-remaining")),
        "used": _try_int(r.headers.get("x-requests-used")),
        "last": _try_int(r.headers.get("x-requests-last")),
        "status_code": r.status_code,
    }

    if r.status_code != 200:
        return None, q

    try:
        data = r.json()
    except Exception:
        return None, q

    return data, q

def _try_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None

def match_event_best_prices(
    events: list,
    league_home_team: str,
    league_away_team: str,
) -> Optional[Dict[str, float]]:
    """
    Procura um evento na lista retornada e extrai as melhores odds decimais para:
      Mandante, Visitante e Empate (Draw) quando existir.
    """
    ht = _norm_team(league_home_team)
    at = _norm_team(league_away_team)

    # Tentativa por match exato
    for ev in events:
        home = _norm_team(ev.get("home_team", ""))
        away = _norm_team(ev.get("away_team", ""))
        if home == ht and away == at:
            return _extract_best_from_event(ev)

    # Fallback: tenta match invertido (às vezes fonte troca)
    for ev in events:
        home = _norm_team(ev.get("home_team", ""))
        away = _norm_team(ev.get("away_team", ""))
        if home == at and away == ht:
            return _extract_best_from_event(ev)

    return None

def _extract_best_from_event(ev: Dict[str, Any]) -> Optional[Dict[str, float]]:
    bookmakers = ev.get("bookmakers", [])
    if not bookmakers:
        return None

    best_all = {}
    for bm in bookmakers:
        markets = bm.get("markets", [])
        for m in markets:
            if m.get("key") != "h2h":
                continue
            outcomes = m.get("outcomes", [])
            best = _pick_best_h2h(outcomes)
            for k, v in best.items():
                if k not in best_all or v > best_all[k]:
                    best_all[k] = v

    return best_all if best_all else None

def implied_probs_from_decimal_odds(odds: Dict[str, float]) -> Dict[str, float]:
    """
    Prob implícita simples: 1/odd (sem remover vigorish).
    """
    out = {}
    for k, o in odds.items():
        try:
            out[k] = 1.0 / float(o)
        except Exception:
            pass
    return out
