#!/usr/bin/env python3
"""
app.py - Análise preditiva de partidas de futebol

OBJETIVO
- Calcular matriz de probabilidades de placares (0x0 até 5x5) via Poisson
- Listar TOP 5 placares mais prováveis e menos prováveis
- Calcular 1X2 (mandante/empate/visitante), Over/Under 2.5 e BTTS
- Modelo alternativo com Machine Learning (RandomForest) para prever gols

DADOS
- Este app suporta:
  (a) CSV próprio (histórico de jogos)
  (b) Dataset fictício gerado via --make-sample (sem dados pagos)

NOTAS IMPORTANTES
- O "xG" aqui é uma *expectativa de gols estimada* (não é xG tracking real).
- O modelo Poisson usa médias ofensivas/defensivas, casa/fora e Elo (opcional).
- O modelo ML usa features simples derivadas do histórico.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# =========================
# Utilidades / Tipos
# =========================

@dataclass
class TeamForm:
    """Resumo do time nos últimos N jogos."""
    team: str
    n_games: int
    gf_avg: float          # média gols marcados
    ga_avg: float          # média gols sofridos
    gf_home_avg: float     # média gols marcados em casa
    ga_home_avg: float     # média gols sofridos em casa
    gf_away_avg: float     # média gols marcados fora
    ga_away_avg: float     # média gols sofridos fora
    points_per_game: float # forma recente em pontos/jogo (3-1-0)
    elo: float             # força relativa (opcional)


# =========================
# Dados fictícios (para teste)
# =========================

def make_sample_dataset_csv(path: str = "sample_matches.csv", seed: int = 42) -> str:
    """
    Gera um CSV fictício de histórico de partidas.
    Colunas:
      date, league, home_team, away_team, home_goals, away_goals, home_elo, away_elo
    """
    rng = np.random.default_rng(seed)

    leagues = ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1", "Brasileirão", "Champions League"]
    teams_by_league = {
        "Premier League": ["Arsenal", "Liverpool", "Man City", "Chelsea", "Tottenham", "Newcastle"],
        "La Liga": ["Real Madrid", "Barcelona", "Atletico", "Sevilla", "Villarreal", "Sociedad"],
        "Serie A": ["Inter", "Milan", "Juventus", "Napoli", "Roma", "Lazio"],
        "Bundesliga": ["Bayern", "Dortmund", "Leipzig", "Leverkusen", "Frankfurt", "Stuttgart"],
        "Ligue 1": ["PSG", "Marseille", "Lyon", "Monaco", "Lille", "Rennes"],
        "Brasileirão": ["Palmeiras", "Flamengo", "Corinthians", "São Paulo", "Grêmio", "Atlético-MG"],
        "Champions League": ["Real Madrid", "Man City", "Bayern", "PSG", "Inter", "Barcelona"],
    }

    rows = []
    # Cria ~420 partidas fictícias (60 por liga, aproximado)
    for league in leagues:
        teams = teams_by_league[league]
        # Elo base por time (fictício)
        base_elos = {t: float(rng.integers(1450, 1850)) for t in teams}

        for i in range(60):
            home, away = rng.choice(teams, size=2, replace=False)
            home_elo = base_elos[home] + float(rng.normal(0, 20))
            away_elo = base_elos[away] + float(rng.normal(0, 20))

            # Intensidades de gols fictícias com leve influência de Elo e fator casa
            elo_diff = (home_elo - away_elo) / 400.0
            lam_home = max(0.2, 1.35 + 0.25 * elo_diff + 0.20)  # +0.20 casa
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
            })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


# =========================
# Engenharia de features / métricas de forma
# =========================

def _points_from_score(gf: int, ga: int) -> int:
    if gf > ga:
        return 3
    if gf == ga:
        return 1
    return 0


def compute_team_form(
    matches: pd.DataFrame,
    team: str,
    n_last: int = 8,
    default_elo: float = 1600.0,
) -> TeamForm:
    """
    Calcula métricas dos últimos N jogos do time (casa+fora) e separa médias casa/fora.
    Espera colunas:
      home_team, away_team, home_goals, away_goals, home_elo, away_elo
    """
    # Jogos do time
    mask = (matches["home_team"] == team) | (matches["away_team"] == team)
    tm = matches.loc[mask].copy()

    if tm.empty:
        # Time não existe no dataset -> retorna defaults
        return TeamForm(
            team=team, n_games=0,
            gf_avg=1.0, ga_avg=1.0,
            gf_home_avg=1.1, ga_home_avg=1.0,
            gf_away_avg=0.9, ga_away_avg=1.1,
            points_per_game=1.0,
            elo=default_elo
        )

    # Ordena por data (string YYYY-MM-DD funciona bem para ordenar)
    tm = tm.sort_values("date").tail(n_last)

    # GF/GA por jogo, e pontos
    gf_list, ga_list, pts_list = [], [], []
    gf_home, ga_home = [], []
    gf_away, ga_away = [], []
    elo_values = []

    for _, r in tm.iterrows():
        if r["home_team"] == team:
            gf, ga = int(r["home_goals"]), int(r["away_goals"])
            gf_home.append(gf)
            ga_home.append(ga)
            elo_values.append(float(r.get("home_elo", default_elo)))
        else:
            gf, ga = int(r["away_goals"]), int(r["home_goals"])
            gf_away.append(gf)
            ga_away.append(ga)
            elo_values.append(float(r.get("away_elo", default_elo)))

        gf_list.append(gf)
        ga_list.append(ga)
        pts_list.append(_points_from_score(gf, ga))

    def _avg(xs: List[int], fallback: float) -> float:
        return float(np.mean(xs)) if len(xs) > 0 else fallback

    return TeamForm(
        team=team,
        n_games=len(tm),
        gf_avg=_avg(gf_list, 1.0),
        ga_avg=_avg(ga_list, 1.0),
        gf_home_avg=_avg(gf_home, 1.1),
        ga_home_avg=_avg(ga_home, 1.0),
        gf_away_avg=_avg(gf_away, 0.9),
        ga_away_avg=_avg(ga_away, 1.1),
        points_per_game=float(np.mean(pts_list)) if pts_list else 1.0,
        elo=float(np.mean(elo_values)) if elo_values else default_elo
    )


def league_goal_averages(matches: pd.DataFrame, league: str) -> Dict[str, float]:
    """
    Calcula médias de gols no campeonato (para normalizar intensidades).
    Retorna:
      avg_home_goals, avg_away_goals, avg_total_goals
    """
    df = matches[matches["league"] == league].copy()
    if df.empty:
        # defaults genéricos
        return {"avg_home_goals": 1.35, "avg_away_goals": 1.15, "avg_total_goals": 2.50}

    avg_h = float(df["home_goals"].mean())
    avg_a = float(df["away_goals"].mean())
    return {"avg_home_goals": avg_h, "avg_away_goals": avg_a, "avg_total_goals": avg_h + avg_a}


# =========================
# Modelo Poisson
# =========================

def estimate_expected_goals_poisson(
    matches: pd.DataFrame,
    league: str,
    home_team: str,
    away_team: str,
    n_last: int = 8,
    home_advantage: float = 0.12,
    elo_k: float = 0.10,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Estima expectativa de gols (lambda) do mandante e visitante.

    Ideia:
    - Força ofensiva ~ (GF casa do mandante / média gols casa da liga)
    - Fraqueza defensiva adversária ~ (GA fora do visitante / média gols casa da liga)
    - Lambda_home = média_gols_casa_liga * ataque_home * defesa_away * ajustes
    - Lambda_away = média_gols_fora_liga * ataque_away * defesa_home * ajustes

    Ajustes:
    - fator casa (home_advantage): +12% por padrão
    - Elo: pequeno ajuste com diferença de Elo
    - forma recente (points_per_game): ajuste leve e estável
    """
    lg = league_goal_averages(matches, league)
    avg_home = lg["avg_home_goals"]
    avg_away = lg["avg_away_goals"]

    home_form = compute_team_form(matches, home_team, n_last=n_last)
    away_form = compute_team_form(matches, away_team, n_last=n_last)

    # Normalizações (evita divisão por zero)
    eps = 1e-6

    attack_home = (home_form.gf_home_avg + eps) / (avg_home + eps)
    defense_away = (away_form.ga_away_avg + eps) / (avg_home + eps)

    attack_away = (away_form.gf_away_avg + eps) / (avg_away + eps)
    defense_home = (home_form.ga_home_avg + eps) / (avg_away + eps)

    # Ajuste de forma recente (estável: centra em 1.0)
    # 1 ponto/jogo ~ "neutro". >1 melhora ataque um pouco e/ou reduz gols sofridos.
    form_home = 1.0 + 0.05 * (home_form.points_per_game - 1.0)
    form_away = 1.0 + 0.05 * (away_form.points_per_game - 1.0)

    # Ajuste Elo: diferença positiva favorece mandante
    elo_diff = (home_form.elo - away_form.elo) / 400.0
    elo_adj_home = 1.0 + elo_k * elo_diff
    elo_adj_away = 1.0 - elo_k * elo_diff

    lam_home = avg_home * attack_home * defense_away * (1.0 + home_advantage) * form_home * elo_adj_home
    lam_away = avg_away * attack_away * defense_home * (1.0 - home_advantage/2.0) * form_away * elo_adj_away

    # Limites básicos para estabilidade
    lam_home = float(np.clip(lam_home, 0.05, 4.50))
    lam_away = float(np.clip(lam_away, 0.05, 4.50))

    debug = {
        "avg_home_goals_league": avg_home,
        "avg_away_goals_league": avg_away,
        "attack_home": float(attack_home),
        "defense_away": float(defense_away),
        "attack_away": float(attack_away),
        "defense_home": float(defense_home),
        "form_home": float(form_home),
        "form_away": float(form_away),
        "elo_home": float(home_form.elo),
        "elo_away": float(away_form.elo),
        "elo_diff_scaled": float(elo_diff),
        "lambda_home": lam_home,
        "lambda_away": lam_away,
    }
    return lam_home, lam_away, debug


def score_matrix_poisson(lambda_home: float, lambda_away: float, max_goals: int = 5) -> pd.DataFrame:
    """
    Gera matriz P(HomeGoals=i, AwayGoals=j) para 0..max_goals.
    Assumindo independência entre gols mandante e visitante (Poisson independente).
    """
    hs = np.arange(0, max_goals + 1)
    as_ = np.arange(0, max_goals + 1)

    p_home = poisson.pmf(hs, mu=lambda_home)
    p_away = poisson.pmf(as_, mu=lambda_away)

    mat = np.outer(p_home, p_away)  # (max+1, max+1)
    df = pd.DataFrame(mat, index=[f"H{i}" for i in hs], columns=[f"A{j}" for j in as_])
    return df


def list_top_and_bottom_scores(mat: pd.DataFrame, top_k: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retorna dois dataframes:
    - top_k placares mais prováveis
    - top_k placares menos prováveis (entre os 36 placares 0..5; inclui valores quase 0)
    """
    flat = []
    for i, rname in enumerate(mat.index):
        for j, cname in enumerate(mat.columns):
            p = float(mat.iloc[i, j])
            flat.append((i, j, p))

    flat_sorted = sorted(flat, key=lambda x: x[2], reverse=True)
    top = flat_sorted[:top_k]
    bottom = sorted(flat, key=lambda x: x[2])[:top_k]

    def _to_df(items):
        rows = []
        for hg, ag, p in items:
            rows.append({"placar": f"{hg}x{ag}", "prob": p, "prob_%": 100.0 * p})
        return pd.DataFrame(rows)

    return _to_df(top), _to_df(bottom)


def probs_1x2_over_btts(mat: pd.DataFrame) -> Dict[str, float]:
    """
    Calcula:
    - 1X2 (home/draw/away)
    - Over/Under 2.5 gols
    - BTTS (ambas marcam)
    """
    probs = mat.to_numpy()
    max_goals = probs.shape[0] - 1

    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    p_over25 = 0.0
    p_btts = 0.0

    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            p = float(probs[hg, ag])
            if hg > ag:
                p_home += p
            elif hg == ag:
                p_draw += p
            else:
                p_away += p

            if (hg + ag) >= 3:
                p_over25 += p

            if hg >= 1 and ag >= 1:
                p_btts += p

    p_under25 = 1.0 - p_over25

    return {
        "home_win": p_home,
        "draw": p_draw,
        "away_win": p_away,
        "over_2_5": p_over25,
        "under_2_5": p_under25,
        "btts_yes": p_btts,
        "btts_no": 1.0 - p_btts,
    }


# =========================
# Modelo ML (RandomForest) - alternativo
# =========================

def build_ml_dataset(matches: pd.DataFrame, n_last: int = 8) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Cria dataset supervisionado para prever gols do mandante e visitante.

    Features (simples e interpretáveis):
    - médias GF/GA do mandante (geral/casa)
    - médias GF/GA do visitante (geral/fora)
    - pontos/jogo recentes de ambos
    - elo home/away e diferença
    - médias de gols da liga

    Targets:
    - y_home_goals, y_away_goals
    """
    rows = []

    # Para evitar vazamento, criamos features usando partidas ANTERIORES à partida atual.
    matches_sorted = matches.sort_values("date").reset_index(drop=True)

    for idx, r in matches_sorted.iterrows():
        league = r["league"]
        home = r["home_team"]
        away = r["away_team"]

        history = matches_sorted.iloc[:idx]  # somente jogos anteriores

        lg = league_goal_averages(history, league) if len(history) > 0 else {"avg_home_goals": 1.35, "avg_away_goals": 1.15}
        hf = compute_team_form(history, home, n_last=n_last)
        af = compute_team_form(history, away, n_last=n_last)

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
        }

        rows.append({
            **feat,
            "y_home_goals": int(r["home_goals"]),
            "y_away_goals": int(r["away_goals"]),
        })

    df = pd.DataFrame(rows).dropna()
    X = df.drop(columns=["y_home_goals", "y_away_goals"])
    y_home = df["y_home_goals"]
    y_away = df["y_away_goals"]
    return X, y_home, y_away


def train_ml_models(matches: pd.DataFrame, n_last: int = 8, random_state: int = 42) -> Dict[str, object]:
    """
    Treina dois RandomForestRegressor:
    - um para gols do mandante
    - outro para gols do visitante

    Retorna modelos e métricas simples (MAE em validação).
    """
    X, y_h, y_a = build_ml_dataset(matches, n_last=n_last)

    if len(X) < 50:
        raise ValueError("Dataset muito pequeno para treinar ML. Gere mais dados ou use um histórico real maior.")

    X_train, X_val, yh_train, yh_val = train_test_split(X, y_h, test_size=0.2, random_state=random_state)
    _, _, ya_train, ya_val = train_test_split(X, y_a, test_size=0.2, random_state=random_state)

    model_h = RandomForestRegressor(
        n_estimators=400,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1,
    )
    model_a = RandomForestRegressor(
        n_estimators=400,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1,
    )

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
    }


def predict_expected_goals_ml(
    matches: pd.DataFrame,
    league: str,
    home_team: str,
    away_team: str,
    trained: Dict[str, object],
    n_last: int = 8,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Usa modelos treinados para prever gols esperados do mandante/visitante.
    Clipa para faixa plausível e retorna "lambdas" que podem ser plugados na Poisson
    para gerar matriz de placares (comparação justa e simples).

    Observação: RandomForest não gera distribuição diretamente; aqui usamos:
    - previsão contínua como "lambda"
    - matriz gerada por Poisson com lambda do ML
    """
    cols = trained["feature_columns"]
    lg = league_goal_averages(matches, league)

    hf = compute_team_form(matches, home_team, n_last=n_last)
    af = compute_team_form(matches, away_team, n_last=n_last)

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
    }

    X = pd.DataFrame([row])[cols]

    model_h = trained["model_home"]
    model_a = trained["model_away"]

    lam_home = float(model_h.predict(X)[0])
    lam_away = float(model_a.predict(X)[0])

    lam_home = float(np.clip(lam_home, 0.05, 4.50))
    lam_away = float(np.clip(lam_away, 0.05, 4.50))

    debug = {
        "lambda_home_ml": lam_home,
        "lambda_away_ml": lam_away,
        "mae_home_val": float(trained["mae_home"]),
        "mae_away_val": float(trained["mae_away"]),
    }
    return lam_home, lam_away, debug


# =========================
# Impressão / UX CLI
# =========================

def _fmt_pct(p: float) -> str:
    return f"{100.0*p:6.2f}%"


def print_score_table(mat: pd.DataFrame, top: pd.DataFrame, bottom: pd.DataFrame, title: str) -> None:
    """
    Mostra:
    - matriz de placares em %
    - TOP 5 e BOTTOM 5
    """
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

    # Matriz em percentual (arredondada)
    mat_pct = (mat * 100.0).round(3)
    # Renomeia índices/colunas para 0..5
    mat_pct.index = [i.replace("H", "") for i in mat_pct.index]
    mat_pct.columns = [j.replace("A", "") for j in mat_pct.columns]
    print("\nMatriz de Probabilidades de Placares (%): (Home x Away)")
    print(mat_pct.to_string())

    print("\nTOP 5 placares MAIS prováveis:")
    t = top.copy()
    t["prob_%"] = t["prob_%"].map(lambda x: round(x, 3))
    print(t[["placar", "prob_%"]].to_string(index=False))

    print("\nTOP 5 placares MENOS prováveis:")
    b = bottom.copy()
    b["prob_%"] = b["prob_%"].map(lambda x: round(x, 6))
    print(b[["placar", "prob_%"]].to_string(index=False))


def print_market_probs(probs: Dict[str, float]) -> None:
    print("\nProbabilidades de Mercado:")
    print(f"  Mandante vence (1): {_fmt_pct(probs['home_win'])}")
    print(f"  Empate (X):        {_fmt_pct(probs['draw'])}")
    print(f"  Visitante vence(2):{_fmt_pct(probs['away_win'])}")
    print(f"  Over 2.5 gols:     {_fmt_pct(probs['over_2_5'])}")
    print(f"  Under 2.5 gols:    {_fmt_pct(probs['under_2_5'])}")
    print(f"  BTTS (Sim):        {_fmt_pct(probs['btts_yes'])}")
    print(f"  BTTS (Não):        {_fmt_pct(probs['btts_no'])}")


# =========================
# Pipeline principal
# =========================

def run_poisson_analysis(
    matches: pd.DataFrame,
    league: str,
    home_team: str,
    away_team: str,
    n_last: int,
    max_goals: int,
) -> Dict[str, object]:
    lam_h, lam_a, dbg = estimate_expected_goals_poisson(
        matches=matches,
        league=league,
        home_team=home_team,
        away_team=away_team,
        n_last=n_last,
    )
    mat = score_matrix_poisson(lam_h, lam_a, max_goals=max_goals)
    top, bottom = list_top_and_bottom_scores(mat, top_k=5)
    probs = probs_1x2_over_btts(mat)
    return {"lambda_home": lam_h, "lambda_away": lam_a, "debug": dbg, "mat": mat, "top": top, "bottom": bottom, "probs": probs}


def run_ml_analysis(
    matches: pd.DataFrame,
    league: str,
    home_team: str,
    away_team: str,
    n_last: int,
    max_goals: int,
) -> Dict[str, object]:
    trained = train_ml_models(matches, n_last=n_last)
    lam_h, lam_a, dbg = predict_expected_goals_ml(
        matches=matches,
        league=league,
        home_team=home_team,
        away_team=away_team,
        trained=trained,
        n_last=n_last,
    )
    mat = score_matrix_poisson(lam_h, lam_a, max_goals=max_goals)
    top, bottom = list_top_and_bottom_scores(mat, top_k=5)
    probs = probs_1x2_over_btts(mat)
    return {"lambda_home": lam_h, "lambda_away": lam_a, "debug": dbg, "mat": mat, "top": top, "bottom": bottom, "probs": probs}


def compare_models(poisson_out: Dict[str, object], ml_out: Dict[str, object]) -> None:
    """
    Comparação simples e direta:
    - lambdas
    - 1X2 / Over / BTTS
    - top placares
    """
    print("\n" + "#" * 78)
    print("COMPARAÇÃO: Poisson vs ML (RandomForest -> lambdas -> Poisson)")
    print("#" * 78)

    print(f"\nLambdas (gols esperados):")
    print(f"  Poisson -> Home: {poisson_out['lambda_home']:.3f} | Away: {poisson_out['lambda_away']:.3f}")
    print(f"  ML      -> Home: {ml_out['lambda_home']:.3f} | Away: {ml_out['lambda_away']:.3f}")

    pP = poisson_out["probs"]
    pM = ml_out["probs"]

    print("\n1X2 (Poisson vs ML):")
    print(f"  1: {_fmt_pct(pP['home_win'])}  |  {_fmt_pct(pM['home_win'])}")
    print(f"  X: {_fmt_pct(pP['draw'])}  |  {_fmt_pct(pM['draw'])}")
    print(f"  2: {_fmt_pct(pP['away_win'])}  |  {_fmt_pct(pM['away_win'])}")

    print("\nOver/Under 2.5 (Poisson vs ML):")
    print(f"  Over 2.5:  {_fmt_pct(pP['over_2_5'])}  |  {_fmt_pct(pM['over_2_5'])}")
    print(f"  Under 2.5: {_fmt_pct(pP['under_2_5'])} |  {_fmt_pct(pM['under_2_5'])}")

    print("\nBTTS (Poisson vs ML):")
    print(f"  Sim: {_fmt_pct(pP['btts_yes'])}  |  {_fmt_pct(pM['btts_yes'])}")
    print(f"  Não: {_fmt_pct(pP['btts_no'])}   |  {_fmt_pct(pM['btts_no'])}")

    print("\nTop 3 placares (Poisson):")
    print(poisson_out["top"][["placar", "prob_%"]].head(3).to_string(index=False))
    print("\nTop 3 placares (ML):")
    print(ml_out["top"][["placar", "prob_%"]].head(3).to_string(index=False))

    print("\nNota:")
    print("- O ML aqui prevê gols esperados com RandomForest e depois convertemos em distribuição via Poisson.")
    print("- Para distribuição diretamente via ML, você precisaria de modelos probabilísticos (ex: PoissonRegressor, NB, etc.).")


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Análise preditiva de partidas de futebol (Poisson + ML).")

    p.add_argument("--data", type=str, default="sample_matches.csv", help="Caminho do CSV de histórico.")
    p.add_argument("--make-sample", action="store_true", help="Gera CSV fictício para testes em --data (padrão sample_matches.csv).")

    p.add_argument("--league", type=str, required=False, default="Premier League", help="Nome da liga/campeonato.")
    p.add_argument("--home", type=str, required=False, default="Arsenal", help="Time mandante.")
    p.add_argument("--away", type=str, required=False, default="Liverpool", help="Time visitante.")

    p.add_argument("--n-last", type=int, default=8, help="Quantidade de jogos recentes (5 a 10 recomendado).")
    p.add_argument("--max-goals", type=int, default=5, help="Máximo de gols na matriz (0..max).")

    p.add_argument("--use-ml", action="store_true", help="Executa também modelo ML e compara com Poisson.")
    return p.parse_args()


def load_matches(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"date", "league", "home_team", "away_team", "home_goals", "away_goals"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV inválido. Faltam colunas: {sorted(missing)}")

    # Campos opcionais (elo)
    if "home_elo" not in df.columns:
        df["home_elo"] = 1600.0
    if "away_elo" not in df.columns:
        df["away_elo"] = 1600.0

    # Normalização de tipos
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    return df


def main() -> None:
    args = parse_args()

    if args.make_sample:
        out = make_sample_dataset_csv(args.data)
        print(f"[OK] CSV fictício criado em: {out}")

    matches = load_matches(args.data)

    # Filtra por liga (mas mantém dataset inteiro para formas se quiser - aqui usamos a liga para médias)
    # Mantemos tudo para computar forma do time inclusive em outras competições, se existirem.
    # Se quiser restringir forma por liga, basta filtrar antes de compute_team_form.
    league = args.league
    home = args.home
    away = args.away

    print("\n" + "=" * 78)
    print(f"ANÁLISE DA PARTIDA: {home} (casa) vs {away} (fora) | Liga: {league}")
    print("=" * 78)

    # 1) Poisson
    outP = run_poisson_analysis(
        matches=matches,
        league=league,
        home_team=home,
        away_team=away,
        n_last=args.n_last,
        max_goals=args.max_goals
    )

    print(f"\nxG estimado (Poisson / lambdas): Home={outP['lambda_home']:.3f} | Away={outP['lambda_away']:.3f}")
    # Debug opcional: descomente se quiser ver fatores
    # print(pd.Series(outP["debug"]).to_string())

    print_score_table(outP["mat"], outP["top"], outP["bottom"], title="MODELO POISSON")
    print_market_probs(outP["probs"])

    # 2) ML opcional
    if args.use_ml:
        outM = run_ml_analysis(
            matches=matches,
            league=league,
            home_team=home,
            away_team=away,
            n_last=args.n_last,
            max_goals=args.max_goals
        )

        print(f"\nxG estimado (ML->lambdas): Home={outM['lambda_home']:.3f} | Away={outM['lambda_away']:.3f}")
        print(f"Validação do ML (MAE): home={outM['debug']['mae_home_val']:.3f} | away={outM['debug']['mae_away_val']:.3f}")

        print_score_table(outM["mat"], outM["top"], outM["bottom"], title="MODELO ML (RandomForest) -> lambdas -> Poisson")
        print_market_probs(outM["probs"])

        compare_models(outP, outM)

    print("\n[FIM] Dica: troque --home/--away/--league e aponte --data para um CSV real.")


if __name__ == "__main__":
    main()
