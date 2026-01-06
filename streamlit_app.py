# =========================
# TRECHO ADICIONAL – ALERTAS + RECOMENDAÇÃO + TEXTO
# (cole ESTE BLOCO NO FINAL do streamlit_app.py atual)
# =========================

def market_alert(diff_pp: float) -> tuple[str, str]:
    if diff_pp <= 8:
        return "🟢 Baixo risco", "green"
    if diff_pp <= 15:
        return "🟡 Risco médio", "orange"
    return "🔴 Alto risco", "red"


def recommended_market(diffs_pp: dict, probsP: dict) -> tuple[str, str]:
    priority = ["over_2_5", "under_2_5", "btts_yes", "home_win", "away_win", "draw"]
    labels = {
        "over_2_5": "Over 2.5 gols",
        "under_2_5": "Under 2.5 gols",
        "btts_yes": "Ambas marcam (SIM)",
        "home_win": "Vitória do mandante",
        "away_win": "Vitória do visitante",
        "draw": "Empate"
    }

    safe = {k: v for k, v in diffs_pp.items() if v <= 8}
    if not safe:
        return "Nenhum mercado confiável", "Alta divergência entre os modelos."

    best = min(safe, key=safe.get)
    prob = probsP.get(best, 0) * 100

    return labels[best], f"Baixa divergência ({safe[best]:.1f} p.p.) e probabilidade estimada de {prob:.1f}%."


def analysis_text(conf_score: int, level: str, diffs_pp: dict) -> str:
    text = []

    text.append(f"🔎 **Nível geral de confiança:** {level} ({conf_score}/100).")

    if conf_score >= 80:
        text.append("Os modelos estatístico (Poisson) e de Machine Learning estão fortemente alinhados.")
    elif conf_score >= 60:
        text.append("Os modelos concordam em pontos-chave, mas há divergências relevantes.")
    else:
        text.append("Os modelos divergem bastante. Jogo considerado instável.")

    risky = [k for k, v in diffs_pp.items() if v > 15]
    if risky:
        text.append("⚠️ Mercados com alta incerteza: " + ", ".join(risky))

    safe = [k for k, v in diffs_pp.items() if v <= 8]
    if safe:
        text.append("✅ Mercados mais confiáveis: " + ", ".join(safe))

    return "\n\n".join(text)


# ======== INTEGRAÇÃO VISUAL (chamar APÓS calcular diffs_pp, probsP e score) ========

st.subheader("📌 Mercado recomendado automaticamente")

market, reason = recommended_market(diffs_pp, probsP)
st.success(f"**{market}**\n\n{reason}")

st.subheader("🚦 Alertas por mercado")

alert_rows = []
labels_map = {
    "home_win": "Vitória Mandante",
    "draw": "Empate",
    "away_win": "Vitória Visitante",
    "over_2_5": "Over 2.5",
    "btts_yes": "BTTS SIM"
}

for k, label in labels_map.items():
    diff = diffs_pp.get(k, None)
    if diff is None:
        continue
    status, _ = market_alert(diff)
    alert_rows.append({
        "Mercado": label,
        "Divergência (p.p.)": round(diff, 2),
        "Risco": status
    })

df_alerts = pd.DataFrame(alert_rows)
st.dataframe(df_alerts, use_container_width=True)

st.subheader("📝 Análise automática do jogo")
st.markdown(analysis_text(score, label, diffs_pp))
