from __future__ import annotations

import sys
from pathlib import Path

import folium
import h3
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

st.set_page_config(page_title="VTB ATM Game", page_icon="🎯", layout="wide")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
      .stApp {
        background:
          radial-gradient(circle at top left, rgba(255,140,66,0.20), transparent 30%),
          radial-gradient(circle at bottom right, rgba(17,48,92,0.18), transparent 24%),
          linear-gradient(180deg, #fcfaf7 0%, #eee6db 100%);
      }
      html, body, [class*="css"]  { font-family: 'Space Grotesk', sans-serif; }
      .hero { padding: 1rem 1.2rem; border-radius: 18px; background: #0f2747; color: #fff6eb; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_game_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cells = pd.read_parquet(DATA_DIR / "game_cells.parquet")
    cells_by_scenario = pd.read_parquet(DATA_DIR / "game_cells_by_scenario.parquet")
    recommendations = pd.read_parquet(DATA_DIR / "game_recommendations.parquet")
    best = pd.read_parquet(DATA_DIR / "game_best_by_scenario.parquet")
    return cells, cells_by_scenario, recommendations, best


def nearest_cell(lat: float, lon: float, cells: pd.DataFrame) -> pd.Series:
    query_cell = h3.latlng_to_cell(lat, lon, 9)
    match = cells[cells["h3_index"] == query_cell]
    if not match.empty:
        return match.iloc[0]
    idx = ((cells["lat"] - lat) ** 2 + (cells["lon"] - lon) ** 2).idxmin()
    return cells.loc[idx]


def objective_for_row(row: pd.Series, scenario: str) -> float:
    mapping = {
        "profit": row["profit_core_score"],
        "coverage": row["coverage_core_score"],
        "social": row["social_core_score"],
        "competitor": row["competitor_core_score"],
        "business": row["business_core_score"],
        "balanced": (
            0.35 * row["profit_core_score"]
            + 0.25 * row["coverage_core_score"]
            + 0.15 * row["social_core_score"]
            + 0.15 * row["competitor_core_score"]
            + 0.10 * row["business_core_score"]
        ),
    }
    return float(mapping[scenario])


cells, cells_by_scenario, recommendations, best_by_scenario = load_game_data()
scenarios = best_by_scenario["scenario"].tolist()
scenario = st.sidebar.selectbox("Scenario", scenarios)
best_row = best_by_scenario[best_by_scenario["scenario"] == scenario].iloc[0]

st.markdown(
    """
    <div class="hero">
      <h1>VTB ATM Placement Game</h1>
      <p>Поставьте банкомат кликом по карте. Модель сравнит ваш выбор с лучшей точкой в выбранном сценарии.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

m = folium.Map(location=[55.75, 37.62], zoom_start=9.5, tiles="CartoDB positron")
folium.CircleMarker(
    [best_row["lat"], best_row["lon"]],
    radius=8,
    color="#0f2747",
    fill=True,
    fill_opacity=0.9,
    tooltip="Model best",
).add_to(m)

result = st_folium(m, width=None, height=560)

left, right = st.columns([1.0, 1.0])
with left:
    st.subheader("Scenario Best")
    st.write(
        {
            "scenario": scenario,
            "best_cell": best_row["h3_index"],
            "best_rank": int(best_row["selected_rank"]),
            "recommended_atm_type": best_row["recommended_atm_type"],
        }
    )

with right:
    st.subheader("Your Guess")
    if result.get("last_clicked"):
        lat = result["last_clicked"]["lat"]
        lon = result["last_clicked"]["lng"]
        guessed = nearest_cell(lat, lon, cells_by_scenario[cells_by_scenario["scenario"] == scenario])
        user_score = objective_for_row(guessed, scenario)
        best_score = objective_for_row(best_row, scenario)
        pct = 100 * user_score / best_score if best_score > 0 else 0.0
        st.metric("Player Score", f"{pct:.1f}/100")
        st.write(
            {
                "your_cell": guessed["h3_index"],
                "model_best_cell": best_row["h3_index"],
                "your_atm_type": guessed["recommended_atm_type"],
                "model_atm_type": best_row["recommended_atm_type"],
                "delta_objective": round(best_score - user_score, 4),
            }
        )
        st.caption("Чем ближе Player Score к 100, тем ближе ваша догадка к оптимальному placement.")
    else:
        st.info("Кликните по карте, чтобы сделать ставку на новую точку ATM.")
