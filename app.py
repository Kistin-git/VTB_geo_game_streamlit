from __future__ import annotations

import math
import random
from pathlib import Path

import folium
import h3
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

SCENARIO_DESCRIPTIONS = {
    "balanced": "Компромисс между прибылью, охватом и штрафом за избыточную плотность.",
    "profit": "Главная цель — выбрать максимально выгодную точку для банка.",
    "coverage": "Главная цель — закрыть зону, где присутствие ВТБ сейчас слабое.",
    "social": "Главная цель — выбрать точку с полезностью для социальных объектов.",
    "competitor": "Главная цель — поставить банкомат в зоне сильного конкурентного давления.",
    "business": "Главная цель — найти место под cash-in / малый бизнес и торговую активность.",
}


def apply_style() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
          .stApp {
            background:
              radial-gradient(circle at top left, rgba(255,140,66,0.20), transparent 30%),
              radial-gradient(circle at bottom right, rgba(17,48,92,0.14), transparent 25%),
              linear-gradient(180deg, #fcfaf7 0%, #efe7db 100%);
          }
          html, body, [class*="css"]  { font-family: 'Space Grotesk', sans-serif; }
          .hero {
            padding: 1.1rem 1.3rem;
            border-radius: 20px;
            background: linear-gradient(135deg, #0f2747 0%, #193b63 100%);
            color: #fff6eb;
            box-shadow: 0 14px 34px rgba(15,39,71,0.16);
          }
          .hero h1 { margin: 0; font-size: 2rem; }
          .hero p { margin: 0.4rem 0 0 0; color: #d7e2ef; }
          section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #fff9f0 0%, #efe5d6 100%);
          }
          .soft-card {
            padding: 0.95rem 1rem;
            border-radius: 18px;
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(15,39,71,0.08);
            box-shadow: 0 10px 24px rgba(15,39,71,0.05);
          }
          .status-card {
            padding: 1rem 1rem;
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(255,255,255,0.88), rgba(255,255,255,0.74));
            border: 1px solid rgba(15,39,71,0.08);
          }
          .mono { font-family: 'IBM Plex Mono', monospace; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_game_data() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "game_cells_by_scenario.parquet")


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


def focus_cells(center_row: pd.Series, cells: pd.DataFrame, radius_km: float) -> pd.DataFrame:
    def distance(row: pd.Series) -> float:
        lat1, lon1 = center_row["lat"], center_row["lon"]
        lat2, lon2 = row["lat"], row["lon"]
        radius = 6371.0088
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 2 * radius * math.asin(math.sqrt(a))

    local = cells.copy()
    local["focus_distance_km"] = local.apply(distance, axis=1)
    return local[local["focus_distance_km"] <= radius_km].copy()


def build_sector_bounds(center_row: pd.Series, radius_km: float) -> list[list[float]]:
    lat_pad = radius_km / 110.574
    lon_pad = radius_km / (111.320 * max(0.2, abs(math.cos(math.radians(center_row["lat"])))))
    return [
        [center_row["lat"] - lat_pad, center_row["lon"] - lon_pad],
        [center_row["lat"] + lat_pad, center_row["lon"] + lon_pad],
    ]


def choose_random_sector(scenario: str, cells: pd.DataFrame, radius_km: float) -> dict:
    frame = cells[cells["scenario"] == scenario].copy()
    frame["objective_score"] = frame.apply(lambda row: objective_for_row(row, scenario), axis=1)
    candidate_pool = frame.nlargest(220, "objective_score")
    sampled_indices = list(candidate_pool.sample(frac=1.0).index)

    for idx in sampled_indices:
        anchor = candidate_pool.loc[idx]
        sector = focus_cells(anchor, frame, radius_km)
        if len(sector) < 32:
            sector = focus_cells(anchor, frame, radius_km + 0.9)
        if len(sector) < 28:
            continue
        sector = sector.copy()
        sector["objective_score"] = sector.apply(lambda row: objective_for_row(row, scenario), axis=1)
        sector = sector.sort_values("objective_score", ascending=False).head(110).copy()
        local_best = sector.iloc[0]
        return {
            "anchor_h3": anchor["h3_index"],
            "sector_cells": sector,
            "local_best": local_best,
            "bounds": build_sector_bounds(local_best, radius_km),
        }

    fallback = frame.sort_values("objective_score", ascending=False).head(100).copy()
    local_best = fallback.iloc[0]
    return {
        "anchor_h3": local_best["h3_index"],
        "sector_cells": fallback,
        "local_best": local_best,
        "bounds": build_sector_bounds(local_best, radius_km),
    }


def initialize_or_refresh_sector(scenario: str, radius_km: float, force_new: bool = False) -> None:
    sector_key = f"{scenario}:{radius_km:.1f}"
    if force_new or st.session_state.get("game_sector_key") != sector_key or "game_sector_data" not in st.session_state:
        scenario_cells = GAME_CELLS[GAME_CELLS["scenario"] == scenario].copy()
        st.session_state["game_sector_key"] = sector_key
        st.session_state["game_sector_data"] = choose_random_sector(scenario, scenario_cells, radius_km)
        st.session_state["pending_guess"] = None
        st.session_state["locked_guess"] = None
        st.session_state["reveal_answer"] = False
        st.session_state["last_clicked_h3"] = None


def build_map(sector_cells: pd.DataFrame, local_best: pd.Series, bounds: list[list[float]]) -> folium.Map:
    m = folium.Map(
        location=[local_best["lat"], local_best["lon"]],
        zoom_start=13,
        tiles="CartoDB Voyager",
        control_scale=True,
        prefer_canvas=True,
        max_zoom=18,
        min_zoom=11,
        max_bounds=True,
    )
    m.fit_bounds(bounds)
    folium.Rectangle(
        bounds=bounds,
        color="#0f2747",
        weight=2,
        fill=True,
        fill_opacity=0.05,
        dash_array="6 6",
        tooltip="Игровой сектор",
    ).add_to(m)

    for row in sector_cells.itertuples():
        marker_color = "#90a7be"
        radius = 4.5
        if st.session_state.get("pending_guess") and row.h3_index == st.session_state["pending_guess"]["h3_index"]:
            marker_color = "#ff8c42"
            radius = 8.5
        if st.session_state.get("locked_guess") and row.h3_index == st.session_state["locked_guess"]["h3_index"]:
            marker_color = "#ff8c42"
            radius = 9.5

        popup = None
        if st.session_state.get("pending_guess") and row.h3_index == st.session_state["pending_guess"]["h3_index"]:
            popup = folium.Popup(
                """
                <div style="min-width:190px">
                  <div style="font-weight:700; color:#0f2747; margin-bottom:6px;">Точка выбрана</div>
                  <button style="background:#ff8c42;color:white;border:none;border-radius:10px;padding:8px 12px;font-weight:700;width:100%;">
                    Зафиксируйте точку справа
                  </button>
                </div>
                """,
                max_width=250,
                show=True,
            )

        folium.CircleMarker(
            [row.lat, row.lon],
            radius=radius,
            color=marker_color,
            weight=2,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.86 if marker_color == "#ff8c42" else 0.42,
            tooltip=(
                f"H3: {row.h3_index}<br>"
                f"Objective score: {row.objective_score:.3f}<br>"
                f"ATM type: {row.recommended_atm_type}"
            ),
            popup=popup,
        ).add_to(m)

    if st.session_state.get("reveal_answer"):
        folium.CircleMarker(
            [local_best["lat"], local_best["lon"]],
            radius=10,
            color="#0f2747",
            weight=3,
            fill=True,
            fill_color="#0f2747",
            fill_opacity=0.95,
            tooltip="Ответ модели",
        ).add_to(m)
        if st.session_state.get("locked_guess") is not None:
            folium.PolyLine(
                locations=[
                    [st.session_state["locked_guess"]["lat"], st.session_state["locked_guess"]["lon"]],
                    [local_best["lat"], local_best["lon"]],
                ],
                color="#0f2747",
                weight=2,
                dash_array="8 8",
            ).add_to(m)

    return m


def render_status_panel(scenario: str, local_best: pd.Series) -> None:
    with st.container(border=False):
        st.markdown("<div class='status-card'>", unsafe_allow_html=True)
        st.subheader("Панель игрока")
        st.caption("Вся обратная связь собрана здесь, без дополнительного скролла.")

        pending = st.session_state.get("pending_guess")
        locked = st.session_state.get("locked_guess")

        if locked is None and pending is None:
            st.info("Кликните по одной из точек на карте слева. После этого здесь появится кнопка фиксации.")
        elif locked is None and pending is not None:
            st.success(f"Вы выбрали H3 `{pending['h3_index']}`. Если точка подходит, зафиксируйте ее.")
            st.write(
                {
                    "candidate_cell": pending["h3_index"],
                    "atm_type": pending["recommended_atm_type"],
                    "profit_score": round(float(pending["profit_core_score"]), 3),
                    "coverage_score": round(float(pending["coverage_core_score"]), 3),
                }
            )
        else:
            guessed = pd.Series(locked)
            user_score = objective_for_row(guessed, scenario)
            best_score = objective_for_row(local_best, scenario)
            pct = 100.0 * user_score / best_score if best_score > 0 else 0.0

            col_a, col_b = st.columns(2)
            col_a.metric("Player Score", f"{pct:.1f}/100")
            col_b.metric("Ваш ATM тип", guessed["recommended_atm_type"])

            st.write(
                {
                    "your_cell": guessed["h3_index"],
                    "profit_score": round(float(guessed["profit_core_score"]), 3),
                    "coverage_score": round(float(guessed["coverage_core_score"]), 3),
                    "social_score": round(float(guessed["social_core_score"]), 3),
                }
            )

            if st.session_state.get("reveal_answer"):
                st.write(
                    {
                        "model_best_cell": local_best["h3_index"],
                        "model_atm_type": local_best["recommended_atm_type"],
                        "delta_objective": round(best_score - user_score, 4),
                    }
                )
                if pct >= 92:
                    st.success("Очень сильный выбор: вы почти попали в решение модели.")
                elif pct >= 75:
                    st.warning("Хорошая догадка, но модель нашла вариант лучше.")
                else:
                    st.error("Разрыв заметный: модель использует более сильную комбинацию факторов.")
            else:
                st.caption("Ваш score уже посчитан. Нажмите `Показать ответ модели`, чтобы открыть лучший вариант.")
        st.markdown("</div>", unsafe_allow_html=True)


st.set_page_config(page_title="VTB ATM Game", page_icon="🎯", layout="wide")
apply_style()

GAME_CELLS = load_game_data()
scenario = st.sidebar.selectbox("Игровой режим", sorted(GAME_CELLS["scenario"].unique().tolist()))
sector_radius_km = st.sidebar.slider("Размер игрового сектора, км", min_value=2.0, max_value=4.2, value=2.8, step=0.2)
initialize_or_refresh_sector(scenario, sector_radius_km)

st.sidebar.markdown("### Что это за режим")
st.sidebar.info(SCENARIO_DESCRIPTIONS[scenario])
with st.sidebar.expander("Как играть", expanded=True):
    st.markdown(
        """
        1. Выберите режим.
        2. Получите случайный сектор Москвы.
        3. Нажмите по точке на карте.
        4. Зафиксируйте выбор в панели справа.
        5. Откройте ответ модели и сравните свой выбор с локальным оптимумом внутри этого сектора.
        """
    )
with st.sidebar.expander("Как считается результат"):
    st.markdown(
        """
        - `Player Score = 100 * ваш objective / лучший objective модели`
        - Сравнение идет не по всей Москве, а по случайной локальной задаче.
        - Поэтому каждый новый сектор — это новая мини-игра.
        """
    )

sector_data = st.session_state["game_sector_data"]
sector_cells = sector_data["sector_cells"].copy()
local_best = pd.Series(sector_data["local_best"])
bounds = sector_data["bounds"]

st.markdown(
    """
    <div class="hero">
      <h1>VTB ATM Placement Game</h1>
      <p>Каждый раунд дает новый сектор Москвы. Ваша задача — выбрать внутри него более удачную точку для банкомата, чем предложит интуиция большинства пользователей.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

layout_left, layout_right = st.columns([2.8, 1.2], gap="large")

with layout_left:
    game_map = build_map(sector_cells, local_best, bounds)
    result = st_folium(game_map, width=None, height=780, returned_objects=["last_clicked"])

    if result and result.get("last_clicked"):
        clicked = result["last_clicked"]
        pending = nearest_cell(clicked["lat"], clicked["lng"], sector_cells)
        pending_h3 = pending["h3_index"]
        if st.session_state.get("last_clicked_h3") != pending_h3:
            st.session_state["last_clicked_h3"] = pending_h3
            st.session_state["pending_guess"] = pending.to_dict()
            if st.session_state.get("locked_guess") is None:
                st.rerun()

with layout_right:
    st.markdown(
        """
        <div class="soft-card">
          <b>Текущий раунд</b><br>
          Сектор выбран случайно и содержит достаточное число кандидатных точек, чтобы выбор не сводился к одному очевидному ответу.
        </div>
        """,
        unsafe_allow_html=True,
    )

    action_1, action_2 = st.columns(2)
    with action_1:
        if st.button(
            "Зафиксировать точку",
            use_container_width=True,
            disabled=st.session_state.get("pending_guess") is None,
        ):
            st.session_state["locked_guess"] = st.session_state.get("pending_guess")
            st.session_state["reveal_answer"] = False
            st.rerun()
    with action_2:
        if st.button(
            "Показать ответ",
            use_container_width=True,
            disabled=st.session_state.get("locked_guess") is None,
        ):
            st.session_state["reveal_answer"] = True
            st.rerun()

    if st.button("Новый случайный сектор", use_container_width=True):
        initialize_or_refresh_sector(scenario, sector_radius_km, force_new=True)
        st.rerun()

    render_status_panel(scenario, local_best)
