from __future__ import annotations

import math
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
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(15,39,71,0.08);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_game_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cells_by_scenario = pd.read_parquet(DATA_DIR / "game_cells_by_scenario.parquet")
    recommendations = pd.read_parquet(DATA_DIR / "game_recommendations.parquet")
    best = pd.read_parquet(DATA_DIR / "game_best_by_scenario.parquet")
    return cells_by_scenario, recommendations, best


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


def focus_cells(best_row: pd.Series, cells: pd.DataFrame, radius_km: float) -> pd.DataFrame:
    def distance(row: pd.Series) -> float:
        lat1, lon1 = best_row["lat"], best_row["lon"]
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


def build_sector_bounds(best_row: pd.Series, radius_km: float) -> list[list[float]]:
    lat_pad = radius_km / 110.574
    lon_pad = radius_km / (111.320 * max(0.2, abs(math.cos(math.radians(best_row["lat"])))))
    return [
        [best_row["lat"] - lat_pad, best_row["lon"] - lon_pad],
        [best_row["lat"] + lat_pad, best_row["lon"] + lon_pad],
    ]


def reset_state_if_needed(scenario: str) -> None:
    if st.session_state.get("game_scenario") != scenario:
        st.session_state["game_scenario"] = scenario
        st.session_state["locked_guess"] = None
        st.session_state["reveal_answer"] = False
        st.session_state["pending_guess"] = None


def build_map(best_row: pd.Series, sector_cells: pd.DataFrame, bounds: list[list[float]]) -> folium.Map:
    m = folium.Map(
        location=[best_row["lat"], best_row["lon"]],
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
        tooltip="Игровой сектор: кликните внутри него",
    ).add_to(m)

    for row in sector_cells.nlargest(40, "composite_model_score").itertuples():
        folium.CircleMarker(
            [row.lat, row.lon],
            radius=2.5,
            color="#8ea7bf",
            weight=1,
            fill=True,
            fill_color="#8ea7bf",
            fill_opacity=0.45,
        ).add_to(m)

    if st.session_state.get("locked_guess") is not None:
        guessed = st.session_state["locked_guess"]
        folium.CircleMarker(
            [guessed["lat"], guessed["lon"]],
            radius=9,
            color="#ff8c42",
            weight=3,
            fill=True,
            fill_color="#ff8c42",
            fill_opacity=0.95,
            tooltip="Ваш выбор",
        ).add_to(m)

    if st.session_state.get("reveal_answer"):
        folium.CircleMarker(
            [best_row["lat"], best_row["lon"]],
            radius=9,
            color="#0f2747",
            weight=3,
            fill=True,
            fill_color="#0f2747",
            fill_opacity=0.95,
            tooltip="Ответ модели",
        ).add_to(m)
        folium.PolyLine(
            locations=[
                [st.session_state["locked_guess"]["lat"], st.session_state["locked_guess"]["lon"]],
                [best_row["lat"], best_row["lon"]],
            ],
            color="#0f2747",
            weight=2,
            dash_array="8 8",
        ).add_to(m)

    return m


st.set_page_config(page_title="VTB ATM Game", page_icon="🎯", layout="wide")
apply_style()

cells_by_scenario, recommendations, best_by_scenario = load_game_data()
scenario = st.sidebar.selectbox("Игровой режим", best_by_scenario["scenario"].tolist())
sector_radius_km = st.sidebar.slider("Размер игрового сектора, км", min_value=1.6, max_value=4.0, value=2.6, step=0.2)
reset_state_if_needed(scenario)

best_row = best_by_scenario[best_by_scenario["scenario"] == scenario].iloc[0]
scenario_cells = cells_by_scenario[cells_by_scenario["scenario"] == scenario].copy()
sector_cells = focus_cells(best_row, scenario_cells, sector_radius_km)
if len(sector_cells) < 25:
    expanded = focus_cells(best_row, scenario_cells, sector_radius_km + 1.2)
    sector_cells = expanded.nsmallest(120, "focus_distance_km").copy()
bounds = build_sector_bounds(best_row, sector_radius_km)

st.sidebar.markdown("### Что это за режим")
st.sidebar.info(SCENARIO_DESCRIPTIONS[scenario])
with st.sidebar.expander("Как играть", expanded=True):
    st.markdown(
        """
        1. Слева выберите режим.
        2. На карте показан только один сектор Москвы, чтобы можно было думать на уровне улиц, а не всего города.
        3. Кликните по карте внутри сектора.
        4. Нажмите `Зафиксировать точку`.
        5. После этого нажмите `Показать ответ модели` и сравните свой выбор с оптимальным.
        """
    )
with st.sidebar.expander("Как считается победа"):
    st.markdown(
        """
        - `Player Score = 100 * ваш objective / лучший objective модели`
        - Чем ближе к 100, тем лучше выбор.
        - Сравнение идет в рамках выбранного режима, поэтому ответ модели в `social` и `profit` может отличаться.
        """
    )

st.markdown(
    """
    <div class="hero">
      <h1>VTB ATM Placement Game</h1>
      <p>Выберите лучшую точку для банкомата внутри одного сильного сектора Москвы. Оптимальная точка модели скрыта до тех пор, пока вы не сделаете собственную ставку.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

game_map = build_map(best_row, sector_cells, bounds)
result = st_folium(game_map, width=None, height=780, returned_objects=["last_clicked"])

if result and result.get("last_clicked"):
    clicked = result["last_clicked"]
    pending = nearest_cell(clicked["lat"], clicked["lng"], sector_cells)
    st.session_state["pending_guess"] = pending.to_dict()

action_cols = st.columns([1.2, 1.2, 3.6])
with action_cols[0]:
    if st.button("Зафиксировать точку", use_container_width=True, disabled=st.session_state.get("pending_guess") is None):
        st.session_state["locked_guess"] = st.session_state.get("pending_guess")
        st.session_state["reveal_answer"] = False
        st.rerun()
with action_cols[1]:
    if st.button("Показать ответ модели", use_container_width=True, disabled=st.session_state.get("locked_guess") is None):
        st.session_state["reveal_answer"] = True
        st.rerun()
with action_cols[2]:
    if st.button("Начать заново", use_container_width=False):
        st.session_state["locked_guess"] = None
        st.session_state["reveal_answer"] = False
        st.session_state["pending_guess"] = None
        st.rerun()

info_cols = st.columns([1.2, 1.0])
with info_cols[0]:
    st.markdown(
        """
        <div class="soft-card">
          <b>Что видно на карте</b><br>
          Синий прямоугольник — игровой сектор.<br>
          Маленькие точки — сильные кандидатные H3-ячейки внутри сектора.<br>
          Оранжевая точка появится после фиксации вашего выбора.<br>
          Темно-синяя точка появится только после раскрытия ответа модели.
        </div>
        """,
        unsafe_allow_html=True,
    )
with info_cols[1]:
    st.markdown(
        """
        <div class="soft-card">
          <b>Почему сектор, а не вся Москва</b><br>
          Игра должна проверять интуицию на реальной городской геометрии: уличная сеть, здания, близость к локальным якорям и пробелам сети ВТБ. Поэтому карта сразу приближена.
        </div>
        """,
        unsafe_allow_html=True,
    )

left, right = st.columns([1.0, 1.0])
with left:
    st.subheader("Ваш выбор")
    if st.session_state.get("pending_guess") and st.session_state.get("locked_guess") is None:
        pending = st.session_state["pending_guess"]
        st.info(
            f"Последний клик: {pending['h3_index']}. "
            "Если точка нравится, нажмите `Зафиксировать точку`."
        )
    elif st.session_state.get("locked_guess") is not None:
        guessed = pd.Series(st.session_state["locked_guess"])
        st.write(
            {
                "your_cell": guessed["h3_index"],
                "recommended_atm_type": guessed["recommended_atm_type"],
                "profit_score": round(float(guessed["profit_core_score"]), 3),
                "coverage_score": round(float(guessed["coverage_core_score"]), 3),
                "social_score": round(float(guessed["social_core_score"]), 3),
            }
        )
    else:
        st.info("Сделайте клик по сектору на карте, чтобы выбрать кандидатную точку.")

with right:
    st.subheader("Сравнение с моделью")
    if st.session_state.get("locked_guess") is None:
        st.info("Сначала зафиксируйте свою точку.")
    else:
        guessed = pd.Series(st.session_state["locked_guess"])
        user_score = objective_for_row(guessed, scenario)
        best_score = objective_for_row(best_row, scenario)
        pct = 100.0 * user_score / best_score if best_score > 0 else 0.0
        st.metric("Player Score", f"{pct:.1f}/100")
        if st.session_state.get("reveal_answer"):
            st.write(
                {
                    "model_best_cell": best_row["h3_index"],
                    "model_atm_type": best_row["recommended_atm_type"],
                    "delta_objective": round(best_score - user_score, 4),
                    "optimal_rank": int(best_row["selected_rank"]),
                }
            )
            if pct >= 92:
                st.success("Очень сильный выбор: вы почти попали в решение модели.")
            elif pct >= 75:
                st.warning("Хорошая догадка, но модель нашла вариант лучше.")
            else:
                st.error("Разрыв заметный: модель использует более сильную комбинацию локальных факторов.")
        else:
            st.caption("Ваш score уже посчитан, но ответ модели пока скрыт.")
