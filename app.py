import os
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

DATA_DIR = "data"
MODEL_DIR = "models"

@st.cache_data
def load_csv(game_id: str, model_tag: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{model_tag}_{game_id}.csv")
    return pd.read_csv(path)

@st.cache_resource
def load_model(model_tag: str) -> XGBClassifier:
    model_path = os.path.join(MODEL_DIR, f"{model_tag}_xgb.json")
    clf = XGBClassifier()
    if os.path.exists(model_path):
        clf.load_model(model_path)
    else:
        st.warning(f"Model file {model_path} not found.")
    return clf


def heat_map(df: pd.DataFrame):
    import altair as alt
    if "shot_distance_ft" not in df.columns:
        st.info("No shot distance info available for heat map.")
        return
    chart = alt.Chart(df).mark_rect().encode(
        alt.X("shot_distance_ft:Q", bin=alt.Bin(maxbins=30), title="Shot distance (ft)"),
        alt.Y("points_scored:Q", bin=alt.Bin(maxbins=4), title="Points scored"),
        alt.Color("count():Q", scale=alt.Scale(scheme="oranges"))
    )
    st.altair_chart(chart, use_container_width=True)


def main():
    st.title("Flowstate Basketball")

    game_id = st.sidebar.text_input("Game ID", "0022400001")

    model_choice = st.sidebar.radio(
        "Model",
        ("baseline", "sequence"),
        format_func=lambda x: "Baseline (memory‑0)" if x == "baseline" else "Sequence (memory‑3)",
    )

    memory_depth = st.sidebar.slider("Memory depth", 1, 7, 3)
    show_heat = st.sidebar.checkbox("Show heat-map overlay")

    df = load_csv(game_id, model_choice)
    model = load_model(model_choice)

    min_poss = int(df["poss_id"].min())
    max_poss = int(df["poss_id"].max())
    poss = st.slider("Timeline", min_poss, max_poss, min_poss)
    row = df[df["poss_id"] == poss]

    st.subheader(f"Possession {poss}")
    st.write(row)

    if model.get_booster().feature_names:
        features = pd.get_dummies(row.drop(columns=["points_scored"]), drop_first=True)
        for col in model.get_booster().feature_names:
            if col not in features.columns:
                features[col] = 0
        features = features[model.get_booster().feature_names]
        probs = model.predict_proba(features)[0]
        exp_pts = sum(p * i for i, p in enumerate(probs))
        st.metric("Expected points", f"{exp_pts:.2f}")
    else:
        st.info("Model not loaded; predictions unavailable.")

    if show_heat:
        st.subheader("Shot heat map")
        heat_map(df)

    st.sidebar.caption(
        "Memory depth slider is a placeholder; models are trained with fixed depth."
    )


if __name__ == "__main__":
    main()
