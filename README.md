# Flowstate‑Basketball
**A short‑memory Expected‑Possession‑Value engine + interactive film room proving that basketball possessions are *not* independent.**

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/<your‑org>/flowstate-basketball?quickstart=1)

---

## ⚡ Why this exists
Coaches and models still treat each trip down the floor like a coin flip. **Flowstate** captures what really happens: the *echoes* of the previous three possessions—outcomes, help positions, tempo spikes, micro‑fatigue—shape the next shot.  
Our memory‑3 model consistently beats the classic memory‑0 EPV by **≥ 4 % log‑loss** out‑of‑sample.

---

## 🔑 What you get
| Layer | Feature |
|-------|---------|
| **Data ingest** | `pbpstats` pulls play‑by‑play + shot chart for any NBA game. |
| **Feature stack** | Auto‑engineered rolling window: last‑3 outcomes, coverage tags, help‑XY centroid, tempo Δ, (opt) wearable load. |
| **Models** | Baseline EPV (memory‑0) vs SequenceEPV (memory‑3) — both XGBoost; CLI trains & saves models and prints log‑loss delta. |
| **API** | FastAPI: `/game/{id}/epv` (array) • `/game/{id}/swing` (top‑20 swing possessions). |
| **Dashboard** | Streamlit timeline scrubber + video clips; toggle models; slider to test memory depth 1‑7; heat‑map overlay. |
| **One‑click dev env** | GitHub Codespaces dev‑container: Python 3.11, Node 18, ffmpeg pre‑installed. |
| **Deploy** | Free Streamlit Cloud URL + Render/Fly API in one GitHub Actions push. |

---

## 🚀 Quick start (5 minutes)

```bash
# 1. open the repo in Codespaces (browser)
# 2. run:
python -m pip install -r requirements.txt   # should be no‑op in Codespace
python src/ingest.py 0022400001          # pulls a Bucks‑vs‑Celtics demo game
python src/train.py                         # trains & saves models, prints log‑loss
# open a second terminal
streamlit run app.py                        # launches the dashboard
```

## 🔄 Bulk ingest
Use `ingest_bulk.py` to pull several games at once and build features for each.

```bash
python src/ingest_bulk.py 0022400001 0022400002 0022400003
for gid in 0022400001 0022400002 0022400003; do
    python src/features.py "$gid"
    python src/sequence_features.py "$gid"
done
```
## License

This project is licensed under the [MIT License](LICENSE).
