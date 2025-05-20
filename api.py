from fastapi import FastAPI, HTTPException
from src import model_utils

app = FastAPI()

@app.get("/game/{game_id}/epv")
def epv(game_id: str):
    """Return EPV values for each possession."""
    try:
        df = model_utils.sequence_epv(game_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return df["epv"].tolist()


@app.get("/game/{game_id}/swing")
def swing(game_id: str):
    """Return top-20 swing possessions."""
    try:
        df = model_utils.swing(game_id, top_n=20)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return df.to_dict(orient="records")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
