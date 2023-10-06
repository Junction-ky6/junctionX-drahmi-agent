from inference import model, prediction
from schema import Observation, Prediction

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def welcome():
    return {"message": "Hello From Drahmi!"}

@app.post("/recommendation")
def recommendation(observation: Observation) -> Prediction:
    return prediction(model, observation)