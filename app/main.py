from fastapi import FastAPI, Request
import pickle
from app.code import Predict_car


app = FastAPI()

m = pickle.load(open(r"/CarBrandClass/model/CarBranClass.pkl", "rb"))
url = "http://localhost:8080/api/gethog"


@app.get("/")
def root():
    return {"messages": "this is my api"}


@app.get("/api/carbrand")
async def prediect_car(request: Request):
    item = await request.json()
    item_img = item["img"]
    predict = Predict_car(m, item_img)
    return {"Predict": predict}
