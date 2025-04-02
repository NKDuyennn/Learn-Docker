from fastapi import FastAPI
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "ML model deployment"}

@app.post("/predict")
def predict(data: dict):
    prediction = model.predict([data["feature"]])[0]
    predicted_class = class_names[prediction]
    return {"message": predicted_class}