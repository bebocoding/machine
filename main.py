# imports
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from Users import User
import numpy as np
import pandas as pd
import pickle
import json
import subprocess





subprocess.run(["python", "model.py"])
# Create the app object

app = FastAPI()


# Set up CORS middleware
origins = ["*"]  
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




pickle_in = open("classifier.pkl",'rb')
classifier = pickle.load(pickle_in)

@app.get('/')
async def index():
    return {"message":"hello stranger :D"}


@app.get('/{user_name}')
async def get_name(user_name):
    return {"message": f'hello {user_name}'}


# using model for prediction
@app.post('/predict')
async def predict_workout(user_data:User):
    user_data = user_data.model_dump()
    user_df = pd.DataFrame([user_data])
    prediction = classifier.predict(user_df)[0]
    print(f'chosen plan {prediction} for {user_data.Name}')

    if prediction == 1.0:
        plan = pd.read_csv("./plans/plan1.csv")
    elif prediction == 2.0:
        plan = pd.read_csv("./plans/plan2.csv")
    json_plan = plan.to_json(orient="records") # contains escape literals
    exercise_data = json.loads(json_plan) # clean dictionary

    return exercise_data



