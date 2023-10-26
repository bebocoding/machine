# imports
import uvicorn
from fastapi import FastAPI
from Users import User
import numpy as np
import pandas as pd
import pickle
import json

# Create the app object
app = FastAPI()
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
    print(user_data)
    user_df = pd.DataFrame([user_data])
    print(user_df)
    prediction = classifier.predict(user_df)[0]
    print(prediction)

    if prediction == 1.0:
        workout = pd.read_csv("./workouts/leg1.csv")
    else:
        workout = pd.read_csv("./workouts/leg2.csv")
    json_workout = workout.to_json(orient="records") # contains escape literals
    exercise_data = json.loads(json_workout) # clean dictionary

    return exercise_data




if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1',port=8000)

