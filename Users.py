from pydantic import BaseModel

class User(BaseModel):
    Age: int
    Gender: str
    Height_cm: float
    Weight_kg: float
    Fitness_Goals: str
    Fitness_Level: str
    Days_Available:str 
    Time_per_Session_mins: int
    Exercise_Preferences: str
    Gym_Equipment:str
    