from pydantic import BaseModel

class User(BaseModel):
    name:str
    muscle_group:str
    weight_kg:float
    height_cm:float
    level:str
    gender:str
    