# importing libraries
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import pickle

# loading local data
df = pd.read_csv("user.csv")



# Data Preparation
df.drop(columns=["name","muscle_group"],inplace=True)
features = df.drop(columns="workout").columns

# categorical and numerical featuers
num_features = [col for col in features if df[col].dtype != "object"]

cat_features = [col for col in features if df[col].dtype=="object"]


X = df.drop(columns=["workout"])
y = df["workout"]


# transforming the data part
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_features),  # pass through numeric features
        ('cat', OneHotEncoder(), cat_features)  # apply OneHotEncoder to categorical features
    ])


# pipelining the transformed data with the classifier model into a single model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Training part
model.fit(X,y)


with open("classifier.pkl","wb") as pickle_out:
    pickle.dump(model,pickle_out)
    pickle_out.close()