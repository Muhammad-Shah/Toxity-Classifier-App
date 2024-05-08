from fastapi import FastAPI, Request, Response
import pickle
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Load the TF-IDF vectorizer and the trained random forest model
with open("tf_idf.pickle", "rb") as f:
    tf_idf = pickle.load(f)

with open("trained_models_rf.pkl", "rb") as f:
    model = pickle.load(f)

categories = ["toxic", "severe_toxic", "obscene",
              "threat", "insult", "identity_hate"]


class TextInput(BaseModel):
    text: str


@app.post("/predict")
async def predict(input_data: TextInput):
    """
    Predict the toxicity of a given text
    """
    text = input_data.text
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        print(text)
        # Convert the text to a TF-IDF vector
        vector = tf_idf.transform([text])
        # Make predictions using the trained model
        prediction = model.predict(vector)[0]
        # Create a dictionary with the category names and predictions
        output = {category: int(prediction[i])
                  for i, category in enumerate(categories)}

        # Return the output as a JSON response
        return output
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="Error processing request: " + str(e))


@app.post("/predict_batch")
async def predict_batch(texts: str):
    """
    Predict the toxicity of a batch of texts
    """

    # Convert the texts to TF-IDF vectors
    vectors = tf_idf.transform(texts)

    # Make predictions using the trained model
    predictions = model.predict(vectors)

    # Create a list of dictionaries with the category names and predictions
    output = []
    for prediction in predictions:
        output.append({category: int(prediction[i])
                      for i, category in enumerate(categories)})

    # Return the output as a JSON response
    return output
