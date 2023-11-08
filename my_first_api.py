# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("my_first_api")

# Create input/output pydantic models
input_model = create_model("my_first_api_input", **{'Hospitalizacion': 3, 'Antioboticos': 'SI', 'DX_pre_cx': 'S064', 'DX_R3': 'S060', 'UCI_UCE': 'SI', 'D_DX_MEDICO': 'TRASTORNOS MENTALES Y DEL COMPORTAMIENTO DEBIDOS AL USO DE MULTIPLES DROGAS Y AL USO DE OTRAS SUSTANCIAS PSICOACTIVAS  SINDROME DE DEPENDEND', 'SubgrupoOncologia': 'HEMORRAGIA EPIDURAL', 'DX_MUERTE': '-1', 'Transfusiones': 'NO'})
output_model = create_model("my_first_api_output", prediction=0.04739159)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
