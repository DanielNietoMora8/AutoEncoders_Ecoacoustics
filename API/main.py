from __future__ import absolute_import
import torch
import torch.nn as nn
from enum import Enum
from fastapi import FastAPI
from Models import ModelName
# from ..Dataloader import EcoDataTesis


app = FastAPI()

# a = ConvAE()

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name == ModelName.ConvAE:
        return {"model_name": model_name, "message": "Convolutional Autoencoder for images"}
    if model_name == ModelName.AE:
        return {"model_name": model_name, "message": "Autoencoder for 1D signal"}

    return {"model_name": model_name, "message": "Have some residuals"}



@app.get("/")
async def root():
    return {"First Time Trying FastAPI"}




