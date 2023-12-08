from fastapi import FastAPI, HTTPException
from typing import List 

from .model import InferenceModel
from .schemas import Model, EmbeddingRequest, EmbeddingResponse

model = InferenceModel()

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/v1/models', response_model=List[Model])
def get_models():
    return [model.model_schema]

@app.post('/v1/embeddings', response_model=EmbeddingResponse)
def embed(embedding_request: EmbeddingRequest):

    if embedding_request.model != model.model_name:
        raise HTTPException(status_code=404, detail=f'model {embedding_request.model} not found')
        
    return model.embed(embedding_request)

