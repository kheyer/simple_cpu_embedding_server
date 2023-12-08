from fastapi.testclient import TestClient

import base64 
import struct

def get_current_model_name(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    response_dict = response.json()
    assert len(response_dict)>0
    return response_dict[0]['id']

def make_embedding_request(client, input, model, encoding_format):
    request_data = {
        'input' : input, 
        'model' : model,
        'encoding_format' : encoding_format
    }
    response = client.post('/v1/embeddings', json=request_data)
    assert response.status_code == 200
    return response.json()

def decode_embedding(base64_embedding):
    decoded_bytes = base64.b64decode(base64_embedding)
    embedding = list(struct.unpack('<' + 'f' * (len(decoded_bytes) // 4), decoded_bytes))
    return embedding 

