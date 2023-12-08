from fastapi.testclient import TestClient
from tests.utils import get_current_model_name, make_embedding_request, decode_embedding


def test_base64_encode(client):
    model = get_current_model_name(client)
    input = 'test text'
    response1 = make_embedding_request(client, input, model, 'float')
    response2 = make_embedding_request(client, input, model, 'base64')

    embedding1 = response1['data'][0]['embedding']

    base64_embedding = response2['data'][0]['embedding']
    embedding2 = decode_embedding(base64_embedding)

    assert embedding1 == embedding2
