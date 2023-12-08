from fastapi.testclient import TestClient
from tests.utils import get_current_model_name, make_embedding_request

def test_text_embedding(client):
    model = get_current_model_name(client)
    input = 'test text'
    response = make_embedding_request(client, input, model, 'float')
    assert len(response['data']) == 1
    embedding = response['data'][0]['embedding']
    assert len(embedding)>0

def test_text_list_embedding(client):
    model = get_current_model_name(client)
    input = ['test text', 'test text']
    response = make_embedding_request(client, input, model, 'float')
    assert len(response['data']) == len(input)
    embedding = response['data'][0]['embedding']
    assert len(embedding)>0

def test_int_embedding(client):
    model = get_current_model_name(client)
    input = [1,2,3,4]
    response = make_embedding_request(client, input, model, 'float')
    assert len(response['data']) == 1
    embedding = response['data'][0]['embedding']
    assert len(embedding)>0

def test_int_embedding(client):
    model = get_current_model_name(client)
    input = [[1,2,3,4], [5,6,7,8]]
    response = make_embedding_request(client, input, model, 'float')
    assert len(response['data']) == len(input)
    embedding = response['data'][0]['embedding']
    assert len(embedding)>0
