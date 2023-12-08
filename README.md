# Simple CPU Embedding Server

This repository hosts a basic FastAPI embedding inference server designed for CPU-based environments. 
It's compatible with the OpenAI API and utilizes Hugging Face models for generating embeddings. 
The server is optimized for running CPU inference on multiple concurrent requests without batching.

## Setup

To set up:
1. Clone the repo 
2. Update `simple_cpu_embedding_server/src/.env` to set the model you want to use
3. Start the server with docker compose 

```
git clone https://github.com/kheyer/simple_cpu_embedding_server

cd simple_cpu_embedding_server/src

docker-compose up -d --build

docker-compose exec embedding_server tests/start-tests.sh
```

## Making Requests

The server follows the [OpenAI Embedding API](https://platform.openai.com/docs/api-reference/embeddings/create). 
The request payload can be any of `Union[str, List[str], List[int], List[List[int]]]`. Response embeddings 
can be returned as a list of floats or a base64 encoded string.

API docs can be found at `http://localhost:{server_port}/docs#/`

Requests can be made in several ways:

### Curl

```
curl http://localhost:7860/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "The food was delicious and the waiter...",
    "model": "bert-base-uncased",
    "encoding_format": "float"
  }'
```

### OpenAI API Client

```python
from openai import OpenAI

client = OpenAI(api_key='placeholder', # a dummy value is required
                base_url='http://localhost:7860/v1')

response = client.embeddings.create(
                model="bert-base-uncased",
                input='The food was delicious and the waiter...',
                encoding_format='float'
            )
```

### Python Requests

```python
import requests

request_data = {
    'input' : 'The food was delicious and the waiter...',
    'model' : 'bert-base-uncased',
    'encoding_format' : 'float
}

response = requests.post('http://localhost:7860/v1/embeddings', json=request_data)
```

### Concurrent Async Requests With HTTPX

For CPU inference, it is advantageous to make many concurrent requests. We can use 
`httpx` to make async requests, combined with a `asyncio` `Semaphore` to control the 
max number of concurrent requests.

```python
import httpx 
import asyncio

texts = [
    'The food was delicious and the waiter...',
    ...
]

requests = [
    {
        'input' : i,
        'model' : 'bert-base-uncased',
        'encoding_format' : 'float
    }
    for i in texts
]

async def post_request(data, url=None):
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data)

    return response.json()

async def concurrency_bounded_func(semaphore, func, input, kwargs):
    async with semaphore:
        output = await func(input, **kwargs)
    return output

async def concurrency_wrapper(concurrency, func, iterable, kwargs):
    semaphore = asyncio.Semaphore(concurrency)
    
    tasks = [concurrency_bounded_func(semaphore, func, item, kwargs) for item in iterable]
    results = await asyncio.gather(*tasks)
    return results

max_concurrent_requests = 8

response = await concurrency_wrapper(
                                    max_concurrent_requests, 
                                    post_request, 
                                    requests, 
                                    {'url' : 'http://localhost:7860/v1/embeddings'})
```

## Performance

Performance can be tuned through the `EMBEDDING_SERVER_WORKERS` and `EMBEDDING_SERVER_THREADS_PER_WORKER` 
parameters in `simple_cpu_embedding_server/src/.env`.

`EMBEDDING_SERVER_WORKERS` controls the number of workers. More workers gives more concurrency, but comes 
at a cost of more memory overhead as each worker loads a copy of the model.

`EMBEDDING_SERVER_THREADS_PER_WORKER` controls the number of threads available to each worker.

For a high concurrency, high memory loadout, set `EMBEDDING_SERVER_WORKERS` to the number of CPUs 
on your machine and `EMBEDDING_SERVER_THREADS_PER_WORKER=1`.

For lower concurrency and lower memory, reduce `EMBEDDING_SERVER_WORKERS` and increase `EMBEDDING_SERVER_THREADS_PER_WORKER`.

Stress test the server with concurrent requests up to `EMBEDDING_SERVER_WORKERS`. If you see the workers 
are not maxing out CPU usage, reduce `EMBEDDING_SERVER_THREADS_PER_WORKER`.


The `.env` file has the `QUANTIZE` flag which will automatically quantize linear layers in the model. This 
is convenient for downloading models from the Huggingface hub, but is less efficient compared to quantizing 
the model prior to loading.


Downloading the model from the Huggingface hub (once per worker) adds to startup overhead. To avoid this, 
download the model once, save it to disk, and update `docker-compose.yml` to add the directory containing the 
model as a docker volume.
