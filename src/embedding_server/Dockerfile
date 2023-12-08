FROM python:3.9-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

COPY ./tests /code/tests 
RUN chmod +x /code/tests/start-tests.sh

CMD uvicorn app.main:app --host $EMBEDDING_SERVER_HOST --port $EMBEDDING_SERVER_PORT --workers $EMBEDDING_SERVER_WORKERS --timeout-keep-alive $EMBEDDING_SERVER_TIMEOUT