from pydantic import BaseModel

from typing import List, Union, Optional
from typing_extensions import Literal

class Model(BaseModel):
    id: str 
    object: Literal["model"]='model'
    created: int 
    owned_by: str 

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str], List[int], List[List[int]]]
    model: str
    encoding_format: Literal["float", "base64"]='float'
    user: Union[str, Literal['user']]='user'

class Embedding(BaseModel):
    embedding: Union[List[float], str]
    index: int
    object: Literal["embedding"]='embedding'

class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    data: List[Embedding]
    model: str
    object: Literal["list"]
    usage: Usage

class StringRequest(BaseModel):
    input: Union[str, List[str]]

class PoolType(BaseModel):
    pool_type: Literal['mean', 'max', 'eos', 'bos']='mean'
