import torch 
import base64 
import struct
from typing import List 

def mean_pool(embeddings, mask):
    pooled = ((embeddings * mask.unsqueeze(-1)).sum(1)) / torch.clamp(mask.sum(-1).unsqueeze(-1), min=1e-9)
    return pooled

def max_pool(embeddings, mask):
    fill_val = torch.finfo(embeddings.dtype).min
    
    if mask.ndim==2:
        mask = mask.unsqueeze(-1)
        
    filled_embeddings = embeddings.masked_fill(~(mask.bool()), fill_val)
    pooled = torch.max(filled_embeddings, dim=1)[0]
    return pooled

def eos_pool(embeddings, mask):
    eos_idxs = mask.sum(-1)-1
    pooled = embeddings[torch.arange(embeddings.shape[0]), eos_idxs]
    return pooled

def bos_pool(embeddings, mask):
    pooled = embeddings[:, 0]
    return pooled

pool_dict = {
    'mean' : mean_pool,
    'max' : max_pool, 
    'eos' : eos_pool, 
    'bos' : bos_pool
}

def pool_router(embeddings, mask, pool_type):
    return pool_dict[pool_type](embeddings, mask)


def encode_embedding(embedding: List[float]):
    packed_bytes = struct.pack('<' + 'f' * len(embedding), *embedding)
    base64_string = base64.b64encode(packed_bytes).decode('utf-8')
    return base64_string

def decode_embedding(base64_embedding: str):
    decoded_bytes = base64.b64decode(base64_embedding)
    embedding = list(struct.unpack('<' + 'f' * (len(decoded_bytes) // 4), decoded_bytes))
    return embedding 
