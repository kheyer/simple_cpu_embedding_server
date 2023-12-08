import os 
import time 

import torch 
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding

from . import utils 
from .schemas import (
                        EmbeddingRequest, 
                        EmbeddingResponse, 
                        Embedding, 
                        Usage, 
                        StringRequest,
                        Model,
                        PoolType
                        )

torch.set_num_threads(int(os.environ.get('EMBEDDING_SERVER_THREADS_PER_WORKER')))
torch.set_grad_enabled(False)

class InferenceModel():
    def __init__(self):
        torch.set_num_threads(int(os.environ.get('EMBEDDING_SERVER_THREADS_PER_WORKER')))
        torch.set_grad_enabled(False)

        self.load_model()
        
        self.pool_type = os.environ.get('POOL_TYPE')
        PoolType(pool_type=self.pool_type) # check if pool type is valid 

        self.model_schema = Model(id=self.model_name, object='model', created=int(time.time()), owned_by='')

    def load_model(self):
        self.model_name = os.environ.get('MODEL_NAME')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=os.environ.get('HF_TOKEN'))
        self.collator = DataCollatorWithPadding(self.tokenizer, return_tensors='pt')

        self.model = AutoModel.from_pretrained(self.model_name, token=os.environ.get('HF_TOKEN'))
        if bool(os.environ.get('QUANTIZE')):
            self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)

        self.model.eval()
    
    def process_string_request(self, embedding_request: EmbeddingRequest):
        inputs = embedding_request.input
        if type(inputs) == str:
            inputs = [inputs]
            
        inputs = self.collator(self.tokenizer(inputs, truncation=True))
        return inputs 
    
    def process_token_request(self, embedding_request: EmbeddingRequest):
        inputs = embedding_request.input
        if type(inputs[0]) == int:
            inputs = [inputs]
            
        inputs = {'input_ids' : inputs}
        inputs = self.collator(inputs)
        return inputs
    
    def build_embedding_response(self, inputs, embeddings, encoding_format):
        total_tokens = inputs['attention_mask'].sum()
        usage = Usage(prompt_tokens=total_tokens, total_tokens=total_tokens)

        if encoding_format=='base64':
            embeddings = [utils.encode_embedding(i) for i in embeddings]

        embeddings = [Embedding(embedding=embeddings[i], index=i) for i in range(len(embeddings))]
                
        response = EmbeddingResponse(
                                    data=embeddings, 
                                    model=self.model_name, 
                                    object='list',
                                    usage=usage
                                    )
        return response
        
    def embed(self, embedding_request: EmbeddingRequest):
        try:
            StringRequest.model_validate(embedding_request.model_dump(include='input'))
            inputs = self.process_string_request(embedding_request)
        except:
            inputs = self.process_token_request(embedding_request)
            
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            full_embeddings = outputs.hidden_states[-1]
            mask = inputs['attention_mask']
            
            embeddings = utils.pool_router(full_embeddings, mask, self.pool_type)
            
        embeddings = embeddings.detach().cpu().tolist()
        return self.build_embedding_response(inputs, embeddings, embedding_request.encoding_format)
