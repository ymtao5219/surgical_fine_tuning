import torch
from transformers import AutoTokenizer, AutoModel

class NeuronExtractor: 
    
    '''
    Take a trained model (either pretrained or finetuned) and N test sentences and return the cls embedding of shape (N, d_model). 
    '''
    
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
    
    # todo: batched inference, due to the limited memory of the GPU
    @torch.no_grad()
    def extract_cls(self, sentences):
        input_ids = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.model(**input_ids)
        cls_embedding = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        return cls_embedding