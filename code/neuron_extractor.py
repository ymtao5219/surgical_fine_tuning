import torch
from transformers import AutoTokenizer, AutoModel

class NeuronExtractor: 
    
    '''
    Take a trained model (either pretrained or finetuned) and N test sentences and return the cls embedding of shape (N, d_model). 
    '''
    
    def __init__(self, model, tokenizer, is_split_into_words=False) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_split_into_words = is_split_into_words
        self.model.to(self.device)
    
    @torch.no_grad()
    def extract_layer_embedding(self, sentences, layer_num=-1):
        input_ids = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", is_split_into_words=self.is_split_into_words).to(self.device)
        outputs = self.model(**input_ids, output_hidden_states=True)
        # Extract the CLS embedding for the specified layer
        cls_embedding = outputs.hidden_states[layer_num][:, 0, :].detach().cpu().numpy()
        return cls_embedding