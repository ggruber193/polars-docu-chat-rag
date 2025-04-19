from transformers import AutoModel, AutoTokenizer
from torch import Tensor
from torch import functional as F

from src.config import EMBEDDING_MODEL
from src.utils import batched


class TextEmbedder:
    def __init__(self, modelname=EMBEDDING_MODEL, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)
        self.model = AutoModel.from_pretrained(modelname)
        self.max_length = max_length

    @staticmethod
    def average_pool(last_hidden_states: Tensor,
                     attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def embed_text(self, text: str | list[str], batch_size=128):
        if isinstance(text, str):
            text = [text]

        outputs = []

        for batch in batched(text, n=batch_size):
            batch_dict = self.tokenizer(batch, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt')
            output = self.model(**batch_dict)
            embeddings = self.average_pool(output.last_hidden_state, batch_dict['attention_mask'])

            # embeddings = F.norm(embeddings, p=2, dim=1)
            # scores = (embeddings[:1] @ embeddings[1:].T) * 100

            embeddings = embeddings.tolist()
            outputs += embeddings
        return outputs
