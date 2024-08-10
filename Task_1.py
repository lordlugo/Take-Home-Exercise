import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

class SentenceTransformer(nn.Module):
    def __init__(self, bert_model='bert-base-uncased', embedding_dim=768):
        super(SentenceTransformer, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.activation = nn.Tanh()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        sentence_embedding = self.activation(self.linear(pooled_output))
        return sentence_embedding

#Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = SentenceTransformer()


sample_sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "I love machine learning and natural language processing.",
    "Transformers have revolutionized the field of NLP."
]

# Tokenize and encode sentences
encoded_input = tokenizer(sample_sentences, padding=True, truncation=True, return_tensors='pt')

#Generate embeddings
with torch.no_grad():
    sentence_embeddings = model(encoded_input['input_ids'], encoded_input['attention_mask'])

# Print embeddings
for i, sentence in enumerate(sample_sentences):
    print(f"Sentence: {sentence}")
    print(f"Embedding shape: {sentence_embeddings[i].shape}")
    print(f"Embedding: {sentence_embeddings[i][:5]}")  # Showing first 5 values
    print()