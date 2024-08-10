import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self, bert_model='bert-base-uncased', embedding_dim=768, num_classes_A=3, num_classes_B=3):
        super(MultiTaskSentenceTransformer, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.activation = nn.Tanh()
        
        # Task A: Sentence Classification
        self.task_A_classifier = nn.Linear(embedding_dim, num_classes_A)
        
        # Task B: Sentiment Analysis
        self.task_B_classifier = nn.Linear(embedding_dim, num_classes_B)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        sentence_embedding = self.activation(self.linear(pooled_output))
        
        # Task A output
        task_A_output = self.task_A_classifier(sentence_embedding)
        
        # Task B output
        task_B_output = self.task_B_classifier(sentence_embedding)
        
        return sentence_embedding, task_A_output, task_B_output

#Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = MultiTaskSentenceTransformer()

# Sample sentences
sample_sentences = [
    "Breaking news: Major earthquake hits coastal city.",
    "What is the capital of France?",
    "The weather is beautiful today.",
    "I absolutely love this new restaurant!",
    "Unfortunately, the movie was a huge disappointment."
]

# Tokenize and encode sentences
encoded_input = tokenizer(sample_sentences, padding=True, truncation=True, return_tensors='pt')

# Generate embeddings and task outputs
with torch.no_grad():
    sentence_embeddings, task_A_outputs, task_B_outputs = model(encoded_input['input_ids'], encoded_input['attention_mask'])

# Define class labels
task_A_labels = ['News', 'Question', 'Statement']
task_B_labels = ['Positive', 'Negative', 'Neutral']

#Print results
for i, sentence in enumerate(sample_sentences):
    print(f"Sentence: {sentence}")
    print(f"Embedding shape: {sentence_embeddings[i].shape}")
    print(f"Embedding (first 5 values): {sentence_embeddings[i][:5]}")
    
    task_A_pred = task_A_labels[torch.argmax(task_A_outputs[i]).item()]
    print(f"Task A (Sentence Classification) prediction: {task_A_pred}")
    
    task_B_pred = task_B_labels[torch.argmax(task_B_outputs[i]).item()]
    print(f"Task B (Sentiment Analysis) prediction: {task_B_pred}")
    print()