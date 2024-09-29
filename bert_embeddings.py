from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
# Example input text

text = "Hello, how are you doing today?"

# Tokenize input text and add special tokens
input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])

# Generate BERT embeddings
with torch.no_grad():
    outputs = model(input_ids)
    embeddings = outputs[0][0]

# Print embeddings
print(embeddings)
