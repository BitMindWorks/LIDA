import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import string


# Load saved vocabulary and metadata
with open("model_config.json", "r") as f:
    config = json.load(f)

vocab = config["vocab"]
lang_to_idx = config["lang_to_idx"]
idx_to_lang = {int(k): v for k, v in config["idx_to_lang"].items()}
max_length = config["max_length"]

# Define model class
class LanguageLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LanguageLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

# Load model with correct vocabulary size
vocab_size = len(vocab) + 1  # Use exact vocab size from config
embed_dim = 64
hidden_dim = 128
output_dim = len(lang_to_idx)

model = LanguageLSTM(vocab_size, embed_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("language_model.pth"))
model.eval()

# Preprocessing functions
def preprocess(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text.split()

def encode_text(text):
    return [vocab.get(word, vocab["<UNK>"]) for word in preprocess(text)]

def predict_language(text):
    # Encoding the text and padding
    encoded = encode_text(text)
    padded = encoded + [0] * (max_length - len(encoded))
    tensor_input = torch.tensor([padded], dtype=torch.long)

    with torch.no_grad():
        # Getting the model output
        output = model(tensor_input)
        
        # Apply softmax to convert logits to probabilities
        probabilities = F.softmax(output, dim=1).squeeze().cpu().numpy()

    # Get predicted language
    predicted_lang_idx = torch.argmax(output).item()

    # Mapping of languages to index (assuming idx_to_lang is a dictionary)
    predicted_lang = idx_to_lang[predicted_lang_idx]
    
    # Get probability distribution of languages
    language_probs = {lang: prob * 100 for lang, prob in zip(idx_to_lang.values(), probabilities)}

    return predicted_lang, language_probs