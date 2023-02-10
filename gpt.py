import torch
import torch.nn as nn

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(GPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x)
        return x

# # Define the model hyperparameters
# vocab_size = 10000
# d_model = 512
# nhead = 8
# num_layers = 6
# 
# # Instantiate the model
# model = GPT(vocab_size, d_model, nhead, num_layers)
