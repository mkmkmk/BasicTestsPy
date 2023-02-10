
import torch
import torch.optim as optim

import nltk

from gpt import *
# nltk.download("punkt")

# Define the dataset
data = ["This is a sample sentence", "This is another sentence", "This is a third sentence"]

# Tokenize the dataset
tokens = []
for sentence in data:
    tokens.append(nltk.word_tokenize(sentence))

# Tokenize the dataset
# tokenizer = Tokenizer()
# tokens = tokenizer.tokenize(data)

# Define the model hyperparameters
# vocab_size = len(tokenizer.vocab)
vocab_size = len(set([word.lower() for token in tokens for word in token]))
d_model = 512
nhead = 8
num_layers = 6

# Instantiate the model
model = GPT(vocab_size, d_model, nhead, num_layers)


# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Define the number of training epochs
num_epochs = 10

for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()

    # Loop over the dataset
    for i, sample in enumerate(tokens):
        
        # tu się wysypuje!!
        # Convert the sample to a tensor
        sample = torch.tensor(sample)

        # Forward pass
        output = model(sample)

        # Compute the loss
        loss = criterion(output.view(-1, vocab_size), sample.view(-1))

        # Zero the gradients
        optimizer.zero_grad()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Print the average loss
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")


print("ok")

