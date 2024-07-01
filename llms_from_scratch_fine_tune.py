"""
    A Complete Guide to BERT with Code
    https://towardsdatascience.com/a-complete-guide-to-bert-with-code-9f87602e4a11
    
    osobny venv ze starymi bibliotekami
    
    $ python3 -m venv venvTransformers
    $ source source venvTransformers/bin/activate
    $ pip install scipy==1.10.1
    $ pip install torch
    $ pip install transformers


"""
# from pyglet.gl.wgl import PROC
# from numba.core.types import none

FAST_DBG_TEST = True
FAST_DBG_TEST = False

import os
import gc
import torch
import pandas as pd

import numpy as np 

# from transformers import AutoModel, AutoTokenizer
from transformers import BertTokenizer, BertForSequenceClassification
# from transformers import DistilBertTokenizerFast, DistilBertModel

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from torch.optim import AdamW
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup


batch_size = 8
batch_size = 16
batch_size = 12

os.getcwd()


# -----------------------------------
def calculate_accuracy(preds, labels):
    """ Calculate the accuracy of model predictions against true labels.

    Parameters:
        preds (np.array): The predicted label from the model
        labels (np.array): The true label

    Returns:
        accuracy (float): The accuracy as a percentage of the correct
            predictions.
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    accuracy = np.sum(pred_flat == labels_flat) / len(labels_flat)

    return accuracy


# -----------------------------------
# https://huggingface.co/datasets/scikit-learn/imdb/blob/main/IMDB%20Dataset.csv
df = pd.read_csv('~/Pobrane/IMDB Dataset.csv')
df.head()

# Remove the break tags (<br />)
df['review_cleaned'] = df['review'].apply(lambda x: x.replace('<br />', ''))

# Remove unnecessary whitespace
df['review_cleaned'] = df['review_cleaned'].replace('\s+', ' ', regex=True)

# Compare 72 characters of the second review before and after cleaning
print('Before cleaning:')
print(df.iloc[1]['review'][0:72])

print('\nAfter cleaning:')
print(df.iloc[1]['review_cleaned'][0:72])

df['sentiment_encoded'] = df['sentiment'].\
    apply(lambda x: 0 if x == 'negative' else 1)
df.head()



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
print(tokenizer)

# Encode a sample input sentence
sample_sentence = 'I liked this movie'
token_ids = tokenizer.encode(sample_sentence, return_tensors='np')[0]
tokenizer.convert_ids_to_tokens(token_ids)

review = df['review_cleaned'].iloc[0]
review 

token_ids = tokenizer.encode(
    review,
    max_length = 512,
    padding = 'max_length',
    truncation = True,
    return_tensors = 'pt')

tokenizer.convert_ids_to_tokens(token_ids[0])

batch_encoder = tokenizer.encode_plus(
    review,
    max_length = 512,
    padding = 'max_length',
    truncation = True,
    return_tensors = 'pt')

batch_encoder.keys()

batch_encoder['attention_mask']

token_ids = []
attention_masks = []

print("encoding..")
id = 0
# Encode each review
for review in df['review_cleaned']:
    batch_encoder = tokenizer.encode_plus(
        review,
        max_length = 512,
        padding = 'max_length',
        truncation = True,
        return_tensors = 'pt')

    token_ids.append(batch_encoder['input_ids'])
    attention_masks.append(batch_encoder['attention_mask'])
    id = id + 1
    if id % (len(df) // 100) == 0:
        proc = id * 100 / len(df)
        print(f"progress {round(proc, 1)}%")

    if id > 512 and FAST_DBG_TEST:
        break

print("..encoding done")

# Convert token IDs and attention mask lists to PyTorch tensors
token_ids = torch.cat(token_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

attention_masks.shape
token_ids.shape


val_size = 0.1

# Split the token IDs
train_ids, val_ids = train_test_split(
                        token_ids,
                        test_size=val_size,
                        shuffle=False)
train_ids.shape
val_ids.shape 


# Split the attention masks
train_masks, val_masks = train_test_split(
                            attention_masks,
                            test_size=val_size,
                            shuffle=False)

# Split the labels
labels = torch.tensor(df['sentiment_encoded'].values)

if FAST_DBG_TEST:
    labels = labels.narrow(0, 0, token_ids.shape[0]) 

train_labels, val_labels = train_test_split(
                                labels,
                                test_size=val_size,
                                shuffle=False)

# Create the DataLoaders
train_data = TensorDataset(train_ids, train_masks, train_labels)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size = batch_size)
val_data = TensorDataset(val_ids, val_masks, val_labels)
val_dataloader = DataLoader(val_data, batch_size = batch_size)


# Check if GPU is available for faster training time
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')

# device = torch.device('cpu')
    
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
# model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

EPOCHS = 2

# Optimizer
optimizer = AdamW(model.parameters())

# Loss function
loss_function = nn.CrossEntropyLoss()

num_training_steps = EPOCHS * len(train_dataloader)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps)


print(f"len(train_dataloader) : {len(train_dataloader)}")


for batch in train_dataloader:
    batch_token_ids = batch[0]
    batch_attention_mask = batch[1]
    batch_labels = batch[2]
    break

train_dataloader


if False:
    model.eval()
    loss, logits = model(
                batch_token_ids,
                token_type_ids = None,
                attention_mask=batch_attention_mask,
                labels=batch_labels,
                return_dict=False)
    for item in model.parameters(True):
        print(item)

train_accuracy = 0;

for epoch in range(0, EPOCHS):
    
    print(f"----- epoch: {epoch}")
    model.train()
    training_loss = 0
    id = 0
    
    for batch in train_dataloader:

        batch_token_ids = batch[0].to(device)
        batch_attention_mask = batch[1].to(device)
        batch_labels = batch[2].to(device)

        if False:
            print(batch_token_ids.shape)
            print(batch_attention_mask.shape)
            print(batch_labels.shape)

        model.zero_grad()


        loss, logits = model(
            batch_token_ids,
            token_type_ids = None,
            attention_mask=batch_attention_mask,
            labels=batch_labels,
            return_dict=False)

        training_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            i_label_ids = batch_labels.to('cpu').numpy()
            i_logits = logits.detach().cpu().numpy()
            train_accuracy += calculate_accuracy(i_logits, i_label_ids)

        if True:
            del batch_token_ids
            del batch_attention_mask
            del batch_labels
            del loss
            del logits
            gc.collect()
            with torch.no_grad():
                torch.cuda.empty_cache()

        if False:
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) and obj.is_cuda or (hasattr(obj, 'data') and torch.is_tensor(obj.data) and obj.data.is_cuda):
                        print(type(obj), obj.size())
                except:
                    pass
    

        id = id + 1
        if id % max(1, len(train_dataloader) // 100) == 0:
            proc = id * 100 / len(train_dataloader)
            print("--")
            print(f"progress {round(proc, 1)}%")
            print(f"training_loss {round(training_loss/id, 3)}")
            print(f"train_accuracy = {round(train_accuracy / id, 3)} ")
            print(f"GPU mem free: {100 * torch.cuda.mem_get_info()[0] // torch.cuda.mem_get_info()[1]}%")


    if FAST_DBG_TEST:
        break

print("-- --")
training_loss = training_loss / len(train_dataloader)
print(f'training_loss = {round(training_loss, 3)}')

train_accuracy = train_accuracy / len(train_dataloader)
print(f"train_accuracy = {round(train_accuracy, 3)} ")



# -----------------------------------
model.eval()

val_loss = 0
val_accuracy = 0

for batch in val_dataloader:

    batch_token_ids = batch[0].to(device)
    batch_attention_mask = batch[1].to(device)
    batch_labels = batch[2].to(device)

    with torch.no_grad():
        (loss, logits) = model(
            batch_token_ids,
            attention_mask = batch_attention_mask,
            labels = batch_labels,
            token_type_ids = None,
            return_dict=False)

    logits = logits.detach().cpu().numpy()
    label_ids = batch_labels.to('cpu').numpy()
    val_loss += loss.item()
    val_accuracy += calculate_accuracy(logits, label_ids)


print("-- --")

val_loss = val_loss /  len(val_dataloader)
print(f'val_loss = {round(val_loss, 3)}')

val_accuracy = val_accuracy / len(val_dataloader)
print(f"val_accuracy = {round(val_accuracy, 3)} ")
    
print("-- DONE --")

# batch_size = 12
# progress 48.3%
# training_loss 0.744
# train_accuracy = 0.505
# GPU mem free: 48%
    
    
