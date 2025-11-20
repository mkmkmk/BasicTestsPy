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
# %%
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

#  pip install scikit-learn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from torch.optim import AdamW
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

# %%

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
    accuracy = (np.argmax(preds, axis=1) == labels).mean()
    return accuracy


# %%
# -----------------------------------
# https://huggingface.co/datasets/scikit-learn/imdb/blob/main/IMDB%20Dataset.csv
df = pd.read_csv('~/Pobrane/IMDB Dataset.csv')
len(df)
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

# %%
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

# encode_plus dodaje attention mask czyli które tokeny ignorować
# encode_plus method returns a dictionary (called a Batch Encoder in Hugging Face)
batch_encoder = tokenizer.encode_plus(
    review,
    max_length = 512,
    padding = 'max_length',
    truncation = True,
    return_tensors = 'pt')

batch_encoder.keys()
batch_encoder['attention_mask']

# %%
print("encoding..")
token_ids = []
attention_masks = []
lens = []
id = 0
# Encode each review
for review in df['review_cleaned']:
    lens.append(len(tokenizer.encode(review, return_tensors='np')[0]))

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

    if id >= 512 and FAST_DBG_TEST:
        break
print("..encoding done")


# %%
if False:
    import matplotlib.pyplot as plt
    lens = np.array(lens)
    np.mean(lens)
    np.median(lens)
    np.std(lens)
    np.percentile(lens, 75)
    plt.figure(figsize=(10, 6))
    plt.hist(lens, bins=30, alpha=0.7, color='skyblue', edgecolor='black')

token_ids[0].size()
len(token_ids)

# Convert token IDs and attention mask lists to PyTorch tensors
token_ids = torch.cat(token_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

attention_masks.shape
token_ids.shape

# val jak validation
val_size = 0.1

# Split the labels
labels = torch.tensor(df['sentiment_encoded'].values)
labels.shape

if FAST_DBG_TEST:
    labels = labels.narrow(0, 0, token_ids.shape[0]) 

# wspólny podział danych
train_ids, val_ids, train_masks, val_masks, train_labels, val_labels = train_test_split(
    token_ids, attention_masks, labels,
    test_size=0.1,
    # random_state=42,
    shuffle=True,            # teraz może być True
    stratify=labels.numpy()  # zachowuje balans klas
)

# Create the DataLoaders
train_data = TensorDataset(train_ids, train_masks, train_labels)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size = batch_size)
val_data = TensorDataset(val_ids, val_masks, val_labels)
val_dataloader = DataLoader(val_data, batch_size = batch_size)

# %%

# Check if GPU is available for faster training time
if True and torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')

gc.collect()

# device = torch.device('cpu')
if False:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from transformers import TRANSFORMERS_CACHE
    print(f"TRANSFORMERS_CACHE={TRANSFORMERS_CACHE}")


model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
# model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

# git clone https://huggingface.co/google-bert/bert-base-uncased
# local_dir = "/home/mkrej/dysk2T/NowyG/SourceCodeZNetu/Huggingface/bert-base-uncased"
# model = BertForSequenceClassification.from_pretrained(local_dir, num_labels=2).to(device)

# %%

EPOCHS = 2

# Optimizer
optimizer = AdamW(model.parameters(), lr = 2e-5)

# Loss function
loss_function = nn.CrossEntropyLoss()

num_training_steps = EPOCHS * len(train_dataloader)
num_warmup_steps = int(0.1 * num_training_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
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
                batch_token_ids.to(device),
                token_type_ids=None,
                attention_mask=batch_attention_mask.to(device),
                labels=batch_labels.to(device),
                return_dict=False)
    for item in model.parameters(True):
        print(item)

# %%

for epoch in range(EPOCHS):
    
    print(f"\n\n----- epoch: {epoch}\n")
    model.train()
    total_training_loss = 0
    total_train_accuracy = 0
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

        curr_training_loss = loss.item()
        total_training_loss += curr_training_loss
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            i_label_ids = batch_labels.detach().to('cpu').numpy()
            i_logits = logits.detach().cpu().numpy()
            curr_train_accuracy = calculate_accuracy(i_logits, i_label_ids)
            total_train_accuracy += curr_train_accuracy

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
            print(f"progress {proc:.1f}%")
            print(f"curr_training_loss {curr_training_loss:.3f}")
            print(f"curr_train_accuracy = {curr_train_accuracy:.3f} ")
            print(f"total_training_loss {total_training_loss / id:.3f}")
            print(f"total_train_accuracy = {total_train_accuracy / id:.3f} ")
            print(f"GPU mem free: {100 * torch.cuda.mem_get_info()[0] // torch.cuda.mem_get_info()[1]}%")
            print(i_logits)

    if FAST_DBG_TEST:
        break

print("-- --")
print(f'training_loss = {total_training_loss/ len(train_dataloader): .3f}')
print(f"train_accuracy = {total_train_accuracy/ len(train_dataloader):.3f} ")


# %%
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
    # val_accuracy += (np.argmax(logits, axis=1) == label_ids).mean()


print("-- --")
print(f'val_loss = {val_loss/ len(val_dataloader): .3f}')
print(f'val_accuracy = {val_accuracy/ len(val_dataloader): .3f}')

print("-- DONE --")

# stare i do bani:
#   batch_size = 12
#   progress 48.3%
#   training_loss 0.744
#   train_accuracy = 0.505
#   GPU mem free: 48%
#   -- --
#   val_loss = 0.693
#   val_accuracy = 0.494
#   -- DONE --
#
# nowe, działa!
#   ----- epoch: 0
#   ...
#   progress 99.7%
#   curr_training_loss 0.108
#   curr_train_accuracy = 0.917
#   total_training_loss 0.224
#   total_train_accuracy = 0.906
#   ...
#   ----- epoch: 1
#   ...
#   progress 99.7%
#   curr_training_loss 0.003
#   curr_train_accuracy = 1.000
#   total_training_loss 0.089
#   total_train_accuracy = 0.970
#   --
#   -- --
#   val_loss =  0.165
#   val_accuracy =  0.944
#   -- DONE --
# %%
if False:
    model.save_pretrained("bert-imdb-poc")
    tokenizer.save_pretrained("bert-imdb-poc")
    # %pwd
# %%
