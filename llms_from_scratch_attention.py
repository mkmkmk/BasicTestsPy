"""
    Self-Attention Explained with Code
    https://medium.com/p/d7a9f0f4d94e
    
    
    osobny venv ze starymi bibliotekami
    
    $ python3 -m venv venvTransformers
    $ source source venvTransformers/bin/activate
    $ pip install scipy==1.10.1
    $ pip install torch
    $ pip install transformers

"""

import os
import torch
from transformers import AutoModel, AutoTokenizer

def extract_le(sequence, tokenizer, model):
    """ Extract the learned embedding for each token in an input sequence.

    Tokenize an input sequence (string) to produce a tensor of token IDs.
    Return a tensor containing the learned embedding for each token in the
    input sequence.

    Args:
        sequence (str): The input sentence(s) to tokenize and extract
            embeddings from.
        tokenizer: The tokenizer used to produce tokens.
        model: The model to extract learned embeddings from.

    Returns:
        learned_embeddings (torch.tensor): A tensor containing tensors of
            learned embeddings for each token in the input sequence.
    """
    token_dict = tokenizer(sequence, return_tensors='pt')
    token_ids = token_dict['input_ids']
    learned_embeddings = model.embeddings.word_embeddings(token_ids)[0]

    # Additional processing for display purposes
    learned_embeddings = learned_embeddings.tolist()
    learned_embeddings = [[round(i,2) for i in le] \
                          for le in learned_embeddings]

    return learned_embeddings

def extract_te(sequence, tokenizer, model):
    """ Extract the tranformer embedding for each token in an input sequence.

    Tokenize an input sequence (string) to produce a tensor of token IDs.
    Return a tensor containing the transformer embedding for each token in the
    input sequence.

    Args:
        sequence (str): The input sentence(s) to tokenize and extract
            embeddings from.
        tokenizer: The tokenizer used to produce tokens.
        model: The model to extract learned embeddings from.

    Returns:
        transformer_embeddings (torch.tensor): A tensor containing tensors of
            transformer embeddings for each token in the input sequence.
    """
    token_dict = tokenizer(sequence, return_tensors='pt')

    with torch.no_grad():
        base_model_output = model(**token_dict)

    transformer_embeddings = base_model_output.last_hidden_state[0]

    # Additional processing for display purposes
    transformer_embeddings = transformer_embeddings.tolist()
    transformer_embeddings = [[round(i,2) for i in te] \
                              for te in transformer_embeddings]

    return transformer_embeddings


# Instantiate DistilBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModel.from_pretrained('distilbert-base-uncased')

corpora = gensim.downloader.info()['corpora'].keys()
for corpus in corpora:
    print(corpus)


# Extract the learned embedding for bank from DistilBERT
le_bank = extract_le('bank', tokenizer, model)[1]
type(le_bank)
len(le_bank)

# Write sentences containing "bank" in two different contexts
s1 = 'Write a poem about a man fishing on a river bank.'
s2 = 'Write a poem about a man withdrawing money from a bank.'

le_bank[:5]



token_dict = tokenizer(s1, return_tensors='pt')
type(token_dict)
type(token_dict['input_ids'])
token_dict['input_ids'].size()
token_ids = token_dict['input_ids'][0]

token_dict['attention_mask'].size()


tokens = tokenizer.convert_ids_to_tokens(token_ids)
tokens
# 11 to nr banku w liście tokenów
tokens[11]

# Extract the transformer embedding for bank from DistilBERT in each sentence
extract_te(s1, tokenizer, model)[11]
extract_te(s2, tokenizer, model)[11]

len(extract_te(s1, tokenizer, model))
len(extract_te(s1, tokenizer, model)[11])

s3 = 'Ala ma kota, a kot ma Alę i mieszkają w małym domu na Pomorzu Zachodnim'
s3 = 'The first step to produce transformer embeddings is to choose a model from the Hugging Face transformers library.'
s3 = 'dom Ala mama tata zupa rzeka lody river home running biegać sentences'
tdict3 = tokenizer(s3, return_tensors='pt')
tokenizer.convert_ids_to_tokens(tdict3['input_ids'][0])




