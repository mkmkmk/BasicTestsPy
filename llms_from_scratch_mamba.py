"""
    https://huggingface.co/Q-bert/Mamba-1B


    osobny venv ze starymi bibliotekami

    $ python3 -m venv venvTransformers
    $ source source venvTransformers/bin/activate
    $ pip install scipy==1.10.1
    $ pip install torch
    $ pip install transformers

"""

import os
import torch
from transformers import AutoModelForCausalLM , AutoTokenizer

if False:
    os.environ['HTTPS_PROXY'] = 'http://192.168.44.1:8080'

model = AutoModelForCausalLM.from_pretrained('Q-bert/Mamba-1B', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('Q-bert/Mamba-1B')


text = "Hi, what do jou know about Wikipedia?"

input_ids = tokenizer.encode(text, return_tensors="pt")
tokenizer.convert_ids_to_tokens(input_ids[0])

# oryginalnie 20, 200 - b dłuuugo
output = model.generate(input_ids, max_length=200, num_beams=5, no_repeat_ngram_size=2)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
generated_text

# 'Hi, what do jou know about Wikipedia?\n\n
# Wikipedia is a free online encyclopedia that anyone can edit. 
# It was founded in 2001 by Jimmy Wales and Larry Sanger. Wikipedia is written in the English language 
# and is available in many languages, including Chinese, French, German, Italian, Japanese, Korean, 
# Polish, Portuguese, Russian, Spanish, Swedish, Turkish, and Vietnamese. 
# There are currently more than 1.2 million articles on Wikipedia. The articles are written 
# by volunteer editors, who are not paid for their work. Anyone with an Internet connection 
# and a computer can contribute to Wikipedia by editing the articles, which are then reviewed 
# by other editors before being added to the main article. Articles can be edited by anyone, 
# but only registered users are allowed to make changes to articles that have been created 
# by a registered user. If you want to edit an article, you will need to register with the 
# site and create an account. You can then edit any article that you have access to.'
