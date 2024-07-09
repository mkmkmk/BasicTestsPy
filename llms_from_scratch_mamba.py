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
import gc
from transformers import AutoModelForCausalLM , AutoTokenizer

gc.collect()

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:50"

    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    free_mem, total_mem = torch.cuda.mem_get_info()
    free_percentage = 100 * free_mem // total_mem
    free_mem = free_mem / 1024**2
    total_mem = total_mem / 1024**2
    print('\n')
    print(f'allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB, free_mem: {free_mem:.2f} MB, total_mem: {total_mem:.2f} MB, free_percentage: {free_percentage}% ')
    print('\n')

else:
    device = torch.device('cpu')
print(f"device: {device}")

model = AutoModelForCausalLM.from_pretrained('Q-bert/Mamba-1B', trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained('Q-bert/Mamba-1B')

text = "Hi"
text = "Hi, what do jou know about Wikipedia?"

input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

tokenizer.convert_ids_to_tokens(input_ids[0])

# oryginalnie 20, 200 - b dłuuugo
try:
    output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2)

except Exception as e:
    print(e)
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    free_mem, total_mem = torch.cuda.mem_get_info()
    free_percentage = 100 * free_mem // total_mem
    free_mem = free_mem / 1024**2
    total_mem = total_mem / 1024**2
    print('\n')
    print(f'allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB, free_mem: {free_mem:.2f} MB, total_mem: {total_mem:.2f} MB, free_percentage: {free_percentage}% ')
    print('\n')
    exit()

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)

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
