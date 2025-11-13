"""
    klon llms_from_scratch_mamba.py

    osobny venv

    $ python3 -m venv venvTransformers
    $ source source venvTransformers/bin/activate
    $ pip install scipy
    $ pip install torch
    $ pip install transformers

"""
# %%
import os
import torch
import gc
from transformers import AutoModelForCausalLM , AutoTokenizer
from transformers import TextStreamer
import transformers
from huggingface_hub import model_info

print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA: {torch.version.cuda}")

gc.collect()

def meminfo():
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    free_mem, total_mem = torch.cuda.mem_get_info()
    free_percentage = 100 * free_mem // total_mem
    free_mem = free_mem / 1024**2
    total_mem = total_mem / 1024**2
    print('\n')
    print(f'allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB, free_mem: {free_mem:.2f} MB, total_mem: {total_mem:.2f} MB, free_percentage: {free_percentage}% ')
    print('\n')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:50"
    meminfo()

else:
    device = torch.device('cpu')
print(f"device: {device}")

# GPT-2 small (124M)
# model = AutoModelForCausalLM.from_pretrained("gpt2")

# %%
# poszło to (2025.10.24), dzięki temu mam jakie warstwy są w tym modelu
# model = AutoModelForCausalLM.from_pretrained('Q-bert/Mamba-1B', trust_remote_code=True).to(device)
if False:
    model = AutoModelForCausalLM.from_pretrained(
        'Q-bert/Mamba-1B', 
        trust_remote_code=True,
        device_map="auto",
        # max_memory={"0": "6GB", "cpu": "30GB"}
        dtype=torch.float16
    )
    for name, module in model.named_modules():
        print(f"{name}: {type(module).__name__}")

# %%
device_map = {
    "model.embedding": "cuda:0",
    "lm_head": "cuda:0",
    "model.norm_f": "cuda:0"
}
for i in range(16):
    device_map[f"model.layers.{i}"] = "cuda:0"
for i in range(16, 48):
    device_map[f"model.layers.{i}"] = "cpu"

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
model_name = 'Q-bert/Mamba-1B'
model_name = 'gpt2'
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_info(model_name)


# local_dir = "/home/mkrej/dysk2T/NowyG/MojeDebianFractal/cache-huggingface/hub/DeepSeek-R1-Distill-Qwen-7B"
# git clone git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
local_dir = "/home/mkrej/dysk2T/NowyG/SourceCodeZNetu/DeepSeek/DeepSeek-R1-Distill-Qwen-7B"

if False:
    from huggingface_hub import snapshot_download
    import requests
    response = requests.get("https://huggingface.co", timeout=10)
    print(f"Status: {response.status_code}")
    print("✓ Połączenie z HF działa")

if False:
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )

model = AutoModelForCausalLM.from_pretrained(
    # model_name,
    local_dir,
    trust_remote_code=True,
    dtype=torch.float16,  # Oszczędność pamięci
    low_cpu_mem_usage=True,     # Optymalizacja pamięci CPU
    device_map="auto"           # Automatyczne zarządzanie urządzeniami
)

# ~10 minut:
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=True,
    # device_map=device_map,
    device_map = "cuda:0"
    # dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

for name, param in model.named_parameters():
    print(f"{name}: {param.device}")

meminfo()

# model = AutoModelForCausalLM.from_pretrained(
#     'Q-bert/Mamba-1B', 
#     trust_remote_code=True,
#     device_map="auto"
# )

# %%

# %%
text = "Hi"
text = "Hi, what do jou know about Wikipedia?"

input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

if False:
    tokenizer.convert_ids_to_tokens(input_ids[0])

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# oryginalnie 20, 200 - b dłuuugo
try:
    output = model.generate(input_ids, max_length=100) #, num_beams=5, no_repeat_ngram_size=2)

except Exception as e:
    print(e)
    meminfo()
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

# %%
# # strumieniowanie - pełna kontrola
gc.collect()
torch.cuda.empty_cache()
meminfo()
    
text = "Hi"
text = "System: You are a friendly assistant\nUser: Hi, what do jou know about Wikipedia?\nAssistant:"
input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
generated = input_ids.clone()

for _ in range(500):
    with torch.no_grad():
        outputs = model(generated)
        logits = outputs[0]  # pierwszy element tuple to logits
        next_token = torch.argmax(logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
        generated = torch.cat([generated, next_token], dim=1)
        
        new_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
        print(new_text, end='', flush=True)
        
        if next_token.item() == tokenizer.eos_token_id:
            break

if False:
    tokenizer.decode(generated[0], skip_special_tokens=True)

# %%
outputs = model(input_ids)
print(type(outputs))
print(len(outputs) if hasattr(outputs, '__len__') else 'no len')

generated_text = tokenizer.decode(outputs, skip_special_tokens=True)

# %%
# Streaming z TextStreamer

text = "Hi"
text = "Hi, what do jou know about Wikipedia?"
# input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
inputs = tokenizer(text, return_tensors="pt", padding=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=False)

# Generate ze streamerem
output = model.generate(
    inputs.input_ids,
    max_length=100, 
    streamer=streamer,  # to wydrukuje tokeny na żywo
    do_sample=True,
    attention_mask=inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id
)
# %%

system_prompt = """
Jesteś pomocnym asystentem AI.
Bloki kodu oznaczaj znacznikiem 3 x tylda.
Nie stosuj 'inner speach', 'thinking' i znaczników '\<think\>'
"""
conversation = f"System: {system_prompt}"
