import os
import sys
import time
import shutil
import torch
import transformers
sys.path.append("..")
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
from peft import PeftModel
from model_mapping import MODEL_SPECIAL_TOKENS


model_path='path to base model'
lora_adapter='path to lora adaptor ckpt'
save_path='path to new merged model'
model_type = 'gpt_neox'

t0 = time.time()
config = {"model_type": model_type}
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    model_path,  
    trust_remote_code=True,
    torch_dtype=torch.bfloat16, 
    return_dict=True,
    device_map="auto"
)
print(base_model)

# merge, save model and tokenizer
model_to_merge = PeftModel.from_pretrained(base_model, lora_adapter)
merged_model = model_to_merge.merge_and_unload()
print(merged_model.config)
merged_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Merge finised: {save_path} saved, Cost {time.time()-t0:.2f}s")