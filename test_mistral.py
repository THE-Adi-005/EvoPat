import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Model device:", next(model.parameters()).device)

prompt = "Explain patent novelty in simple terms."

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

output = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.2
)

print(tokenizer.decode(output[0], skip_special_tokens=True))