import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import LLM_MODEL

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()


def generate_response(prompt, max_new_tokens=500):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print("Prompt tokens:", inputs["input_ids"].shape[1])
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

