from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

app = FastAPI()

def generate_text(prompt: str, max_length: int = 50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.get("/generate")
def get_generation(prompt: str, max_length: int = 50):
    return {"prompt": prompt, "generated_text": generate_text(prompt, max_length)}