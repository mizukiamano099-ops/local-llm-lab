from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu"
)

def infer(prompt):
    tokens = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **tokens,
        max_new_tokens=150,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    print(infer("こんにちは。あなたは誰？"))