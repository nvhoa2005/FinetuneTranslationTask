import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "nvh1101/Qwen2.5-1.5B-VLSP-Finetuned"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

def translate(message, history):
    system_msg = {"role": "system", "content": "You are a professional medical translator."}
    user_prompt = f"""### Task: Translate this sentence into Vietnamese accurately
### English: {message}
### Vietnamese:"""
    
    user_msg = {"role": "user", "content": user_prompt}
    messages = [system_msg, user_msg]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=256,    
            temperature=0.3,       
            repetition_penalty=1.1 
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response.strip()

demo = gr.ChatInterface(
    fn=translate,
    title="Demo Qwen 2.5 - 1.5B Fine-tune (VLSP)",
    description="Model được fine-tune cho bài toán dịch máy, hãy đưa câu mà bạn muốn dịch vào",
    examples=["Hello, I am a medical translation assistant."],
    cache_examples=False
)

if __name__ == "__main__":
    demo.launch()