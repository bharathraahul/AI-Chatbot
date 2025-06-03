
import torch
from transformers import AutoTokenizer, pipeline
import keyboard

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

generator = pipeline(task="text-generation",torch_dtype=torch.bfloat16,model=model_id,device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_id)


user_history =[]

def response_generator(user_input):
    chats = [
    {
        "role": "system",
        "content": "You are a friendly chatbot",
    }
    ]
    
    for i in user_history:
        chats.append({"role":"user","content":i["user"]})
        chats.append({"role":"system","content":i["assistant"]})
        print(chats)
    
    chats.append({"role":"user","content":user_input})
    
    prompt = tokenizer.apply_chat_template(chats, tokenize=False, add_generation_prompt=True)
    response = generator(prompt,max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    output = response[0]["generated_text"]

    # new_text = output[len(user_input):].strip()
    

    if "<|assistant|>" in output:
        new_text = output.split("<|assistant|>")[-1]
    else:
        new_text = output.strip()
        
    return new_text
        
def user_message(user_input):
    if user_input:
        output_text = response_generator(user_input)
        user_history.append({"user":user_input,"assistant":output_text})
        return output_text


while True:
    user_input = input("You:")
    
    if user_input:
        console_output= user_message(user_input)
        print("Bot:",console_output)
        
        # if keyboard.is_pressed('esc'):
        #     print("Exiting chat")
        #     break
        
    if user_input.lower()=="exit":
        print("Exiting")
        break
    
        
    
        

        
    





    

# response = generator(messages,max_new_tokens=512)

# prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# outputs = generator(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
# print(outputs[0]["generated_text"])


# outputs = pipe(
#     messages,
#     max_new_tokens=256,
# )
# print(response[0]["generated_text"][-1])

