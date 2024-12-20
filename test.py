from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B"
model_name = 'gpt2'

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)

for name, _ in model.named_parameters():
    print(name)

sentence = "I am a student"

input_ids = tokenizer.encode(sentence, return_tensors='pt')

import pdb; pdb.set_trace()