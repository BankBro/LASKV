import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

text = "The largest city of China is"
encoded_input = tokenizer(text, return_tensors='pt')
forward_output = model(**encoded_input)

# print(forward_output)
# print(f'output logits shape: {forward_output['logits'].shape}')
# print(f'output len of past_key_values: {len(forward_output['past_key_values'])}')

generate_output = model.generate(**encoded_input)
generate_text = tokenizer.decode(generate_output[0])
print(generate_text)