import os
from datasets import load_dataset

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

dataset = load_dataset('Salesforce/wikitext', 'wikitext-103-v1')
print(dataset)
print(dataset['train'][0])