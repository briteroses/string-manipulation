import numpy as np
import pickle
import tiktoken
from openlogprobs import extract_logprobs, OpenAIModel

from pathlib import Path
dir_of_this = Path(__file__).parent.absolute()

# model = OpenAIModel("gpt-3.5-turbo")
# logprobs, calls = extract_logprobs(model, "Test", topk=False, multithread=True, workers=16)
# print("Total Calls", calls)

# with open('Test-gpt35.pkl', 'wb') as f:
#     pickle.dump(logprobs, f)

with open(str(dir_of_this / 'Test-gpt35.pkl'), 'rb') as f:
    logprobs = pickle.load(f)

# with open('hello.pkl', 'rb') as f:
#     logprobs = pickle.load(f)

encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')

k = 10
min_k_idx = np.argsort(logprobs)[:k]
# min_k_tokens = encoding.decode(min_k_idx)
print("Min k Tokens")
for i, idx in enumerate(min_k_idx):
    tok = encoding.decode([idx])
    print(tok)

max_k_idx = np.argsort(logprobs)[-k:]
# max_k_tokens = encoding.decode(max_k_idx)
print("Max k Tokens")
for i, tok in enumerate(max_k_idx):
    print(f'Max Token {i}')
    tok = encoding.decode([idx])
    print(tok)