"""2.5–2.8: Byte-pair encoding tokenizer through to input embeddings (the pipeline an LLM actually consumes)."""

import os

import requests
import tiktoken
import torch

from gpt_dataset import create_dataloader_v1


# 2.2 Load the raw text ("The Verdict" by Edith Wharton, a public-domain story).
if not os.path.exists("the-verdict.txt"):
    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open("the-verdict.txt", "wb") as f:
        f.write(response.content)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


# 2.5 Byte-pair encoding (BPE): GPT-2's tokenizer, splits unknown words into sub-words (no <|unk|> needed).
tokenizer = tiktoken.get_encoding("gpt2")

sample = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
integers = tokenizer.encode(sample, allowed_special={"<|endoftext|>"})
print("BPE encode:", integers)
print("BPE decode:", tokenizer.decode(integers))


# 2.6 Data sampling: batch of 8 sequences, 4 tokens each; stride == max_length so batches don't overlap.
max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False,
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("\nToken IDs:\n", inputs)
print("Inputs shape:", inputs.shape)  # [batch_size, max_length] = [8, 4]


# 2.7 Token embeddings: a lookup table of trainable vectors, one row per token ID.
vocab_size = 50257   # must match the BPE vocabulary
output_dim = 256     # embedding vector size

torch.manual_seed(123)
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

token_embeddings = token_embedding_layer(inputs)  # [8, 4] IDs -> [8, 4, 256] vectors
print("\nToken embeddings shape:", token_embeddings.shape)


# 2.8 Positional embeddings: a second table indexed by position (0, 1, 2, ...).
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

pos_embeddings = pos_embedding_layer(torch.arange(context_length))  # [4, 256]
print("Positional embeddings shape:", pos_embeddings.shape)

# Broadcast-add positions to each sequence -> final [8, 4, 256] input the LLM consumes.
input_embeddings = token_embeddings + pos_embeddings
print("Input embeddings shape:", input_embeddings.shape)
