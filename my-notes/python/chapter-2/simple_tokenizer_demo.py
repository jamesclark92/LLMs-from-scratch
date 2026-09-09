"""2.2–2.4: build a vocabulary from raw text and demo the from-scratch SimpleTokenizerV2."""

import os
import re
import requests

class SimpleTokenizerV2:
    """encode: text -> IDs, decode: IDs -> text."""

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # Map unseen tokens to <|unk|> instead of crashing.
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>"
            for item in preprocessed
        ]
        return [self.str_to_int[s] for s in preprocessed]

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)  # drop space before punctuation
        return text

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

print("Total number of characters:", len(raw_text))
print(raw_text[:99])


# 2.2 Tokenizing: split on whitespace and punctuation, keeping punctuation as tokens.
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print("\nNumber of tokens:", len(preprocessed))
print("First 30 tokens:", preprocessed[:30])


# 2.3 / 2.4 Build a vocabulary (token -> unique ID) plus <|endoftext|> and <|unk|> special tokens.
all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}
print("\nVocabulary size:", len(vocab))


# Demo the simple tokenizer, including <|unk|> and <|endoftext|> handling.
tokenizer = SimpleTokenizerV2(vocab)
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print("\nSimpleTokenizerV2 encode:", tokenizer.encode(text))
print("SimpleTokenizerV2 decode:", tokenizer.decode(tokenizer.encode(text)))
