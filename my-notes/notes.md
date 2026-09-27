# Build an LLM from scratch — notes

The whole book is one pipeline. Each chapter builds one box, and every later chapter reuses the earlier ones:

<div class="pipeline">
<span class="ch1">Text</span>→<span class="ch2">Tokens</span>→<span class="ch2">Token IDs</span>→<span class="ch2">Embeddings + positions</span>→<span class="ch3">Attention</span>→<span class="ch4">Transformer blocks</span>→<span class="ch4">Logits → softmax</span>→<span class="ch1">Next token</span>
</div>

Colour key: <span class="tag ch1">Ch 1 · the big picture</span> <span class="tag ch2">Ch 2 · text → vectors</span> <span class="tag ch3">Ch 3 · attention</span> <span class="tag ch4">Ch 4 · GPT model (next)</span>

---

# Chapter 1 · Understanding LLMs

## Key terms

- **LLM (large language model)** — a deep neural network trained on huge amounts of text to predict the **next token**. "Large" = both the parameter count (billions) and the training data (hundreds of billions of tokens).
- **Parameters / weights** — the adjustable numbers inside the network, learned in training. Same thing as the weights in your MNIST net, just ~10⁸–10¹¹ of them instead of ~10⁵.
- **Transformer** — the neural-network architecture from *"Attention Is All You Need"* (2017), originally built for translation. Its key idea is **self-attention**: each token can look at every other token to decide what matters. Nearly all modern LLMs are transformers.
- **Encoder / decoder** — the two halves of the original transformer. The encoder *reads* the whole input into context-aware vectors; the decoder *writes* output one token at a time.
- **GPT (Generative Pretrained Transformer)** — a transformer that keeps **only the decoder**. *Generative* = it produces text; *pretrained* = trained on raw text first, specialised later.
- **BERT** — the opposite choice: **encoder only**, trained to fill in masked words. Good at understanding/classifying, not generating.
- **Autoregressive** — each output token is appended to the input and fed back in to produce the next one. That loop is how GPT writes.
- **Pretraining** — stage 1 of training: next-token prediction on raw, unlabelled text. Produces a **foundation (base) model**.
- **Self-supervised learning** — the labels come from the data itself: the "label" for each position is simply the next token in the text. No humans labelling, unlike MNIST where every image needed a digit label.
- **Fine-tuning** — stage 2: further training on a smaller *labelled* dataset. *Classification* fine-tuning (e.g. spam / not spam) or *instruction* fine-tuning (question → answer pairs, which is what makes a chatbot).
- **Zero-shot / few-shot** — doing a task with no examples / with a few examples given in the prompt, without retraining.
- **Emergent behaviour** — abilities nobody trained for directly (e.g. translation) that appear from next-token prediction at scale.
- **Token** — the unit of text the model reads: a word, sub-word, or punctuation mark (chapter 2).

## Where LLMs sit

AI ⊃ machine learning ⊃ deep learning ⊃ LLMs. Deep learning = neural nets with many layers that learn their own features (your MNIST net didn't need hand-written "has a loop" features — it learned them from pixels). An LLM is that idea applied to text.

## The original transformer (Fig 1.4)

<figure class="dg"><img src="transformer_diagram_v3.svg" alt="The Transformer — Encoder / Decoder"></figure>

- **Encoder (left)**: reads the English source, *"The cat sat on the wall."* — **all tokens at once**, making each embedding context-aware. The only loop is over stacked layers:

  ```python
  x = embed(tokens) + positional_info
  for layer in encoder_layers:  # e.g. 6–12 blocks
      x = layer(x)              # self-attention + feed-forward
  return x                      # context-rich embeddings
  ```

- **Decoder (right)**: reads and writes German only. Its input is the German produced so far (*"Die Katze saß"*), preprocessed exactly like the encoder's input.
- **Cross-attention (the 4 → 7 arrow)**: the only place the two languages meet. The decoder forms *queries*; the encoder's output supplies *keys* and *values* (ch 3 terms). That link is the translation.
- **Output layers (8)**: a final linear layer + **softmax** → a probability for every token in the vocabulary. Exactly your MNIST output layer, with ~50,000 classes instead of 10.
- **Loop (5 → 9)**: append the chosen token, run again, until `<|endoftext|>`.

## From transformer to GPT

GPT = **the right-hand column on its own**: preprocessing → decoder → softmax, no encoder, no cross-attention arrow. "Partial output" simply becomes "the input text". Trained only on next-token prediction, yet it can still translate — an emergent behaviour.

| | Original transformer | BERT | GPT |
|---|---|---|---|
| Uses | encoder + decoder | encoder | decoder |
| Trained to | translate | fill in masked words | predict next word |
| Sees | whole input | both directions | left context only (causal) |
| Good at | translation | classification, search | generating text |

## The book's plan (Fig 1.9)

1. **Stage 1 — build it**: data prep & sampling (ch 2) → attention (ch 3) → LLM architecture (ch 4)
2. **Stage 2 — pretrain it**: training loop, evaluation, loading pretrained weights (ch 5)
3. **Stage 3 — fine-tune it**: classifier (ch 6), personal assistant (ch 7)

---

# Chapter 2 · Working with text data

Chapter 2 builds everything to the left of the attention box: **text → tokens → token IDs → input embeddings**, plus the machinery that chops text into training examples.

<figure class="dg"><img src="python/chapter-2-preprocessing/input_prep_pipeline.svg" alt="From text to input embeddings: text, tokens, IDs, token + positional embeddings"></figure>

## Key terms

- **Embedding** — a mapping from something discrete (a word, image, document) to a vector of real numbers, so a neural net can do maths on it. Text is categorical; networks need numbers.
- **Word2Vec** — an early (pre-transformer) way to learn word embeddings: words in similar contexts get similar vectors. LLMs instead learn their embeddings as part of training.
- **Tokenizer** — splits text into tokens and maps them to integer IDs (`encode`), and back (`decode`).
- **Vocabulary** — the `token → ID` dictionary. GPT-2's has 50,257 entries.
- **Special tokens** — `<|unk|>` (unknown word — only needed by simple word-level tokenizers) and `<|endoftext|>` (separates unrelated documents, and marks the end of generation).
- **BPE (byte pair encoding)** — GPT's tokenizer. Builds a vocabulary of sub-word chunks by repeatedly merging frequent pairs, so any word — even an unseen one — breaks into known pieces. No `<|unk|>` needed.
- **Context length (`max_length`)** — how many tokens the model sees at once. GPT-2: 1024.
- **Token embedding layer** — a learnable lookup table, shape `(vocab_size × emb_dim)`. Row *i* = the vector for token ID *i*.
- **Positional embedding** — a second vector per *position* (0, 1, 2, …), added to the token embedding so the model knows word order.
- **Input embedding** — token embedding + positional embedding. This is what enters the transformer.

## a. Tokenize → b. Token IDs

Code: `python/chapter-2-preprocessing/simple_tokenizer_demo.py` (word-level, from scratch) and `bpe_pipeline.py` (BPE via `tiktoken`).

```python
tokenizer = tiktoken.get_encoding("gpt2")
ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})  # subword IDs, no <|unk|>
```

The book's hand-built `SimpleTokenizerV2` is **word-level**: anything not in its dictionary collapses to a single `<|unk|>`, throwing information away. BPE exists to avoid exactly that.

## Sliding window: turning IDs into training examples

Code: `python/chapter-2-preprocessing/gpt_dataset.py`. Next-token prediction means the **target is the input shifted by one**:

```
input : [ "In", "the", "heart", "of" ]
target: [ "the", "heart", "of", "the" ]
```

One window of 4 tokens gives 4 training predictions at once (predict "the" from "In"; "heart" from "In the"; …).

- **`max_length`** = width of each window (the context length).
- **`stride`** = how far the window jumps to start the next example.
  - `stride == max_length` → windows tile, every token used once (the usual default).
  - `stride < max_length` → overlap: more examples, more overfitting risk.
  - `stride > max_length` → skips tokens (rarely wanted).

Two senses of "seeing" a token: *within one window*, a token only attends to earlier tokens **in that window**; *across the dataset*, every token still contributes because the windows tile the text.

## c. Token IDs → embeddings

The embedding layer is a learnable lookup table. Its numbers start random and are **learned in training** — exactly like your MNIST weights — so tokens used in similar contexts drift to similar vectors.

**A lookup *is* a matrix multiply.** Write the ID as a one-hot row vector; multiplying it by the embedding matrix `W` selects one row. Stack a sentence's one-hot rows and a single matmul gives every token's embedding. That's the MNIST bridge: your net multiplied a pixel vector by a weight matrix; here the one-hot vector *is* the input, so an embedding layer is "just a linear layer" (PyTorch skips building the one-hots because a lookup is far cheaper).

<figure class="dg narrow"><img src="one_hot_lookup.svg" alt="Stacked one-hot rows times embedding matrix"></figure>

## Positional embeddings

Self-attention (ch 3) treats its input as a **set** — on its own *"dog bites man"* and *"man bites dog"* look the same. Fix: add a per-position vector.

```
input embedding[i] = token embedding[token_id[i]] + positional embedding[i]
```

| Position | Token emb. | Positional emb. | Input emb. (sum) |
|---|---|---|---|
| 0 | `1, 1, 1` | `1.1, 1.2, 1.3` | `2.1, 2.2, 2.3` |
| 1 | `1, 1, 1` | `2.1, 2.2, 2.3` | `3.1, 3.2, 3.3` |
| 2 | `1, 1, 1` | `3.1, 3.2, 3.3` | `4.1, 4.2, 4.3` |

Flavours: *sinusoidal* (original 2017 transformer, fixed), *learned* (GPT-2 and this book — one learnable row per position, so the table size caps context length), *RoPE* (Llama and most modern LLMs — rotates Q and K instead of adding anything).

**Shape at the end of chapter 2:** `(batch, context_length, emb_dim)` — e.g. `(8, 4, 256)` in the book. This tensor is `X`, the input to chapter 3.

---

# Chapter 3 · Attention

Chapter 3 builds the attention box. The diagrams below are **one story told five times**, each adding one idea to the last. They all use the same sentence, *"Your journey starts with one step"*, the same six 3-dim embeddings, and follow token 2, **"journey"**, so you can track the same numbers across them.

## Key terms

- **Attention** — a way for each token to build a new vector from a *weighted mix* of all the tokens' vectors, where the weights say how relevant each other token is.
- **Self-attention** — attention where the tokens attend to others *in the same sequence* (as opposed to cross-attention between two sequences).
- **Attention score (ω)** — raw relevance of token *j* to token *i*: a dot product.
- **Attention weight (α)** — the scores after softmax: positive, summing to 1.
- **Context vector (z)** — the output for one token: the attention-weighted sum. It's the token's embedding *enriched with information from its context* — "bank" beside "river" ends up different from "bank" beside "money".
- **Query / key / value (Q, K, V)** — three learned projections of each token. Query = "what am I looking for?"; key = "what do I contain?"; value = "what do I hand over if chosen?". Score = query · key.
- **Scaled dot-product attention** — scores divided by √dₖ before softmax so they don't get too large (which would make softmax almost one-hot and gradients tiny).
- **Causal (masked) attention** — each token may only attend to itself and **earlier** tokens. Required for next-token prediction, otherwise the model could peek at the answer.
- **Dropout** — randomly zeroing some attention weights during training to reduce overfitting.
- **Multi-head attention** — several attention computations ("heads") in parallel, each with its own W_q/W_k/W_v, outputs concatenated. Each head can learn to track a different kind of relationship.

## Why attention? (§3.1–3.2)

Before transformers, translation used **RNNs**: the encoder read words one at a time and squeezed the whole sentence into a single hidden state that the decoder had to work from. Long sentences lost information in that bottleneck. Attention lets the output look back at **every** input token directly, with learned weights. Self-attention then applies the same trick within one sequence.

## Step 1 · Simple self-attention, no weights (§3.3)

Code: the inline examples in the book (no class yet).

<figure class="dg"><img src="python/chapter-3-attention/Self%20attention%20simple%20v2.svg" alt="Simple self-attention for x2"></figure>

Three steps, repeated for every token: **score** (dot product of the query with every input) → **weight** (softmax) → **context vector** (weighted sum of the inputs). Nothing here is learned — the embeddings are compared directly with themselves.

> **What's missing → next diagram:** the model can't *learn* what to attend to. The same vector is used as the question, the thing being compared, and the thing being mixed. Next step: give each of those three roles its own learned weight matrix.

## Step 2 · Trainable weights: Q, K, V (§3.4.1)

Code: `python/chapter-3-attention/self_attention_v1.py`, `self_attention_v2.py`.

<figure class="dg"><img src="python/chapter-3-attention/Self%20attention%20QKV%20v4.svg" alt="Self-attention with trainable Q, K, V"></figure>

**What changed from step 1:**

| Step 1 (simple) | Step 2 (QKV) |
|---|---|
| query = x⁽²⁾ | q⁽²⁾ = x⁽²⁾ · W_q |
| compared against x⁽ⁱ⁾ | compared against k⁽ⁱ⁾ = x⁽ⁱ⁾ · W_k |
| score = x⁽²⁾ · x⁽ⁱ⁾ | score = q⁽²⁾ · k⁽ⁱ⁾ **/ √dₖ** |
| z = Σ α · x⁽ⁱ⁾ | z = Σ α · v⁽ⁱ⁾ = x⁽ⁱ⁾ · W_v |
| nothing learned | W_q, W_k, W_v learned |

The three steps (score → softmax → weighted sum) are **identical**; only the vectors going in have changed. W_q, W_k, W_v are the same kind of weight matrix as a layer in your MNIST net, and there's one set shared by all tokens.

> **What's missing → next diagram:** this was done one token at a time, in a loop. Next: the same maths for all six tokens at once, as matrix multiplies.

## Step 3 · The same thing as matrix multiplies (§3.4.2)

Code: `self_attention_v2.py` — the `forward` is exactly this diagram.

<figure class="dg"><img src="python/chapter-3-attention/Self%20attention%20QKV%20matmul%20v4.svg" alt="Self-attention as matrix multiplies"></figure>

Every token's row goes through together: `Q = X·W_q`, `K = X·W_k`, `V = X·W_v`, then `softmax(Q·Kᵀ / √dₖ) · V`. **Row 2 of the result is exactly z⁽²⁾ from step 2** — same numbers, no loop. It also starts from token IDs (one-hot · E), linking back to the chapter 2 embedding diagram.

```python
attn_scores  = queries @ keys.T                          # (6×6): every token vs every token
attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)
context_vecs = attn_weights @ values                     # (6×d_out)
```

> **What's missing → next step:** in the 6×6 score matrix, "journey" (row 2) gets weight from "starts", "with", "one", "step" — tokens that come **after** it. When predicting the next word that's cheating.

## Step 4 · Causal attention: hide the future (§3.5)

Code: `python/chapter-3-attention/casual_attention.py`. *(No diagram yet — a candidate for the next SVG: the 6×6 weight matrix from step 3 with its upper triangle masked.)*

Two additions to step 3, both applied to the 6×6 score matrix:

1. **Mask** — set every score above the diagonal to `-inf` *before* softmax. `e^-inf = 0`, so future tokens get weight 0 and each row still sums to 1. Row 2 ("journey") now only mixes "Your" and "journey".

   ```python
   mask = torch.triu(torch.ones(6, 6), diagonal=1)          # 1s above the diagonal
   attn_scores.masked_fill_(mask.bool(), -torch.inf)
   ```

2. **Dropout** — during training only, randomly zero some attention weights (and scale the rest up) to reduce overfitting.

`CausalAttention` also adds a **batch** dimension: input `(batch, tokens, d_in)`, so `keys.transpose(1, 2)` replaces `keys.T`.

> **What's missing → next diagram:** one set of W_q/W_k/W_v can learn one notion of "relevant". Next: run several in parallel.

## Step 5 · Multi-head: stack heads (§3.6.1)

Code: `python/chapter-3-attention/mutli_head_attention_wrapper.py`.

<figure class="dg"><img src="python/chapter-3-attention/Multi-head%20attention%20wrapper%20v2.svg" alt="Multi-head attention wrapper"></figure>

The `MultiHeadAttentionWrapper` is literally a list of step-4 `CausalAttention` modules, each with **its own** W_q/W_k/W_v, all given the **same** X. Their outputs are glued along the last axis with `torch.cat(dim=-1)`: 2 heads × 2 dims → 4 features per token.

> **What's wasteful → next diagram:** N separate modules means N separate small matmuls run one after another.

## Step 6 · Multi-head, combined (§3.6.2)

Code: `python/chapter-3-attention/multi_head_attention.py` (the book's `MultiHeadAttention`, Listing 3.5) and `pytorch_multi_head_attention.py`.

<figure class="dg"><img src="python/chapter-3-attention/Multi-head%20attention%20combined%20v1.svg" alt="Combined multi-head attention"></figure>

**Same maths as step 5, reorganised:** one *wide* W_q (3×4) instead of two narrow ones (3×2) — the wide matrix is just the two narrow ones side by side. Then `.view` splits the 4 columns into 2 heads × 2, `.transpose` puts heads next to batch, and steps 3–4 run for all heads in one batched matmul. Finally heads are merged back and passed through `out_proj`, a linear layer that mixes the heads' outputs.

Shapes through `MultiHeadAttention.forward`:

```
x            (b, tokens, d_in)
Q, K, V      (b, tokens, d_out)                 one wide projection each
.view        (b, tokens, heads, head_dim)       split columns into heads
.transpose   (b, heads, tokens, head_dim)       heads act like extra batch
scores       (b, heads, tokens, tokens)         causal mask + softmax + dropout
context      (b, tokens, d_out)                 transpose back, merge heads
out_proj     (b, tokens, d_out)
```

Interactive 3D view of this listing:

<iframe class="interactive" src="python/chapter-3-attention/Multi-head%20attention%20listing%203%205%203D%20v7.html" loading="lazy"></iframe>

[Open the 3D view full-screen ↗](python/chapter-3-attention/Multi-head%20attention%20listing%203%205%203D%20v7.html)

## Chapter 3 in one line

`X (b, tokens, d_in)` → **MultiHeadAttention** → `(b, tokens, d_out)`: same shape in and out, but every token's vector now carries information from the tokens before it. GPT-2 small uses 12 heads and d_out = 768. Chapter 4 wraps this in a **transformer block** (plus layer norm, a feed-forward network and shortcut connections) and stacks 12 of them.

---

# Parking lot

- Next SVG to draw: causal mask (step 4) — the step-3 6×6 weight matrix with the upper triangle greyed out and rows renormalised.
