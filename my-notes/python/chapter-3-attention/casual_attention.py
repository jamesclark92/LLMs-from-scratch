import torch.nn as nn
import torch


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        # Step 2 (QKV v4 / matmul v4): learned projection weights W_q, W_k, W_v
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # Step 4.1 dropout (QKV v4 extension)
        self.dropout = nn.Dropout(dropout)
        # register buffer ensures buffers are automatically moved to the correct device (gpu/cpu)
        # this means we don't need to manually ensure these tensors are on the same device as
        # our model params
        self.register_buffer(
            'mask',
            # Step 3.1 causal mask (QKV v4): upper triangle above the diagonal
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        # Batching note (matmul v4): X is 3-D (batch, tokens, d_in)
        b, num_tokens, d_in = x.shape
        # Step 2: K = X·W_k, Q = X·W_q, V = X·W_v (matmul v4)
        # one shared weight matrix per projection (not per batch); nn.Linear acts on the
        # last dim, so it flattens (b, tokens) into rows, does one matmul, reshapes back
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        # Step 3 scores: Q·Kᵀ, scaled dot product (QKV v4)
        # transpose(1, 2) = Kᵀ per sequence: flip only the last two axes, keep batch axis 0
        # (plain .T reverses all axes and would scramble the batch dimension)
        attn_scores = queries @ keys.transpose(1, 2)
        # Step 3.1: fill masked (upper-triangle) scores with −∞ before softmax
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        # Step 4 weights: row-wise softmax(scores/√dₖ)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        # Step 4.1: dropout on attention weights (QKV v4)
        attn_weights = self.dropout(attn_weights)
        # Step 5: context vectors Z = A·V (matmul v4)
        context_vec = attn_weights @ values
        return context_vec


if __name__ == '__main__':
    torch.manual_seed(123)

    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your     (x^1)
         [0.55, 0.87, 0.66],  # journey  (x^2)
         [0.57, 0.85, 0.64],  # starts   (x^3)
         [0.22, 0.58, 0.33],  # with     (x^4)
         [0.77, 0.25, 0.10],  # one      (x^5)
         [0.05, 0.80, 0.55]]  # step     (x^6)
    )
    # stack two copies of the 6x3 sentence -> 3-D batch (2, 6, 3)
    batch = torch.stack((inputs, inputs), dim=0)

    d_in = inputs.shape[1]  # input embedding size, d=3
    d_out = 2               # output embedding size, d=2
    context_length = batch.shape[1]
    casual_attention = CausalAttention(d_in, d_out, context_length, 0.0)
    context_vecs = casual_attention(batch)
    print("context_vecs.shape: ", context_vecs.shape)
    print(context_vecs)