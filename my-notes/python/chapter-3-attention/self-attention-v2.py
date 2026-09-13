import torch.nn as nn
import torch


class SelfAttention_v2(nn.Module):
    # Improves on v1 (see self-attention-v1.py) by swapping the raw
    # nn.Parameter weights for nn.Linear layers:
    #   - nn.Linear owns the (d_out, d_in) weight and applies x @ W.T, so
    #     forward just calls the layer instead of using the @ operator
    #   - it uses a smarter init (scaled by d_in) than torch.rand's flat
    #     [0, 1) values, so training is more stable
    #   - bias=False keeps it a pure matmul; qkv_bias lets us add one later

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, inputs_mat):
        keys = self.W_key(inputs_mat)
        queries = self.W_query(inputs_mat)
        values = self.W_value(inputs_mat)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec


torch.manual_seed(123)  # force tensor to use the same random values each time

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your     (x^1)
     [0.55, 0.87, 0.66],  # journey  (x^2)
     [0.57, 0.85, 0.64],  # starts   (x^3)
     [0.22, 0.58, 0.33],  # with     (x^4)
     [0.77, 0.25, 0.10],  # one      (x^5)
     [0.05, 0.80, 0.55]]  # step     (x^6)
)

d_in = inputs.shape[1]  # the input embedding size, d=3
d_out = 2  # the output embedding size, d=2

sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2.forward(inputs))
