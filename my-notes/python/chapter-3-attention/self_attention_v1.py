import torch.nn as nn
import torch


class SelfAttention_v1(nn.Module):
    # v1: weights held as raw nn.Parameter tensors, matmul done by hand.
    # See self_attention_v2.py for v2, which improves on this.

    def __init__(self, d_in, d_out):
        super().__init__()
        # create random weight matrices with the given dimensions
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, inputs_mat):
        # matmul the inputs matrix by the weight matrices
        keys = inputs_mat @ self.W_key
        queries = inputs_mat @ self.W_query
        values = inputs_mat @ self.W_value

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

sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1.forward(inputs))
