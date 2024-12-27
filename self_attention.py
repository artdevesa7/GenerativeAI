import torch
import torch.nn as nn
import torch.nn.functional as F

def self_attention(query, key, value, mask=None):
    """
    Compute scaled dot-product attention.

    Args:
        query: Tensor of shape (batch_size, seq_len, d_k)
        key: Tensor of shape (batch_size, seq_len, d_k)
        value: Tensor of shape (batch_size, seq_len, d_v)
        mask: Optional tensor of shape (batch_size, seq_len, seq_len) where 1 indicates positions to mask.

    Returns:
        output: Tensor of shape (batch_size, seq_len, d_v)
        attention_weights: Tensor of shape (batch_size, seq_len, seq_len)
    """
    d_k = query.size(-1)  # Dimension of query/key

    # Compute the dot products between query and key (scaled by sqrt(d_k))
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 1, float('-inf'))

    # Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # Compute the output as the weighted sum of values
    output = torch.matmul(attention_weights, value)

    return output, attention_weights

# Example usage
if __name__ == "__main__":
    batch_size = 2
    seq_len = 4
    d_k = 8
    d_v = 8

    # Random tensors for query, key, value
    query = torch.rand(batch_size, seq_len, d_k)
    key = torch.rand(batch_size, seq_len, d_k)
    value = torch.rand(batch_size, seq_len, d_v)

    # Optional mask (e.g., to prevent attention to padding tokens)
    mask = torch.zeros(batch_size, seq_len, seq_len)

    output, attention_weights = self_attention(query, key, value, mask)

    print("Output:", output)
    print("Attention Weights:", attention_weights)
