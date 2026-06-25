import torch
import torch.nn as nn
import numpy as np

class HybridLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, hidden_dim):
        super(HybridLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                self.layers.append(LinearAttentionLayer(embed_dim, hidden_dim))
            else:
                self.layers.append(TransformerAttentionLayer(embed_dim, hidden_dim))
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        logits = self.output_layer(x)
        return logits

class LinearAttentionLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(LinearAttentionLayer, self).__init__()
        self.linear = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.output(self.activation(self.linear(x)))

class TransformerAttentionLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(TransformerAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x

def component_aware_self_speculative_decoding(model, input_ids, max_length, draft_length):
    device = input_ids.device
    generated = input_ids
    for _ in range(max_length):
        with torch.no_grad():
            # Generate draft tokens using the linear attention layers
            draft_logits = model.embedding(generated)
            for layer in model.layers:
                if isinstance(layer, LinearAttentionLayer):
                    draft_logits = layer(draft_logits)
                else:
                    break
            draft_probs = torch.softmax(draft_logits[:, -1, :], dim=-1)
            draft_tokens = torch.multinomial(draft_probs, num_samples=draft_length)

        # Verify the draft tokens using the full model
        for token in draft_tokens[0]:
            generated = torch.cat([generated, token.unsqueeze(0).unsqueeze(0)], dim=1)
            logits = model(generated)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            if torch.argmax(probs, dim=-1) == token:
                break  # Accept the token if verified
            else:
                generated = generated[:, :-1]  # Reject the token
    return generated

if __name__ == '__main__':
    # Dummy test
    vocab_size = 100
    embed_dim = 32
    num_layers = 4
    hidden_dim = 64
    max_length = 10
    draft_length = 2

    model = HybridLanguageModel(vocab_size, embed_dim, num_layers, hidden_dim)
    model.eval()

    input_ids = torch.tensor([[1]], dtype=torch.long)  # Start token
    output = component_aware_self_speculative_decoding(model, input_ids, max_length, draft_length)
    print("Generated sequence:", output)