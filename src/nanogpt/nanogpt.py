import torch
from .dataset import ShakespeareDataset
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set the random seed for reproducibility
torch.manual_seed(1337)
# Select device: use GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"


class Head(nn.Module):
    """
    A single head of the NanoGPT model.
    """

    def __init__(self, head_size, n_embd, block_size, device, dropout):
        super().__init__()
        self.n_embd = n_embd
        self.head_size = head_size
        self.block_size = block_size
        self.device = device

        self.key = nn.Linear(self.n_embd, self.head_size, bias=False)
        self.query = nn.Linear(self.n_embd, self.head_size, bias=False)
        self.value = nn.Linear(self.n_embd, self.head_size, bias=False)
        self.register_buffer(
            "lower_triangular_mask",
            torch.tril(torch.ones(self.block_size, self.block_size)),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # compute attention scores
        wei = (
            q @ k.transpose(-2, -1) * self.head_size**-0.5
        )  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        mask = self.lower_triangular_mask[:T, :T].to(self.device)  # type: ignore
        wei = wei.masked_fill(mask == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # compute output
        out = wei @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """
    A multi-head attention mechanism.
    """

    def __init__(self, n_embd, head_size, block_size, n_heads, device, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Head(head_size, n_embd, block_size, device, dropout)
                for _ in range(n_heads)
            ]
        )
        self.proj = nn.Linear(head_size * n_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """
    A feed-forward network.
    """

    def __init__(self, n_embd, device, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    A block of the NanoGPT model.
    """

    def __init__(self, n_embd, n_heads, block_size, device, dropout):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(
            n_embd, head_size, block_size, n_heads, device, dropout
        )
        self.ff = FeedForward(n_embd, device, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = self.sa(self.ln1(x)) + x
        x = self.ff(self.ln2(x)) + x
        return x


class NanoGPT(nn.Module):
    """
    A minimal GPT-like neural network for language modeling.

    Args:
        dataset (ShakespeareDataset): The dataset object containing vocabulary and data processing methods.
    """

    def __init__(self, dataset, config):
        """
        Initializes the NanoGPT model, downloads and processes the dataset, and sets up the token embedding table.

        Args:
            dataset (ShakespeareDataset): The dataset object to use for training and evaluation.
        """
        super().__init__()
        self.dataset = dataset
        self.dataset.download_data()  # Download the dataset if not already present
        self.dataset.process_data()  # Process the dataset (tokenization, etc.)
        # Embedding table: maps token indices to embedding vectors

        self.n_embd = config[
            "n_embd"
        ]  # embedding dimension (number of features for each token)
        self.block_size = config["block_size"]  # context window size
        self.device = config["device"]
        self.n_heads = config["n_heads"]
        self.n_blocks = config["n_blocks"]
        self.dropout = config["dropout"]
        self.head_size = self.n_embd // self.n_heads
        self.token_embedding_table = nn.Embedding(
            self.dataset.vocab_size, self.n_embd
        )  # (B, T, C)

        self.pos_embedding = nn.Embedding(self.block_size, self.n_embd)  # (T, C)
        self.lm_head = nn.Linear(
            self.n_embd, self.dataset.vocab_size
        )  # (B, T, vocab_size)
        self.ma_head = MultiHeadAttention(
            self.n_embd,
            self.head_size // self.n_heads,
            self.block_size,
            self.n_heads,
            self.device,
            self.dropout,
        )
        self.ff = FeedForward(self.n_embd, self.device, self.dropout)

        self.blocks = nn.Sequential(
            *[
                Block(
                    self.n_embd,
                    self.n_heads,
                    self.block_size,
                    self.device,
                    self.dropout,
                )
                for _ in range(self.n_blocks)
            ],
            nn.LayerNorm(self.n_embd),
        )

    def forward(self, idx, targets=None):
        """
        Forward pass of the NanoGPT model.

        Args:
            idx (torch.Tensor): Input tensor of token indices (batch_size, sequence_length).
            targets (torch.Tensor or None): Target tensor of token indices for loss computation (same shape as idx), or None for inference.

        Returns:
            logits (torch.Tensor): The raw, unnormalized scores for each token in the vocabulary (batch_size * sequence_length, vocab_size).
            loss (torch.Tensor or None): Cross-entropy loss if targets are provided, else None.
        """

        B, T = idx.shape
        token_embeddings = self.token_embedding_table(
            idx
        )  # (B, T, C): batch, time, embedding dimension
        pos_embeddings = self.pos_embedding(
            torch.arange(T, device=self.device)
        )  # (T, C)

        x = token_embeddings + pos_embeddings  # (B, T, C)
        x = self.blocks(x)  # apply self attention head to the token embeddings
        logits = self.lm_head(x)  # (B, T, vocab_size)

        B, T, C = logits.shape  # batch size, sequence length, embedding dimension
        if targets is None:
            loss = None
        else:
            # Reshape logits and targets for cross-entropy loss
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_sampled = idx[:, -self.block_size :]
            logits, loss = self(idx_sampled, None)
            logits = logits[
                :, -1, :
            ]  # Consider only the last time step for prediction, (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    """
    Example usage: Initializes the dataset and model, gets a training batch, and runs a forward pass.
    """
    config = {"n_embd": 64, "block_size": 32, "head_size": 32, "device": device}
    dataset = ShakespeareDataset()
    model = NanoGPT(dataset, config)
    train_data, val_data = dataset.train_test_split()
    train_batch_x, train_batch_y = dataset.get_batch("train", 8, 32)
    logits, loss = model(train_batch_x, train_batch_y)
    print(logits.shape, loss)
    print(logits)
    print(loss)
    print(
        dataset.decode(
            model.generate(
                idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100
            ).tolist()[0]
        )
    )
