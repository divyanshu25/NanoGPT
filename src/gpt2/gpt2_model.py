from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from gpt2.block import Block
import tiktoken
import inspect


@dataclass
class GPTConfig:
    block_size: int = 512  # context length
    vocab_size: int = 50257  # number of tokens in the vocabulary
    n_layer: int = 2  # number of layers
    n_head: int = 2  # number of attention heads
    n_embed: int = 128  # embedding dimension
    batch_size: int = 4  # batch size


class GPT(nn.Module):
    """
    The GPT language model.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embed),
                wpe=nn.Embedding(config.block_size, config.n_embed),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embed),
            )
        )
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # initialize all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if hasattr(module, "NANOGPT_SCALE_INIT"):
            std *= (2 * self.config.n_layer) ** -0.5
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, learning_rate, weight_decay, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]

        optim_groups = [
            {"params": [p for p in decay_params], "weight_decay": weight_decay},
            {"params": [p for p in nodecay_params], "weight_decay": 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"Num decay parameter tensors: {len(decay_params)}, with {num_decay_params} parameters"
        )
        print(
            f"Num nodecay parameter tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters"
        )

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == "cuda"
        print(f"Using fused: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8
        )
        return optimizer

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # Shape (T,)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embed)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embed)
        x = tok_emb + pos_emb  # (B, T, n_embed)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Loads pretrained GPT-2 weights from Hugging Face.
        """
        # only GPT-2 models are fine-tuned.
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print(f"Loading weights from {model_type}")

        # n_layer, n_head and n_embed are determined by model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embed=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embed=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=24, n_embed=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=32, n_embed=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model
        config_args["block_size"] = 1024  # always 1024 for GPT model

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)

        # load the weights from the checkpoint
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in size
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        # basically the openai checkpoints use a "Conv1D" module, instead of a Linear module
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        # make sure that the model is completely initialized
        print(f"All parameters initialized and match in size")
        return model


def generate(num_sequences, max_length, model, context, device):
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(context)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_sequences, 1)
    x = tokens.to(device)

    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed(42)

    while x.size(1) < max_length:
        with torch.no_grad():
            logits, _ = model(x)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)
            # get Probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)

            # do top-k sampling of 50
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from topk probabilities
            ix = torch.multinomial(topk_probs, num_samples=1)  # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
            # append to the sequence
            x = torch.cat((x, xcol), dim=1)

    for i in range(num_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)


# test the model
if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # load the model
    model = GPT(GPTConfig())
    # print(model)
    model.eval()
    model.to(device)

    context = "Hello, I'm a language model,"
    generate(
        num_sequences=3, max_length=30, model=model, context=context, device=device
    )
