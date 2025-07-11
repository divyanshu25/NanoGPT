import torch
from gpt2.gpt2_model import GPT, GPTConfig
from gpt2.dataloader import DataLoader
import time
from gpt2.gpt2_model import generate
import math

## Initialize variables ##

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


class Trainer:
    def __init__(self):
        self.config = GPTConfig()
        self.model = GPT(self.config)
        self.model.to(device)
        # Use "reduce-overhead" mode instead of "default" to avoid SM warnings on consumer hardware
        # self.model = torch.compile(self.model, mode="reduce-overhead")
        self.dataloader = DataLoader(
            data_file="/Users/divyanshugoyal/workspace/nanogpt/src/data/input.txt",
            batch_size=self.config.batch_size,
            block_size=self.config.block_size,
        )
        self.num_epochs = 10
        self.num_eval_samples = 100
        self.estimate_loss_after = 1
        self.max_learning_rate = 6e-4
        self.min_learning_rate = self.max_learning_rate * 0.1
        self.warmup_steps = 10
        self.max_steps = 50
        self.optimzer = self.model.configure_optimizers(
            learning_rate=self.max_learning_rate, weight_decay=0.10, device=device
        )

    ## Define function to estimate loss ##
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.num_eval_samples)
            for k in range(self.num_eval_samples):
                X, Y = self.dataloader.get_batch(split)
                X = X.to(device)
                Y = Y.to(device)
                _, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        return out

    def get_lr(self, step):
        if step < self.warmup_steps:
            lr = self.max_learning_rate * (step) / self.warmup_steps
        elif step > self.max_steps:
            lr = self.min_learning_rate
        else:
            decay_ratio = (step - self.warmup_steps) / (
                self.max_steps - self.warmup_steps
            )
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = self.min_learning_rate + coeff * (
                self.max_learning_rate - self.min_learning_rate
            )
        return lr

    def train(self):
        ## Start training ##
        torch.set_float32_matmul_precision("high")

        for epoch in range(self.num_epochs):
            start_time = time.time()
            for step in range(self.dataloader.total_batches):
                x, y = self.dataloader.get_batch()
                x, y = x.to(device), y.to(device)
                self.optimzer.zero_grad()
                logits, loss = self.model(x, y)
                loss.backward()
                # gradient clipping
                norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0
                )  # this is done to prevent the gradients from exploding and making the training unstable
                lr = self.get_lr(step)
                for param_group in self.optimzer.param_groups:
                    param_group["lr"] = lr

                print(f"Gradient norm: {norm: .4e} | Learning rate: {lr: .4e}")
                self.optimzer.step()

            if device == "cuda":
                torch.cuda.synchronize()
            tokens_per_second = (
                self.dataloader.batch_size
                * self.dataloader.block_size
                * self.dataloader.total_batches
                / (time.time() - start_time)
            )
            end_time = time.time()
            losses = self.estimate_loss()
            print(
                f"Epoch {epoch} | Loss: {losses['train']} | Val Loss: {losses['val']} | Tokens per second: {tokens_per_second} | Time taken: {end_time - start_time} seconds"
            )

        # save the model
        torch.save(trainer.model.state_dict(), "gpt2_trained_model.pth")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()

    trainer.model.load_state_dict(torch.load("gpt2_trained_model.pth"))
    trainer.model.to(device)

    generate(
        num_sequences=3,
        max_length=30,
        model=trainer.model,
        context="Hello, I'm a language model,",
        device=device,
    )
