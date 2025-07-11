import torch
from gpt2.gpt2_model import GPT, GPTConfig
from gpt2.dataloader import DataLoader
import time
from gpt2.gpt2_model import generate
import math

## Initialize variables ##

# Device selection: prioritize CUDA (NVIDIA GPUs), then MPS (Apple Silicon), fallback to CPU
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


class Trainer:
    """
    GPT-2 Trainer class that handles model training, evaluation, and optimization.
    Implements modern training techniques like learning rate scheduling and gradient clipping.
    """

    def __init__(self):
        """Initialize trainer with model configuration, data loading, and training parameters."""
        # Initialize GPT model with default configuration
        self.config = GPTConfig()
        self.model = GPT(self.config)
        self.model.to(device)

        # Optional: Compile model for faster training (commented out to avoid warnings)
        # Use "reduce-overhead" mode instead of "default" to avoid SM warnings on consumer hardware
        # self.model = torch.compile(self.model, mode="reduce-overhead")

        # Initialize data loader with training data
        self.dataloader = DataLoader(
            data_file="/Users/divyanshugoyal/workspace/nanogpt/src/data/input.txt",
            batch_size=self.config.batch_size,
            block_size=self.config.block_size,
        )

        # Training hyperparameters
        self.num_epochs = 1  # Number of complete passes through the dataset
        self.num_eval_samples = 100  # Number of samples to use for loss estimation
        self.estimate_loss_after = 1  # Estimate loss every N steps

        # Learning rate scheduling parameters
        self.max_learning_rate = 6e-4  # Peak learning rate
        self.min_learning_rate = (
            self.max_learning_rate * 0.1
        )  # Minimum learning rate (10% of max)
        self.warmup_steps = 10  # Steps to warm up from 0 to max learning rate
        self.max_steps = 100  # Total training steps

        # Initialize optimizer with AdamW and weight decay for regularization
        self.optimzer = self.model.configure_optimizers(
            learning_rate=self.max_learning_rate, weight_decay=0.10, device=device
        )

    ## Define function to estimate loss ##
    def estimate_loss(self):
        """
        Estimate average loss on both training and validation sets.
        This provides a more stable estimate than single-batch loss.

        Returns:
            dict: Contains 'train' and 'val' average losses
        """
        out = {}
        self.model.eval()  # Set model to evaluation mode (disables dropout, etc.)

        # Evaluate on both train and validation splits
        for split in ["train", "val"]:
            losses = torch.zeros(self.num_eval_samples)

            # Calculate loss over multiple samples for stable estimate
            for k in range(self.num_eval_samples):
                X, Y = self.dataloader.get_batch(split)
                X = X.to(device)
                Y = Y.to(device)

                # Forward pass without gradient computation (eval mode)
                _, loss = self.model(X, Y)
                losses[k] = loss.item()

            # Store average loss for this split
            out[split] = losses.mean()
        return out

    def get_lr(self, step):
        """
        Implement learning rate scheduling with warmup and cosine annealing.

        - Warmup: Linear increase from 0 to max_lr over warmup_steps
        - Cosine annealing: Smooth decay from max_lr to min_lr using cosine function
        - Constant: min_lr after max_steps

        Args:
            step (int): Current training step

        Returns:
            float: Learning rate for current step
        """
        if step < self.warmup_steps:
            # Linear warmup: gradually increase learning rate
            lr = self.max_learning_rate * (step) / self.warmup_steps
        elif step > self.max_steps:
            # After max steps, use minimum learning rate
            lr = self.min_learning_rate
        else:
            # Cosine annealing: smooth decay using cosine function
            decay_ratio = (step - self.warmup_steps) / (
                self.max_steps - self.warmup_steps
            )
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Cosine coefficient
            lr = self.min_learning_rate + coeff * (
                self.max_learning_rate - self.min_learning_rate
            )
        return lr

    def train(self):
        """
        Main training loop that implements the full training procedure.
        Includes gradient clipping, learning rate scheduling, and progress monitoring.
        """
        ## Start training ##
        # Set precision for matrix multiplications (improves performance on modern GPUs)
        torch.set_float32_matmul_precision("high")

        # Main training loop over epochs
        for epoch in range(self.num_epochs):
            # Process all batches in the current epoch
            for step in range(self.dataloader.total_batches):
                start_time = time.time()  # Track step timing

                # Get training batch and move to device
                x, y = self.dataloader.get_batch()
                x, y = x.to(device), y.to(device)

                # Reset gradients from previous step
                self.optimzer.zero_grad()

                # Forward pass: compute predictions and loss
                logits, loss = self.model(x, y)

                # Backward pass: compute gradients
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                # This stabilizes training by limiting gradient magnitude
                norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0
                )  # Clip gradients to max norm of 1.0

                # Update learning rate based on current step
                lr = self.get_lr(step)
                for param_group in self.optimzer.param_groups:
                    param_group["lr"] = lr

                # Optional: Print gradient norm and learning rate for debugging
                # print(f"Gradient norm: {norm: .4e} | Learning rate: {lr: .4e}")

                # Apply gradients to update model parameters
                self.optimzer.step()

                # Synchronize CUDA operations for accurate timing
                if device == "cuda":
                    torch.cuda.synchronize()
                end_time = time.time()

                # Calculate training throughput (tokens processed per second)
                tokens_per_second = (
                    self.dataloader.batch_size
                    * self.dataloader.block_size
                    / (end_time - start_time)
                )

                # Periodically estimate loss on train/val sets for monitoring
                if step % self.estimate_loss_after == 0:
                    losses = self.estimate_loss()

                # Print comprehensive training statistics
                print(
                    f"Epoch {epoch} | Loss: {losses['train']} | Val Loss: {losses['val']} | "
                    f"Tokens per second: {tokens_per_second} | Time taken: {end_time - start_time} seconds | "
                    f"Gradient norm: {norm: .4e} | Learning rate: {lr: .4e}"
                )

        # Save trained model parameters to disk
        torch.save(trainer.model.state_dict(), "gpt2_trained_model.pth")


if __name__ == "__main__":
    # Create trainer instance and start training
    trainer = Trainer()
    trainer.train()

    # Load the trained model and generate sample text
    trainer.model.load_state_dict(torch.load("gpt2_trained_model.pth"))
    trainer.model.to(device)

    # Generate sample text using the trained model
    generate(
        num_sequences=3,  # Generate 3 different sequences
        max_length=30,  # Maximum length of each sequence
        model=trainer.model,
        context="Hello, I'm a language model,",  # Starting prompt
        device=device,
    )
