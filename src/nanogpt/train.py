"""
Training module for NanoGPT model.

This module contains the Trainer class and training utilities for training
a NanoGPT model on text data. It handles the training loop, optimization,
and model evaluation.
"""

import torch
from nanogpt.dataset import ShakespeareDataset
from nanogpt.nanogpt import NanoGPT


# Set device to CUDA if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# set config
config = {
    "block_size": 128,  # context window size
    "batch_size": 32,  # batch size for training
    "lr": 3e-4,  # learning rate
    "device": device,  # device to run training on
    "epochs": 5000,  # number of epochs to train for
    "eval_iters": 200,  # number of iterations to evaluate the loss,
    "n_embed": 64,  # embedding dimension
    "eval_interval": 500,  # number of epochs to evaluate the loss
    "n_heads": 4,  # number of heads
    "n_blocks": 4,  # number of blocks
    "dropout": 0.2,  # dropout rate
}


class Trainer:
    """
    A trainer class for training NanoGPT models.

    This class encapsulates the training logic including optimization,
    loss computation, and the main training loop.

    Attributes:
        model: The NanoGPT model to be trained
        dataset: Dataset containing training and validation data
        block_size: Size of the context window for training
        batch_size: Number of samples per training batch
        lr: Learning rate for the optimizer
        device: Device (CPU/GPU) to run training on
        optimizer: AdamW optimizer for updating model parameters
    """

    def __init__(
        self, model, dataset, block_size, batch_size, lr, device, eval_interval
    ):
        """
        Initialize the Trainer with model and training parameters.

        Args:
            model: The NanoGPT model instance to train
            dataset: Dataset object containing train/val splits
            block_size (int): Context window size for training sequences
            batch_size (int): Number of samples per training batch
            lr (float): Learning rate for the AdamW optimizer
            device (str): Device to run training on ('cpu' or 'cuda')
        """
        self.model = model
        self.dataset = dataset
        self.block_size = block_size
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.eval_iters = 100
        self.eval_interval = eval_interval
        # Initialize AdamW optimizer with the specified learning rate
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    @torch.no_grad()  # this decorator is used to disable gradient computation
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.dataset.get_batch(split, self.block_size, self.batch_size)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def train_step(self, xb, yb):
        """
        Perform a single training step.

        This method computes the forward pass, calculates loss,
        performs backpropagation, and updates model parameters.

        Args:
            xb (torch.Tensor): Input batch of shape (batch_size, block_size)
            yb (torch.Tensor): Target batch of shape (batch_size, block_size)

        Returns:
            float: The loss value for this training step
        """
        # Forward pass: compute logits and loss
        logits, loss = self.model(xb, yb)

        # Zero out gradients from previous step
        self.optimizer.zero_grad(set_to_none=True)

        # Backward pass: compute gradients
        loss.backward()

        # Update model parameters
        self.optimizer.step()

        return loss.item()

    def train(self, epochs):
        """
        Train the model for a specified number of epochs.

        This method runs the main training loop, performing training steps
        for each epoch and printing progress information.

        Args:
            epochs (int): Number of training epochs to run
        """

        for epoch in range(epochs):

            # estimate the loss
            if epoch % self.eval_interval == 0 or epoch == epochs - 1:
                losses = self.estimate_loss()
                print(
                    f"Epoch {epoch} Train Loss: {losses['train']:.4f} Val Loss: {losses['val']:.4f}"
                )

            # Get a batch of training data
            xb, yb = self.dataset.get_batch("train", self.block_size, self.batch_size)

            # Perform training step and get loss
            loss = self.train_step(xb, yb)


if __name__ == "__main__":
    # Initialize and process the Shakespeare dataset
    dataset = ShakespeareDataset()
    # Create the NanoGPT model
    model = NanoGPT(dataset, config)
    dataset.train_test_split()

    # Initialize the trainer with hyperparameters
    trainer = Trainer(
        model=model,
        dataset=dataset,
        block_size=config["block_size"],  # Context window size
        batch_size=config["batch_size"],  # Batch size for training,
        lr=config["lr"],  # Learning rate
        device=config["device"],  # Training device (CPU/GPU)
        eval_interval=config["eval_interval"],  # number of epochs to evaluate the loss
    )

    # Train the model for 5000 epochs
    trainer.train(epochs=config["epochs"])
    print(f"Training complete.")

    # sample from the model
    print(
        dataset.decode(
            model.generate(
                torch.zeros((1, 1), dtype=torch.long), max_new_tokens=400
            ).tolist()[0]
        )
    )
