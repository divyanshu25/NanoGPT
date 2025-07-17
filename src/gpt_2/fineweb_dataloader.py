"""
FineWeb DataLoader for GPT-2 Training

This module provides a data loader for the FineWeb-Edu dataset, designed for training
GPT-2 models on educational web content. The loader handles sharded data files and
supports distributed training across multiple processes.

The FineWeb-Edu dataset contains educational web content that has been tokenized
and split into shards for efficient training. Each shard contains approximately
100M tokens and is stored as a numpy array.

Example:
    >>> dataloader = FinewebDataloader(
    ...     data_dir="edu_fineweb10B",
    ...     batch_size=128,
    ...     block_size=1024,
    ...     split="train"
    ... )
    >>> x, y = dataloader.next_batch()
    >>> print(f"Input shape: {x.shape}, Target shape: {y.shape}")
    Input shape: torch.Size([128, 1024]), Target shape: torch.Size([128, 1024])
"""

import os
import torch
import random
import numpy as np


class FinewebDataloader:
    """
    A data loader for the FineWeb-Edu dataset optimized for GPT-2 training.

    This class manages loading and batching of tokenized data from sharded files.
    It supports distributed training by handling different data partitions for
    different processes and provides efficient sequential access to training data.

    The loader automatically handles shard transitions and maintains proper
    positioning for distributed training scenarios.

    Attributes:
        data_dir (str): Directory containing the data shards
        batch_size (int): Number of sequences per batch
        block_size (int): Length of each training sequence
        ddp_world_size (int): Total number of processes in distributed training
        ddp_rank (int): Rank of current process in distributed training
        split (str): Dataset split ('train' or 'val')
        shards (list): List of available data shard filenames
        current_shard (int): Index of currently loaded shard
        tokens (torch.Tensor): Currently loaded token data
        current_position (int): Current position within the token tensor
    """

    def __init__(
        self,
        data_dir="edu_fineweb10B",
        batch_size=128,
        block_size=1024,
        ddp_world_size=1,
        ddp_rank=0,
        split="train",
        master_process=True,
    ):
        """
        Initialize the FineWeb data loader.

        Args:
            data_dir (str, optional): Directory containing the data shards.
                Defaults to "edu_fineweb10B".
            batch_size (int, optional): Number of sequences per batch.
                Defaults to 128.
            block_size (int, optional): Length of each training sequence.
                Defaults to 1024.
            ddp_world_size (int, optional): Total number of processes in
                distributed training. Defaults to 1.
            ddp_rank (int, optional): Rank of current process in distributed
                training. Defaults to 0.
            split (str, optional): Dataset split to use. Must be either
                'train' or 'val'. Defaults to "train".
            master_process (bool, optional): Whether this is the master process.
                Controls logging output. Defaults to True.

        Raises:
            AssertionError: If split is not 'train' or 'val', or if no shards
                are found in the data directory.
        """
        self.data_dir = os.path.join(os.path.dirname(__file__), data_dir)
        print(f"Data directory: {self.data_dir}")

        self.batch_size = batch_size
        self.block_size = block_size
        self.ddp_world_size = ddp_world_size
        self.ddp_rank = ddp_rank
        self.split = split
        assert self.split in ["train", "val"], "split must be either train or val"

        shards = os.listdir(self.data_dir)
        shards = [s for s in shards if self.split in s]
        self.shards = sorted(shards)
        assert len(self.shards) > 0, "No shards found"
        if master_process:
            print(f"Found {len(self.shards)} shards for {self.split} split")

        self.reset()

    def load_tokens(self, shard):
        """
        Load tokenized data from a shard file.

        Args:
            shard (str): Path to the shard file to load.

        Returns:
            torch.Tensor: Tensor containing the tokenized data with dtype
                torch.long and shape (num_tokens,).

        Note:
            The shard files are stored as numpy arrays and are converted to
            torch tensors for efficient GPU processing. The data is cast to
            int32 to ensure compatibility with PyTorch's long dtype.
        """
        npt = np.load(shard)
        npt = npt.astype(np.int32)  # added after video
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt

    def reset(self):
        """
        Reset the data loader to a random starting position.

        This method initializes the loader by:
        1. Selecting a random shard to start from
        2. Loading the token data from that shard
        3. Setting the position based on the distributed training rank

        This ensures that different training runs start from different
        positions in the dataset, providing better randomization.

        Note:
            The starting position is calculated as:
            batch_size * block_size * ddp_rank

            This ensures that different processes in distributed training
            start at different positions within the same shard.
        """
        # state, init at shard zero
        self.current_shard = random.randint(0, len(self.shards) - 1)
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        self.current_position = self.batch_size * self.block_size * self.ddp_rank

    def next_batch(self):
        """
        Get the next batch of training data.

        Returns a batch of input-target pairs for training. The inputs are
        sequences of tokens, and the targets are the same sequences shifted
        by one position (for next-token prediction).

        Returns:
            tuple: A pair of tensors (x, y) where:
                - x (torch.Tensor): Input sequences with shape
                  (batch_size, block_size)
                - y (torch.Tensor): Target sequences with shape
                  (batch_size, block_size)

        Note:
            - The method automatically advances to the next shard when the
              current shard is exhausted
            - In distributed training, each process gets a different subset
              of the data based on its rank
            - The position advances by batch_size * block_size * ddp_world_size
              to ensure no overlap between processes
        """
        buf = self.tokens[
            self.current_position : self.current_position
            + self.batch_size * self.block_size
            + 1
        ]
        x = (buf[:-1]).view(self.batch_size, self.block_size)  # inputs
        y = (buf[1:]).view(self.batch_size, self.block_size)  # targets
        # advance the position in the tensor
        self.current_position += self.batch_size * self.block_size * self.ddp_world_size
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (
            self.batch_size * self.block_size * self.ddp_world_size + 1
        ) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_position = self.batch_size * self.block_size * self.ddp_rank
        return x, y
