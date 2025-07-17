import os
import torch
import random
import numpy as np
import json
import pickle
import math
import time
import logging


class FinewebDataloader:
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
        Args:
            data_dir: The directory containing the data shards.
            batch_size: The batch size.
            block_size: The block size.
            ddp_world_size: The number of processes in the distributed training.
            ddp_rank: The rank of the current process.
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
        npt = np.load(shard)
        npt = npt.astype(np.int32)  # added after video
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt

    def reset(self):
        # state, init at shard zero
        self.current_shard = random.randint(0, len(self.shards) - 1)
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        self.current_position = self.batch_size * self.block_size * self.ddp_rank

    def next_batch(self):
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
