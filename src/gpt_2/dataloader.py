import torch
import tiktoken


class DataLoader:
    def __init__(
        self,
        data_file,
        batch_size,
        block_size,
        ddp_world_size,
        ddp_rank,
        ddp_local_rank,
    ):
        self.data = open(data_file, "r", encoding="utf-8").read()
        self.enc = tiktoken.get_encoding("gpt2")
        self.tokens = self.enc.encode(self.data)
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        self.train_data = self.tokens[: int(0.9 * len(self.tokens))]
        self.val_data = self.tokens[int(0.9 * len(self.tokens)) :]
        self.batch_size = batch_size
        self.block_size = block_size
        self.idx = 0
        self.current_index = 0
        self.total_batches = len(self.tokens) // (self.block_size)
        self.total_train_batches = len(self.train_data) // (self.block_size)
        self.total_val_batches = len(self.val_data) // (self.block_size)
        self.ddp_world_size = ddp_world_size
        self.ddp_rank = ddp_rank
        self.ddp_local_rank = ddp_local_rank

        print(
            f"Total tokens: {len(self.tokens)} , Total train tokens: {len(self.train_data)} , Total val tokens: {len(self.val_data)}"
        )
        print(
            f"Total batches: {self.total_batches}, Total train batches: {self.total_train_batches}, Total val batches: {self.total_val_batches} for 1 epoch"
        )

    def get_batch(self, split="train"):
        if split == "train":
            data = self.train_data
        else:
            data = self.val_data

        if self.current_index + (self.batch_size * self.block_size + 1) > len(data):
            self.current_index = 0
        data_size = self.batch_size * self.block_size
        current_batch = data[self.current_index : self.current_index + data_size].view(
            self.batch_size, self.block_size
        )
        target_batch = data[
            self.current_index + 1 : self.current_index + data_size + 1
        ].view(self.batch_size, self.block_size)
        self.current_index += self.batch_size * self.block_size
        return current_batch, target_batch

    def __len__(self):
        return len(self.tokens)


if __name__ == "__main__":
    data_file = "/Users/divyanshugoyal/workspace/nanogpt/src/data/input.txt"
    dataloader = DataLoader(data_file=data_file, batch_size=4, block_size=10)

    for i in range(10000):
        x, y = dataloader.get_batch()
        print(x.shape, y.shape)
        # print(f"Input: {dataloader.enc.decode(x[0].tolist())}")
        # print(f"Target: {dataloader.enc.decode(y[0].tolist())}")
