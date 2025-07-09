import urllib.request
import os
import torch
from typing import List, Tuple, Dict, Optional

# set random seed
torch.manual_seed(1337)
# set device
device = "cuda" if torch.cuda.is_available() else "cpu"


class ShakespeareDataset:
    """
    A dataset class for handling the Tiny Shakespeare dataset.

    This class provides methods to download the dataset from a remote URL,
    process the text data, create vocabulary mappings, and generate training batches
    for character-level language modeling tasks.

    Attributes:
        data_file_path (str): Local path where the dataset is stored
        data_url (str): URL to download the Tiny Shakespeare dataset
        chars (List[str]): List of unique characters in the dataset
        vocab_size (int): Size of the vocabulary (number of unique characters)
        stoi (Dict[str, int]): Character to integer mapping
        itos (Dict[int, str]): Integer to character mapping
        tensor_data (Optional[torch.Tensor]): Full dataset as tensor
        train_data (Optional[torch.Tensor]): Training split of the data
        val_data (Optional[torch.Tensor]): Validation split of the data
        block_size (int): Default context length for training
        batch_size (int): Default batch size for training
    """

    def __init__(self):
        """
        Initializes the ShakespeareDataset instance.

        Sets up file paths, URL, and initializes data structures for
        vocabulary mapping and data storage.
        """
        self.data_file_path = "data/input.txt"
        self.data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        self.chars = []
        self.vocab_size = 0
        self.stoi = {}
        self.itos = {}
        self.tensor_data = None
        self.train_data = None
        self.val_data = None

    def download_data(self) -> None:
        """
        Downloads the Tiny Shakespeare dataset if it does not already exist locally.

        Creates the data directory if it doesn't exist and downloads the dataset
        from the specified URL to the local file path.

        Raises:
            urllib.error.URLError: If the download fails
            OSError: If directory creation fails
        """
        # Create data directory if it doesn't exist
        if not os.path.exists(os.path.dirname(self.data_file_path)):
            os.makedirs(os.path.dirname(self.data_file_path))

        # Download the dataset from URL to local file
        urllib.request.urlretrieve(self.data_url, self.data_file_path)

    def load_data(self) -> str:
        """
        Loads the contents of the Tiny Shakespeare dataset from the local file.

        Returns:
            str: The complete text content of the dataset

        Raises:
            FileNotFoundError: If the data file doesn't exist
        """
        with open(self.data_file_path, "r") as f:
            return f.read()

    def process_data(self) -> None:
        """
        Processes the raw text data to create vocabulary mappings.

        This method:
        1. Loads the raw text data
        2. Creates a sorted list of unique characters
        3. Builds character-to-index and index-to-character mappings
        4. Sets the vocabulary size

        Note:
            Must be called after download_data() to ensure data is available.
        """
        data = self.load_data()
        # Create sorted list of unique characters
        self.chars = sorted(list(set(data)))
        self.vocab_size = len(self.chars)
        # Create character to index mapping
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        # Create index to character mapping
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text: str) -> List[int]:
        """
        Converts text string to a list of token indices.

        Args:
            text (str): Input text to encode

        Returns:
            List[int]: List of token indices corresponding to the input text

        Raises:
            KeyError: If a character in text is not in the vocabulary
        """
        return [self.stoi[c] for c in text]

    def decode(self, tokens: List[int]) -> str:
        """
        Converts a list of token indices back to text string.

        Args:
            tokens (List[int]): List of token indices to decode

        Returns:
            str: Decoded text string

        Raises:
            KeyError: If a token index is not in the vocabulary
        """
        return "".join([self.itos[i] for i in tokens])

    def train_test_split(
        self, train_size: float = 0.9
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Splits the dataset into training and validation sets.

        Args:
            train_size (float): Proportion of data to use for training (default: 0.9)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Training and validation tensors

        Note:
            This method also sets self.train_data and self.val_data for use in get_batch().
        """
        data = self.load_data()
        # Convert text to tensor of token indices
        self.tensor_data = torch.tensor(self.encode(data), dtype=torch.long)
        # Calculate split point
        n = int(train_size * len(self.tensor_data))
        # Split data into train and validation sets
        self.train_data = self.tensor_data[:n]
        self.val_data = self.tensor_data[n:]
        return self.train_data, self.val_data

    def get_batch(
        self, split: str, block_size: int, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a batch of training data for language modeling.

        Creates input sequences (x) and target sequences (y) where y is x shifted by 1 position.
        This is used for next-token prediction tasks.

        Args:
            split (str): Either "train" or "val" to select which dataset to use
            block_size (int): Length of input sequences (context length)
            batch_size (int): Number of sequences in the batch

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - x: Input sequences of shape (batch_size, block_size)
                - y: Target sequences of shape (batch_size, block_size)

        Raises:
            ValueError: If data hasn't been initialized via train_test_split()
            AssertionError: If data length is insufficient for the given block_size
        """
        # Select appropriate dataset split
        data = self.train_data if split == "train" else self.val_data
        if data is None:
            raise ValueError(f"Data not initialized. Call train_test_split() first.")

        # Ensure we have enough data for the block size
        assert (
            len(data) >= block_size + 1
        ), f"Data length {len(data)} must be at least {block_size + 1}"

        # Generate random starting indices for sequences
        ix = torch.randint(len(data) - block_size, (batch_size,))

        # Create input sequences (x) and target sequences (y)
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x, y


if __name__ == "__main__":
    """
    Example usage: Downloads, processes, and demonstrates the Shakespeare dataset.

    This example:
    1. Downloads the Tiny Shakespeare dataset
    2. Processes the text to create vocabulary mappings
    3. Demonstrates encoding/decoding functionality
    4. Splits data into train/validation sets
    5. Generates a sample training batch
    """
    dataset = ShakespeareDataset()
    dataset.download_data()
    dataset.process_data()

    # Demonstrate encoding/decoding
    print(dataset.encode("Hello, world!"))
    print(dataset.decode(dataset.encode("Hello, world!")))

    # Split data and generate batch
    train_data, val_data = dataset.train_test_split()
    print(
        f"Training data shape: {train_data.shape}, Validation data shape: {val_data.shape}"
    )

    x, y = dataset.get_batch("train", 8, 4)
    print(f"Input batch shape: {x.shape}")
    print(f"Target batch shape: {y.shape}")
