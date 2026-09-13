"""Sliding-window Dataset for next-token prediction, plus a DataLoader helper."""

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    """Packages already-tokenized IDs into (input, target) training batches.

    Different job from the tokenizer: SimpleTokenizerV2 answers "what number is
    this word?"; this answers "how do I chop those numbers into batches the
    network trains on?" It takes BPE-encoded IDs and slides a window over them,
    where each target is the input shifted right by one — the next-token setup.
    (This is the mini-batch DataLoader step, like feeding a digit net; it also
    manufactures the input/target pairs on the fly.)

    Sliding window over "In the heart of the city stood the old library, a ..."
    with max_length=4:

        x (inputs)              y (targets = x shifted right by 1)
        [ In   the   heart of ] [ the   heart of    the ]
        [ the  city  stood the] [ city  stood the   old ]
        [ old  library ,    a  ] [ library ,   a     relic]
        ...

    Each x row is one input context; the matching y row is what comes next.
    """

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, \
            "Number of tokenized inputs must at least be equal to max_length+1"

        # Slide a max_length window across the tokens, stepping by stride.
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    """Wrap raw text in a GPTDatasetV1 and return a DataLoader (drop_last keeps batches equal-sized)."""
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    # Small 16-token story so the windows are easy to eyeball.
    story = "In the heart of the city stood the old library a relic from a bygone era"

    dataloader = create_dataloader_v1(
        story, batch_size=2, max_length=4, stride=2,
        shuffle=False, drop_last=False,
    )

    bpe = tiktoken.get_encoding("gpt2")
    for batch, (inputs, targets) in enumerate(dataloader):
        print(f"\nBatch {batch}  inputs {tuple(inputs.shape)}")
        for x, y in zip(inputs, targets):
            # target is the input shifted right by one (next-token prediction).
            print("  in :", bpe.decode(x.tolist()))
            print("  out:", bpe.decode(y.tolist()))
