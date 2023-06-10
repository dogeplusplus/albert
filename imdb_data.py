import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling


def imdb_data_pipeline(batch_size: int = 16, quick_test: bool = False):
    imdb = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    def preprocess_function(examples):
        text = examples["text"]
        text = text.replace("[CLS]", "")
        text = text.replace("[SEP]", "")

        num_spaces = len(examples["text"].split(" ")) - 1
        midpoint = num_spaces // 2

        start = 0
        for _ in range(midpoint):
            start = examples["text"].find(" ", start+1)

        first = examples["text"][:start]
        second = examples["text"][start+1:]

        swap = float(np.random.randint(0, 2))
        if swap:
            first, second = second, first

        text = first + " [SEP] " + second
        examples["text"] = text

        tokens = tokenizer(examples["text"], truncation=True)
        tokens["swap"] = swap

        return tokens

    if quick_test:
        imdb["train"] = imdb["train"].select(range(batch_size))
        imdb["test"] = imdb["test"].select(range(batch_size))

    tokenized_imdb = imdb.map(preprocess_function, num_proc=8, remove_columns=imdb["train"].column_names)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        pad_to_multiple_of=8,
    )

    workers = os.cpu_count() // 2
    train_loader = DataLoader(
        tokenized_imdb["train"],
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=workers,
    )
    valid_loader = DataLoader(
        tokenized_imdb["test"],
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=workers,
    )

    return train_loader, valid_loader
