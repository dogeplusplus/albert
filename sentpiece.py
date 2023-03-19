import datasets
import itertools
import pandas as pd
import sentencepiece as spm

from pathlib import Path
from tempfile import TemporaryDirectory


def split_sentences(sample):
    article = sample["article"]
    article = article.replace(" \n ", " ")
    article = article.replace("  ", " ")
    sentences = article.split(" . ")
    return sentences


def train_tokenizer(dataset, vocab_size=30000, out_file="pubmed"):
    sentences = itertools.chain(*[split_sentences(s) for s in dataset])

    with TemporaryDirectory() as temp_dir:
        data_path = Path(temp_dir, "dataset.txt")
        with open(data_path, "w") as f:
            f.writelines(sentences)

        spm.SentencePieceTrainer.train(
            input=str(data_path),
            model_prefix=out_file,
            vocab_size=vocab_size,
            user_defined_symbols=["[CLS]", "[SEP]", "[MASK]", "[PAD]"],
        )


def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def create_mlm_dataset(dataset_name="ccdv/pubmed-summarization", out_path="pubmed.csv"):
    dataset = datasets.load_dataset(dataset_name)
    rows = []
    for x in dataset["train"]:
        sentences = split_sentences(x)
        sentence_pairs = pairwise(sentences)
        for pair in sentence_pairs:
            rows.append({"first_sentence": pair[0], "second_sentence": pair[1]})

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
