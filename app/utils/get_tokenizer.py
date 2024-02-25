import torch
from torch.utils.data import random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path


def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]


def get_of_build_tokenizer(config, ds, lang):
    """_summary_
     Hugging face reference code
    Args:
        config (_type_): config file with path to tokenizer file
        ds (_type_): _description_
        lang (_type_): foramt path by language
    """

    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokinazer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokinazer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokinazer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokinazer.save(str(tokenizer_path))

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))


def get_ds(config):
    ds_raw = load_dataset(
        "opus_books", f'{config["lang_src"]}-{config["lang_tgt"]}', split="train"
    )

    # Build Tokenizer
    tokenizer_src = get_of_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_of_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # Split train-valid

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = int(0.1 * len(ds_raw))

    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
