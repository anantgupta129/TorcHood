from pathlib import Path
from typing import Any, List, Optional, Union

from datasets import load_dataset
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader, Dataset, random_split

from .components.opus_books import BilingualDataset


def get_all_sentences(ds: Any, lang: str):
    for item in ds:
        yield item["translation"][lang]


def make_tokenizer(config: Any, ds: Any, lang: str) -> Tokenizer:
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not tokenizer_path.is_file():
        print(f"Tokenizer not found. Making for {lang}...")
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        print("Tokenizer found. Loading from file...")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


class OpusBooksDataModule(LightningDataModule):
    def __init__(self, config: Any, num_workers: int = 0, pin_memory: bool = False):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    def prepare_data(self) -> None:
        # opus_books only has train split, so we will divide it later
        load_dataset(
            "opus_books",
            f"{self.hparams.config['lang_src']}-{self.hparams.config['lang_tgt']}",
            split="train",
        )

    def setup(self, stage: str) -> None:
        # load  only if not loaded already
        if not self.data_train and not self.data_val:
            lang_src = self.hparams.config["lang_src"]
            lang_tgt = self.hparams.config["lang_tgt"]

            ds_raw = load_dataset("opus_books", f"{lang_src}-{lang_tgt}", split="train")

            # build tokenizer
            tokenizer_src = make_tokenizer(self.hparams.config, ds_raw, lang_src)
            tokenizer_tgt = make_tokenizer(self.hparams.config, ds_raw, lang_tgt)

            sorted_ds_raw = sorted(ds_raw, key=lambda x: len(x["translation"][lang_src]))
            filtered_sorted_ds_raw = [
                k for k in sorted_ds_raw if len(k["translation"][lang_src]) < 150
            ]
            filtered_sorted_ds_raw = [
                k for k in filtered_sorted_ds_raw if len(k["translation"][lang_tgt]) < 150
            ]
            filtered_sorted_ds_raw = [
                k
                for k in filtered_sorted_ds_raw
                if len(k["translation"][lang_src]) + 10 > len(k["translation"][lang_tgt])
            ]

            # train val split
            train_size = int(0.9 * len(filtered_sorted_ds_raw))
            val_size = len(filtered_sorted_ds_raw) - train_size
            train_ds_raw, val_ds_raw = random_split(filtered_sorted_ds_raw, [train_size, val_size])

            self.data_train = BilingualDataset(
                train_ds_raw,
                self.hparams.config["seq_len"],
                tokenizer_src,
                tokenizer_tgt,
                lang_src,
                lang_tgt,
            )
            self.data_val = BilingualDataset(
                val_ds_raw,
                self.hparams.config["seq_len"],
                tokenizer_src,
                tokenizer_tgt,
                lang_src,
                lang_tgt,
            )
            self.tokenizer_src = tokenizer_src
            self.tokenizer_tgt = tokenizer_tgt

            # finding max  len of each sentence in source and target dataset
            max_len_src = 0
            max_len_tgt = 0
            for item in ds_raw:
                src_ids = tokenizer_src.encode(item["translation"][lang_src]).ids
                tgt_ids = tokenizer_tgt.encode(item["translation"][lang_tgt]).ids
                max_len_src = max(max_len_src, len(src_ids))
                max_len_tgt = max(max_len_tgt, len(tgt_ids))

            print("[x] Maximum Length of Sentence:-")
            print(
                "\t[*] source sentence: %d\n\t[*] target sentence: %d" % (max_len_src, max_len_tgt)
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.config["batch_size"],
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.data_train.collate_fn,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.data_val.collate_fn,
        )
