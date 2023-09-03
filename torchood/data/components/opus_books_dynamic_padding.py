from typing import Any, List, Tuple, Union

import torch
from torch.utils.data import Dataset


def casual_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask


class BilingualDataset(Dataset):
    def __init__(self, ds, seq_len, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang) -> None:
        super().__init__()

        self.ds = ds
        self.seq_len = seq_len
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int):
        data = self.ds[index]
        src = data["translation"][self.src_lang]
        tgt = data["translation"][self.tgt_lang]

        # Transform the text into tokens
        src_ids = self.tokenizer_src.encode(src).ids  # encode input
        tgt_ids = self.tokenizer_tgt.encode(tgt).ids  # decoder input

        return src_ids, tgt_ids, src, tgt

    def make_item(self, src_ids, tgt_ids, src, tgt, seq_len):
        #  addsos, eos and pad tokens
        src_num_padding = seq_len - len(src_ids) - 2  # we will add <s> and </s>
        # here only add <s>, & </s> only in the label
        tgt_num_padding = seq_len - len(tgt_ids) - 1

        # Make sure that pad is non negative, if it is then sentence is to long
        if src_num_padding < 0 or tgt_num_padding < 0:
            raise ValueError("Sentence is too long")

        enoder_input = [
            self.sos_token,
            torch.tensor(src_ids, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * src_num_padding, dtype=torch.int64),
        ]
        enoder_input = torch.cat(enoder_input, dim=0)

        decoder_input = [
            self.sos_token,
            torch.tensor(tgt_ids, dtype=torch.int64),
            torch.tensor([self.pad_token] * tgt_num_padding, dtype=torch.int64),
        ]
        decoder_input = torch.cat(decoder_input, dim=0)

        # add only </s>
        label = [
            torch.tensor(tgt_ids, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * tgt_num_padding, dtype=torch.int64),
        ]
        label = torch.cat(label, dim=0)

        # double check the size of tensors is correct
        assert enoder_input.size(0) == seq_len
        assert decoder_input.size(0) == seq_len
        assert label.size(0) == seq_len

        return {
            "encoder_input": enoder_input,
            "decoder_input": decoder_input,
            "label": label,
            "encoder_mask": (enoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),  # 1, 1, seq_len
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int()
            & casual_mask(decoder_input.size(0)),  # (1, seq_len) & (1, seq_len, seq_len)
            "src_text": src,
            "tgt_text": tgt,
        }

    def collate_fn(self, batch: Any) -> Tuple[torch.Tensor, Any]:
        # encoder_input_max = max(x["encoder_str_length"] for x in batch)
        # decoder_input_max = max(x["decoder_str_length"] for x in batch)
        encoder_inputs = []
        decoder_inputs = []
        encoder_mask = []
        decoder_mask = []
        label = []
        src_text = []
        tgt_text = []
        for x in batch:
            src_ids, tgt_ids, src, tgt = x[0], x[1], x[2], x[3]
            seq_len = len(src)
            b = self.make_item(src_ids, tgt_ids, src, tgt, seq_len)
            encoder_inputs.append(b["encoder_input"])
            decoder_inputs.append(b["decoder_input"])
            encoder_mask.append(b["encoder_mask"])
            decoder_mask.append(b["decoder_mask"])
            label.append(b["label"])
            src_text.append(b["src_text"])
            tgt_text.append(b["tgt_text"])
        return {
            "encoder_input": torch.stack(encoder_inputs, 0),
            "decoder_input": torch.stack(decoder_inputs, 0),
            "encoder_mask": torch.stack(encoder_mask, 0),
            "decoder_mask": torch.stack(decoder_mask, 0),
            "label": torch.stack(label, 0),
            "src_text": torch.stack(src_text, 0),
            "tgt_text": torch.stack(tgt_text, 0),
        }
