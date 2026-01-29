import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class SmileDataset(Dataset):
    def __init__(self, raw_datasets, data_type, tokenizer, max_length: int = None):
        self.data = raw_datasets[data_type]
        self.tokenizer = tokenizer
        self.max_length = int(max_length) if max_length is not None else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        src_input = sample["input"]  # 'input' is SMILES string

        src_smiles = self.tokenizer.bos_token + src_input + self.tokenizer.eos_token

        ids = self.tokenizer.encode(src_smiles, add_special_tokens=False)

        # Right-side truncation and padding to fixed length at sample level,
        # preventing variable-length stack error in DataLoader when falling back to default_collate.
        if self.max_length is not None:
            if len(ids) > self.max_length:
                ids = ids[: self.max_length]
            if len(ids) < self.max_length:
                pad_id = self.tokenizer.pad_token_id
                ids = ids + [pad_id] * (self.max_length - len(ids))

        smiles_data = torch.tensor(ids, dtype=torch.long)
        if self.max_length is not None:
            pad_id = self.tokenizer.pad_token_id
            attn = (smiles_data != pad_id).to(torch.long)
        else:
            # Degenerate case (should not happen): treat all as valid
            attn = torch.ones_like(smiles_data, dtype=torch.long)
        return {"input_ids": smiles_data, "attention_mask": attn}


class SmileCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        smiles_datas = batch

        # Padding (batch, max_len)
        padded_smiles_datas = pad_sequence(
            smiles_datas, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        x = padded_smiles_datas[:, :-1]
        y = padded_smiles_datas[:, 1:]

        return x, y

