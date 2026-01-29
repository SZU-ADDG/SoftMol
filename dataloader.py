import json
import math
import os
import re
import typing
from typing import Optional

import datasets
import fsspec
import tokenizers
import torch
import transformers

import utils
from hydra.utils import to_absolute_path
from tokenizer import SmilesTokenizer
from dataset import SmileDataset
import torch.nn.functional as F

LOGGER = utils.get_logger(__name__)


class SAFETokenizerBOSAdapter:
  """SAFE tokenizer.json adapter: exposes [BOS]/[EOS] and maps them to internal [CLS]/[SEP] IDs."""

  _REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""

  class _SafeSplitter:
    name = "safe"

    def __init__(self, pattern: Optional[str] = None):
      self.regex = re.compile(pattern or SAFETokenizerBOSAdapter._REGEX_PATTERN)

    def tokenize(self, line):
      if isinstance(line, str):
        tokens = list(self.regex.findall(line))
        if "".join(tokens) != line:
          raise ValueError(line)
        return tokens
      idxs = re.finditer(self.regex, str(line))
      return [line[m.start(0):m.end(0)] for m in idxs]

    def split(self, _n, normalized):
      return self.tokenize(normalized)

    def pre_tokenize(self, pretok):
      pretok.split(self.split)

  def __init__(self, tokenizer_json_path: str):
    self.tokenizer_json_path = str(tokenizer_json_path)
    # External: keep consistent with original SMILES flow
    self.bos_token = '[BOS]'
    self.eos_token = '[EOS]'
    self.sep_token = '[SEP]'
    self.cls_token = '[CLS]'
    self.pad_token = '[PAD]'
    self.mask_token = '[MASK]'
    self.unk_token = '[UNK]'
    # Internal tokenizer (lazy loaded for multiprocessing compatibility)
    self._tok = None
    self._internal_bos = None
    self._internal_eos = None

  def __getstate__(self):
    d = dict(self.__dict__)
    d["_tok"] = None
    return d

  def __setstate__(self, d):
    self.__dict__.update(d)

  def _ensure_loaded(self):
    if self._tok is not None:
      return
    if not os.path.isfile(self.tokenizer_json_path):
      raise FileNotFoundError(f"tokenizer.json not found: {self.tokenizer_json_path}")
    with open(self.tokenizer_json_path, "r", encoding="utf-8") as f:
      data = json.load(f)

    custom_pre = bool(data.pop("custom_pre_tokenizer", False))
    data.pop("tokenizer_type", None)
    data.pop("tokenizer_attrs", None)
    tok = tokenizers.Tokenizer.from_str(json.dumps(data, ensure_ascii=False))

    if custom_pre:
      from tokenizers.pre_tokenizers import PreTokenizer
      tok.pre_tokenizer = PreTokenizer.custom(self._SafeSplitter())

    vocab = tok.get_vocab()

    # SAFE uses [CLS]/[SEP] as BOS/EOS by default; also compatible if [BOS]/[EOS] exist
    for cand in ("[CLS]", "[BOS]"):
      if cand in vocab:
        self._internal_bos = cand
        break
    for cand in ("[SEP]", "[EOS]"):
      if cand in vocab:
        self._internal_eos = cand
        break
    if self._internal_bos is None or self._internal_eos is None:
      raise ValueError(
        "SAFE tokenizer missing required BOS/EOS tokens (expected [CLS]/[SEP] or [BOS]/[EOS])"
      )
    self._tok = tok

  @property
  def vocab_size(self) -> int:
    self._ensure_loaded()
    return len(self._tok.get_vocab())

  def get_vocab(self) -> typing.Dict[str, int]:
    self._ensure_loaded()
    return self._tok.get_vocab()

  def _token_to_id(self, token: str) -> int:
    self._ensure_loaded()
    tid = self._tok.token_to_id(token)
    if tid is None:
      raise ValueError(f"token not in vocab: {token}")
    return int(tid)

  @property
  def bos_token_id(self) -> int:
    self._ensure_loaded()
    return self._token_to_id(self._internal_bos)

  @property
  def eos_token_id(self) -> int:
    self._ensure_loaded()
    return self._token_to_id(self._internal_eos)

  @property
  def pad_token_id(self) -> int:
    return self._token_to_id(self.pad_token)

  @property
  def mask_token_id(self) -> int:
    return self._token_to_id(self.mask_token)

  @property
  def unk_token_id(self) -> int:
    return self._token_to_id(self.unk_token)

  def encode(self, text, add_special_tokens: bool = True, **kwargs):
    """Returns List[int] (aligned with existing SmilesTokenizer.encode)."""
    self._ensure_loaded()
    if isinstance(text, str):
      mapped = text.replace(self.bos_token, self._internal_bos).replace(self.eos_token, self._internal_eos)
      return self._tok.encode(mapped, add_special_tokens=add_special_tokens).ids
    mapped = [
      t.replace(self.bos_token, self._internal_bos).replace(self.eos_token, self._internal_eos)
      for t in text
    ]
    encs = self._tok.encode_batch(mapped, add_special_tokens=add_special_tokens)
    return [e.ids for e in encs]

  def decode(self, ids, skip_special_tokens: bool = False, **kwargs) -> str:
    self._ensure_loaded()
    if hasattr(ids, "tolist"):
      ids = ids.tolist()
    out = self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)
    return out.replace(self._internal_bos, self.bos_token).replace(self._internal_eos, self.eos_token)

  def batch_decode(self, sequences, skip_special_tokens: bool = False, **kwargs) -> typing.List[str]:
    self._ensure_loaded()
    if hasattr(sequences, "tolist"):
      sequences = sequences.tolist()
    outs = self._tok.decode_batch(sequences, skip_special_tokens=skip_special_tokens)
    return [
      s.replace(self._internal_bos, self.bos_token).replace(self._internal_eos, self.eos_token)
      for s in outs
    ]


def get_tokenizer(config):
  tok_path_raw = config.data.tokenizer_name_or_path
  # Hydra switches the cwd to hydra.run.dir; to_absolute_path resolves relative paths based on the "original working directory."
  tok_path = to_absolute_path(os.path.expanduser(tok_path_raw)) if isinstance(tok_path_raw, str) else tok_path_raw
  
  # Prefer local vocab/tokenizer file if provided
  if isinstance(tok_path, str) and os.path.isfile(tok_path):
    if tok_path.endswith(".json"):
      return SAFETokenizerBOSAdapter(tok_path)
    # Default to SMILES WordPiece (vocab.txt)
    tokenizer = SmilesTokenizer(tok_path)
    # Explicitly set special tokens to ensure consistency between training and sampling
    tokenizer.bos_token = '[BOS]'
    tokenizer.eos_token = '[EOS]'
    tokenizer.sep_token = '[SEP]'
    tokenizer.pad_token = '[PAD]'
    tokenizer.mask_token = '[MASK]'
    # Trigger ID initialization
    _ = tokenizer.bos_token_id
    _ = tokenizer.eos_token_id
    _ = tokenizer.pad_token_id
    return tokenizer

  raise ValueError(f"Could not load tokenizer from {tok_path_raw}.")


class SmilesLMDataCollator:
  """SMILES Batching: Padding to max_length, returns input_ids and attention_mask.

  - Right-side padding to `max_length`; truncate if too long.
  - `attention_mask` is 1 for valid tokens, 0 for padding.
  """
  def __init__(self, tokenizer, max_length: int):
    self.tokenizer = tokenizer
    self.max_length = int(max_length)

  def __call__(self, batch):
    pad_id = self.tokenizer.pad_token_id
    # Supports two input formats:
    # - List[Tensor]
    # - List[Dict[str, Tensor]] (must contain 'input_ids')
    if isinstance(batch[0], dict):
      seqs = [b['input_ids'] for b in batch]
    else:
      seqs = batch
    # Truncate
    seqs = [b[:self.max_length] for b in seqs]
    # Pad to longest in batch first, then pad to max_length
    padded = torch.nn.utils.rnn.pad_sequence(
      seqs, batch_first=True, padding_value=pad_id)
    if padded.size(1) < self.max_length:
      diff = self.max_length - padded.size(1)
      padded = F.pad(padded, (0, diff), value=pad_id)
    attn = (padded != pad_id).to(torch.long)
    return {'input_ids': padded, 'attention_mask': attn}
    

def get_dataloaders(config, tokenizer, skip_train=False,
                    skip_valid=False, valid_seed=None):
  num_gpus = torch.cuda.device_count()
  if config.trainer.accumulate_grad_batches > 1:
    assert (config.loader.global_batch_size
            == (config.loader.batch_size
                * config.trainer.num_nodes
                * num_gpus
                * config.trainer.accumulate_grad_batches))
    if config.loader.global_batch_size % (
      num_gpus * config.trainer.accumulate_grad_batches) != 0:
      raise ValueError(
        f'Train Global Batch Size {config.loader.global_batch_size} '
        f'not divisible by {num_gpus} gpus with accumulation '
        f'{config.trainer.accumulate_grad_batches}.')
  if config.loader.global_batch_size % (
    num_gpus * config.trainer.accumulate_grad_batches) != 0:
    raise ValueError(
      f'Train Global Batch Size {config.loader.global_batch_size} '
      f'not divisible by {num_gpus} gpus with accumulation '
      f'{config.trainer.accumulate_grad_batches}.')
  if config.loader.eval_global_batch_size % num_gpus != 0:
    raise ValueError(
      f'Eval Global Batch Size {config.loader.eval_global_batch_size} '
      f'not divisible by {num_gpus}.')
  
  # Explicit switch: Force SMILES data flow and assertions
  force_smiles = getattr(config.data, 'use_smiles', True) or \
                 getattr(config.data, 'format', 'smiles') == 'smiles'
  
  if not force_smiles:
      raise ValueError("Refactored code supports only SMILES data. Please check config.data.use_smiles or config.data.format.")

  smiles_path = getattr(config.data, 'smiles_path', None)
  if smiles_path is None:
    # Compatibility: Read from cache_dir if train/valid is set to 'smiles_disk'
    if str(getattr(config.data, 'train', '')).lower() == 'smiles_disk' or \
       str(getattr(config.data, 'valid', '')).lower() == 'smiles_disk':
      smiles_path = getattr(config.data, 'cache_dir', None)
  if smiles_path is not None:
    # Compatibility: Resolve absolute path for Hydra working directory changes
    try:
      abs_sp = to_absolute_path(smiles_path)
    except Exception:
      abs_sp = smiles_path
    if utils.fsspec_exists(abs_sp):
      smiles_path = abs_sp
      
  assert smiles_path is not None and utils.fsspec_exists(smiles_path), \
    f"data.use_smiles=True but valid smiles_path not found: {smiles_path}"

  raw_datasets = datasets.load_from_disk(smiles_path)
  
  if skip_train:
    train_loader = None
  else:
    train_set = SmileDataset(
      raw_datasets, data_type='train', tokenizer=tokenizer,
      max_length=config.model.length)
    collator = SmilesLMDataCollator(tokenizer, max_length=config.model.length)
    train_loader = torch.utils.data.DataLoader(
      train_set,
      batch_size=config.loader.batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=not config.data.streaming,
      collate_fn=collator,
      persistent_workers=True)
    train_loader.tokenizer = tokenizer

  if skip_valid:
    valid_loader = None
  else:
    valid_split_name = 'validation' if 'validation' in raw_datasets else 'valid'
    valid_set = SmileDataset(
      raw_datasets, data_type=valid_split_name, tokenizer=tokenizer,
      max_length=config.model.length)
    collator = SmilesLMDataCollator(tokenizer, max_length=config.model.length)
    valid_loader = torch.utils.data.DataLoader(
      valid_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=False, 
      generator=None,
      collate_fn=collator)
    valid_loader.tokenizer = tokenizer

  return train_loader, valid_loader


# Samplers adapted from: https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/fault_tolerant_sampler.py


class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):

  def __init__(self, *args, generator=None, **kwargs):
    # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
    # which should be reproducible if pl.seed_everything was called beforehand.
    # This means that changing the seed of the experiment will also change the
    # sampling order.
    if generator is None:
      seed = int(torch.empty((), dtype=torch.int64).random_().item())
      generator = torch.Generator().manual_seed(seed)
    kwargs.pop('shuffle', None)
    super().__init__(*args, generator=generator, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'random_state': self.generator.get_state(),
            'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.generator.set_state(state_dict.get('random_state'))
    self.counter = state_dict['counter']
    # self.start_counter = self.counter
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.

  def __iter__(self) -> typing.Iterator[int]:
    n = len(self.data_source)

    self.state = self.generator.get_state()
    indices = torch.randperm(n, generator=self.generator).tolist()

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'epoch': self.epoch, 'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.epoch = state_dict['epoch']
    self.counter = state_dict['counter']
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.
  def __iter__(self):
    if self.shuffle:
      # deterministically shuffle based on epoch and seed
      g = torch.Generator()
      g.manual_seed(self.seed + self.epoch)
      indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
    else:
      indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

    if not self.drop_last:
      # add extra samples to make it evenly divisible
      padding_size = self.total_size - len(indices)
      if padding_size <= len(indices):
        indices += indices[:padding_size]
      else:
        indices += (indices * math.ceil(
          padding_size / len(indices)))[:padding_size]
    else:
      # remove tail of data to make it evenly divisible.
      indices = indices[:self.total_size]
    assert len(indices) == self.total_size

    # subsample
    indices = indices[self.rank:self.total_size:self.num_replicas]
    assert len(indices) == self.num_samples

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0
