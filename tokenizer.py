import collections
import os
import re
from logging import getLogger
from typing import List, Optional

from transformers import BertTokenizer

logger = getLogger(__name__)
"""
SMI_REGEX_PATTERN: str
SMILES regex pattern for tokenization. Designed by Schwaller et. al.

References
----------
.. [1]  Philippe Schwaller, Teodoro Laino, Théophile Gaudin, Peter Bolgar, Christopher A. Hunter, Costas Bekas, and Alpha A. Lee
ACS Central Science 2019 5 (9): Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction
1572-1583 DOI: 10.1021/acscentsci.9b00576
"""

SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""


class SmilesTokenizer(BertTokenizer):
    """
    Creates the SmilesTokenizer class. The tokenizer heavily inherits from the BertTokenizer
    implementation found in Huggingface's transformers library. It runs a WordPiece tokenization
    algorithm over SMILES strings using the tokenisation SMILES regex developed by Schwaller et. al.

    Please see https://github.com/huggingface/transformers
    and https://github.com/rxn4chemistry/rxnfp for more details.
    """

    def __init__(self, vocab_file: str = "", **kwargs):
        super().__init__(vocab_file, **kwargs)

        if not os.path.isfile(vocab_file):
            raise ValueError("Can't find a vocab file at path '{}'.".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.basic_tokenizer = BasicSmilesTokenizer()

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def vocab_list(self):
        return list(self.vocab.keys())

    def _tokenize(self, text: str, **kwargs):
        """Tokenize a SMILES string using regex without silent truncation."""
        return self.basic_tokenizer.tokenize(text)

    def _convert_token_to_id(self, token: str):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index: int):
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]):
        # For SMILES, tokens should be concatenated without spaces.
        return "".join(tokens)

    def add_special_tokens_ids_single_sequence(self, token_ids: List[Optional[int]]):
        return [self.cls_token_id] + token_ids + [self.sep_token_id]

    def add_special_tokens_single_sequence(self, tokens: List[str]):
        return [self.cls_token] + tokens + [self.sep_token]

    def add_special_tokens_ids_sequence_pair(
        self, token_ids_0: List[Optional[int]], token_ids_1: List[Optional[int]]
    ) -> List[Optional[int]]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def add_padding_tokens(
        self, token_ids: List[Optional[int]], length: int, right: bool = True
    ) -> List[Optional[int]]:
        padding = [self.pad_token_id] * (length - len(token_ids))
        return token_ids + padding if right else padding + token_ids


class BasicSmilesTokenizer:
    def __init__(self, regex_pattern: str = SMI_REGEX_PATTERN):
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)

    def tokenize(self, text):
        return [token for token in self.regex.findall(text)]


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab

