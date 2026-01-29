"""
Compatible with old checkpoints:
Historically, the module was named `my_tokenizer`, but it was later renamed to `tokenizer`.
When deserializing old checkpoints, the system imports using the old module path; therefore, this module is retained to serve as an "alias."
"""

from tokenizer import BasicSmilesTokenizer, SMI_REGEX_PATTERN, SmilesTokenizer, load_vocab

__all__ = ["SmilesTokenizer", "BasicSmilesTokenizer", "SMI_REGEX_PATTERN", "load_vocab"]
