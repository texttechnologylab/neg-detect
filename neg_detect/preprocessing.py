from typing import Any, List, Optional, Dict
import os

import torch
from transformers import PreTrainedTokenizer


BP = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../.."))


class PreprocessorUtility:
    @staticmethod
    def retokenize_with_pos(
            tokens: List[List[str]],
            pos_tags: Optional[List[List[str]]],
            dep_tags: Optional[List[List[str]]],
            pos_tag_to_id: Optional[Dict[str, int]],
            dep_tag_to_id: Optional[Dict[str, int]],
            tokenizer: PreTrainedTokenizer,
            max_seq_length: int = 128,
            default_id: int = 0) -> dict:

        # Ensure tokens and pos_tags have the same length
        assert len(tokens) == len(pos_tags), "Tokens and POS tags must have the same length"
        batch_size = len(tokens)
        # print(100*"=")
        # Convert POS tags to indices
        pos_indices = [[pos_tag_to_id.get(tag, default_id) for tag in sub_lst] for sub_lst in pos_tags]
        dep_indices = [[dep_tag_to_id.get(tag, default_id) for tag in sub_lst] for sub_lst in dep_tags]

        # Tokenize the sentence
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,  # Input is pre-tokenized
            return_tensors=None,
            padding='max_length',
            truncation=True,
            max_length=max_seq_length,
            return_special_tokens_mask=True
        )

        input_ids = encoding['input_ids']  # Shape: [max_seq_length]
        attention_mask = encoding['attention_mask']  # Shape: [max_seq_length]

        # Get word IDs to align subwords with original tokens
        word_id_list = [encoding.word_ids(i) for i in range(len(encoding['input_ids']))]

        # Assign POS indices to subword tokens
        pos_ids = torch.full((batch_size, max_seq_length), default_id, dtype=torch.long).tolist()
        dep_ids = torch.full((batch_size, max_seq_length), default_id, dtype=torch.long).tolist()
        for j, word_ids in enumerate(word_id_list):
            for i, word_id in enumerate(word_ids):
                if word_id is not None:  # Non-special token

                    pos_ids[j][i] = pos_indices[j][word_id]
                    dep_ids[j][i] = dep_indices[j][word_id]


                # Special tokens ([CLS], [SEP], [PAD]) keep default_pos_id
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pos_ids": pos_ids,
            "dep_ids": dep_ids,
            "word_ids": word_id_list,
        }
