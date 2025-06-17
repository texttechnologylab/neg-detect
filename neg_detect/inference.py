import copy
import json

from huggingface_hub import hf_hub_download
import spacy
from safetensors.torch import load_file
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, PreTrainedTokenizer, \
    DataCollatorForTokenClassification, PreTrainedTokenizerBase
from typing import List, Dict, Any, Tuple, Optional, Literal
import numpy as np
import os

from .preprocessing import PreprocessorUtility
from .spacy_utils import process_sent_spacy, upos_dict, dep_dict, get_spacy_model
from .gat_model import BERTResidualGATv2ContextGatedFusion
from .pipeline_utils import split_sequence as splitter


BP = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../.."))


class BasicInference:
    special_tokens = ...

    @staticmethod
    def majority_label(labels: List[str]) -> str:
        ...

    @classmethod
    def init_component(cls, model_path: str, device: Any, max_len: Any, **kwargs):
        ...

    def run(self, batch_tokens: List[List[str]], original_input: Optional[List[List[str]]] = None):
        ...

    @staticmethod
    def pretty_print(result: List[List[Dict[str, str]]]) -> None:
        for sent in result:
            for tok in sent:
                print(f"{str(tok['token']):<{15}} {str(tok['label']):<{15}}")
            print("\n")


class NegBertInference(BasicInference):

    @staticmethod
    def load_model_and_tokenizer(model_path: str, bert_model: str = "prajjwal1/bert-tiny") -> tuple:
        """Load the fine-tuned model and tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(bert_model, trust_remote_code=True)
        model = AutoModelForTokenClassification.from_pretrained(model_path, trust_remote_code=True)
        return model, tokenizer

    @staticmethod
    def preprocess_input(tokens: List[List[str]], tokenizer: PreTrainedTokenizer, max_length: int = 128) -> tuple:
        """Tokenize batched pre-split input tokens and return word IDs for merging subtokens."""
        # Tokenize with is_split_into_words=True to match training
        tokenized_inputs = tokenizer(
            tokens,
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"  # Return PyTorch tensors for inference
        )
        # Get word IDs for each sequence in the batch
        word_ids = [tokenized_inputs.word_ids(batch_index=i) for i in range(len(tokens))]
        return tokenized_inputs, word_ids

    @staticmethod
    def majority_label(labels: List[str]) -> str:
        ...

    @staticmethod
    def predict(model: torch.nn.Module,
                tokenizer: AutoTokenizer,
                input_tokens,
                tokenized_inputs,
                word_ids: List[int],
                max_length: int = 128,
                device: str = "cuda"):
        ...



class NegBertInferenceGAT(BasicInference):

    @staticmethod
    def load_model_and_tokenizer(model_path: str,
                                 model_architecture: Any = BERTResidualGATv2ContextGatedFusion) -> tuple:
        """Load the fine-tuned model and tokenizer."""
        if os.path.isdir(model_path):
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            with open(f"{model_path}/config.json", "r") as f:
                config = json.load(f)
            model = model_architecture(
                bert_id=config.get("bert_id"),
                id2label=config.get("id2label"),
                label2id=config.get("label2id"),
                num_labels=config.get("lcount"),
                pos_vocab_size=config.get("pos_vocab_size"),  # Update as needed
                dep_vocab_size=config.get("dep_vocab_size")  # Update as needed
            )
            if config.get("vocab_size") != model.bert.config.vocab_size:
                model.bert.resize_token_embeddings(config.get("vocab_size"))
            # 3. Load the safetensors file
            state_dict = load_file(f"{model_path}/model.safetensors")

            # 4. Load the weights into your model
            missing, unexpected = model.load_state_dict(state_dict, strict=True)

            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            weights_path = hf_hub_download(repo_id=model_path, filename="model.safetensors")
            config_path = hf_hub_download(repo_id=model_path, filename="config.json")
            with open(config_path, "r") as f:
                config = json.load(f)
            model = model_architecture(
                bert_id=config.get("bert_id"),
                id2label=config.get("id2label"),
                label2id=config.get("label2id"),
                num_labels=config.get("lcount"),
                pos_vocab_size=config.get("pos_vocab_size"),  # Update as needed
                dep_vocab_size=config.get("dep_vocab_size")  # Update as needed
            )
            if config.get("vocab_size") != model.bert.config.vocab_size:
                model.bert.resize_token_embeddings(config.get("vocab_size"))
            state_dict = load_file(weights_path)
            # 4. Load the weights into your model
            missing, unexpected = model.load_state_dict(state_dict, strict=True)

            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        max_length = config.get("max_length")

        spacy_model = get_spacy_model(lang=config.get("spacy_id").split("_")[0])
        print("spacymodel:", spacy_model, "lang", config.get("spacy_id").split("_")[0])

        return model, tokenizer, spacy_model, max_length, config.get("id2label")


    @staticmethod
    def preprocess_input(tokens: List[List[str]],
                         original_tokens: List[List[str]],
                         tokenizer: PreTrainedTokenizer,
                         spacy_model: Any,
                         max_length: int = 128):
        pos_tags = []
        dep_tags = []
        edge_indices = []
        for sent in original_tokens:
            upos, deps, edge_index = process_sent_spacy(sent, spacy_model)
            pos_tags.append(upos)
            dep_tags.append(deps)
            edge_indices.append(edge_index)

        result = PreprocessorUtility.retokenize_with_pos(tokens=tokens,
                                                         pos_tags=pos_tags,
                                                         dep_tags=dep_tags,
                                                         pos_tag_to_id=upos_dict,
                                                         dep_tag_to_id=dep_dict[
                                                             f'{spacy_model.meta.get("lang", "xx")}_{spacy_model.meta.get("name", "unknown")}'],
                                                         tokenizer=tokenizer,
                                                         max_seq_length=max_length
                                                         )

        # result["event_labels"] = examples["event_masks"]
        # result["focus_labels"] = examples["focus_masks"]
        result["word_count"] = [len(sent) for sent in tokens]
        for key in result:
            try:
                result[key] = torch.tensor(result[key])
            except:
                pass
        result["edge_index"] = [torch.tensor(edge_index, dtype=torch.int64) for edge_index in edge_indices]

        return result

    @staticmethod
    def majority_label(labels: List[str]) -> str:
        ...

    @staticmethod
    def predict(model: AutoModelForTokenClassification,
                tokenizer: AutoTokenizer,
                input_tokens,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                pos_ids: torch.Tensor,
                dep_ids: torch.Tensor,
                edge_index: List[torch.Tensor],
                word_ids: List[int],
                word_count: List[int],
                id2label: dict,
                max_length: int = 128,
                device: str = "cuda"):
        ...


class CueBertInference(NegBertInference):
    special_tokens = {"C": "[CUE]"}

    def __init__(self,
                 model: Optional[Any] = None,
                 tokenizer: Optional[Any] = None,
                 max_length: Optional[int] = None,
                 device: Optional[str] = None
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

        self.model.to(device)
        self.model.eval()

    @classmethod
    def init_component(cls, model_path: str, device: Any, max_len: Optional[int] = None, **kwargs):
        return cls(*CueBertInference.load_model_and_tokenizer(model_path, model_path), device=device, max_length=max_len)

    def run(self,
            batch_tokens: List[List[str]], original_input: Optional[List[List[str]]] = None):
        # Preprocess input
        tokenized_inputs, word_ids = CueBertInference.preprocess_input(batch_tokens, self.tokenizer, self.max_length)

        # Perform inference
        batch_predictions = CueBertInference.predict(self.model,
                                                     self.tokenizer,
                                                     batch_tokens,
                                                     tokenized_inputs,
                                                     word_ids,
                                                     self.max_length,
                                                     self.device
                                                     )
        return batch_predictions

    @staticmethod
    def majority_label(labels: List[str]) -> str:
        """Return the majority label from a list of labels ('X' or 'C')."""
        if not labels:
            return "X"  # Default to "X" if no labels
        count_c = labels.count("C")
        count_x = labels.count("X")
        return "C" if count_c >= count_x else "X"

    @staticmethod
    def predict(model: AutoModelForTokenClassification,
                tokenizer: AutoTokenizer,
                input_tokens: List[List[str]],
                tokenized_inputs,
                word_ids: List[List[int]],
                max_length: int = 128,
                device: str = "cuda:0") -> List[List[Dict]]:
        """Perform inference on batched inputs, merging subtoken predictions by majority vote."""
        # Move inputs to the same device as the model
        tokenized_inputs = {key: val.to(device) for key, val in tokenized_inputs.items()}

        # Perform inference
        with torch.no_grad():
            outputs = model(**tokenized_inputs)
            logits = outputs.logits  # Shape: (batch_size, sequence_length, num_labels)

        # Convert logits to predictions
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()  # Shape: (batch_size, sequence_length)

        # Process each sequence in the batch
        batch_results = []
        for seq_idx, (seq_tokens, seq_word_ids, seq_predictions) in enumerate(zip(input_tokens, word_ids, predictions)):
            pred_labels = [model.config.id2label[pred] for pred in seq_predictions]

            # Merge subtoken predictions to original tokens
            results = []
            current_word_id = None
            current_subtoken_labels = []
            for token_idx, (word_id, label) in enumerate(zip(seq_word_ids, pred_labels)):
                if word_id is None:  # Skip special tokens ([CLS], [SEP], [PAD])
                    continue
                if word_id != current_word_id:
                    # Process previous word
                    if current_subtoken_labels and current_word_id is not None:
                        merged_label = CueBertInference.majority_label(current_subtoken_labels)
                        original_token = seq_tokens[current_word_id]
                        results.append({"token": original_token, "label": merged_label})
                    # Start new word
                    current_word_id = word_id
                    current_subtoken_labels = [label]
                else:
                    # Add to current word's subtokens
                    current_subtoken_labels.append(label)

            # Process the last word
            if current_subtoken_labels and current_word_id is not None:
                merged_label = CueBertInference.majority_label(current_subtoken_labels)
                original_token = seq_tokens[current_word_id]
                results.append({"token": original_token, "label": merged_label})

            batch_results.append(results)

        return batch_results



class ScopeBertInference(NegBertInference):
    special_tokens = {"S": "[SCO]"}

    def __init__(self,
                 model: Optional[Any] = None,
                 tokenizer: Optional[Any] = None,
                 max_length: Optional[int] = None,
                 device: Optional[str] = None
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

        self.model.to(device)
        self.model.eval()

    @classmethod
    def init_component(cls, model_path: str, device: Any, max_len: Optional[int] = None, **kwargs):
        return cls(*ScopeBertInference.load_model_and_tokenizer(model_path, model_path), device=device, max_length=max_len)

    def run(self,
            batch_tokens: List[List[str]], original_input: Optional[List[List[str]]] = None):
        # Preprocess input
        tokenized_inputs, word_ids = ScopeBertInference.preprocess_input(batch_tokens, self.tokenizer, self.max_length)

        # Perform inference
        batch_predictions = ScopeBertInference.predict(self.model,
                                                     self.tokenizer,
                                                     batch_tokens,
                                                     tokenized_inputs,
                                                     word_ids,
                                                     self.max_length,
                                                     self.device
                                                     )
        return batch_predictions

    @staticmethod
    def majority_label(labels: List[str]) -> str:
        """Return the majority label from a list of labels ('X' or 'C')."""
        if not labels:
            return "X"  # Default to "X" if no labels
        count_c = labels.count("S")
        count_x = labels.count("X")
        return "S" if count_c >= count_x else "X"

    @staticmethod
    def predict(model: AutoModelForTokenClassification,
                tokenizer: AutoTokenizer,
                input_tokens: List[List[str]],
                tokenized_inputs,
                word_ids: List[List[int]],
                max_length: int = 128,
                device: str = "cuda:0") -> List[List[Dict]]:
        """Perform inference on batched inputs, merging subtoken predictions by majority vote."""
        # Move inputs to the same device as the model
        tokenized_inputs = {key: val.to(device) for key, val in tokenized_inputs.items()}

        # Perform inference
        with torch.no_grad():
            outputs = model(**tokenized_inputs)
            logits = outputs.logits  # Shape: (batch_size, sequence_length, num_labels)

        # Convert logits to predictions
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()  # Shape: (batch_size, sequence_length)
        # Process each sequence in the batch
        batch_results = []
        for seq_idx, (seq_tokens, seq_word_ids, seq_predictions) in enumerate(zip(input_tokens, word_ids, predictions)):
            pred_labels = [model.config.id2label[pred] for pred in seq_predictions]

            # Merge subtoken predictions to original tokens
            results = []
            current_word_id = None
            current_subtoken_labels = []
            for token_idx, (word_id, label) in enumerate(zip(seq_word_ids, pred_labels)):
                if word_id is None:  # Skip special tokens ([CLS], [SEP], [PAD])
                    continue
                if word_id != current_word_id:
                    # Process previous word
                    if current_subtoken_labels and current_word_id is not None:
                        merged_label = ScopeBertInference.majority_label(current_subtoken_labels)
                        original_token = seq_tokens[current_word_id]
                        results.append({"token": original_token, "label": merged_label})
                    # Start new word
                    current_word_id = word_id
                    current_subtoken_labels = [label]
                else:
                    # Add to current word's subtokens
                    current_subtoken_labels.append(label)

            # Process the last word
            if current_subtoken_labels and current_word_id is not None:
                merged_label = ScopeBertInference.majority_label(current_subtoken_labels)
                original_token = seq_tokens[current_word_id]
                results.append({"token": original_token, "label": merged_label})

            batch_results.append(results)

        return batch_results



class CueBertInferenceGAT(NegBertInferenceGAT):
    special_tokens = {"C": "[CUE]"}

    def __init__(self,
                 model: Optional[Any] = None,
                 tokenizer: Optional[Any] = None,
                 spacy_model: Optional[Any] = None,
                 max_length: Optional[int] = None,
                 id2label: Optional[dict] = None,
                 device: Optional[str] = None
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.spacy_model = spacy_model
        self.max_length = max_length
        self.id2label = id2label
        self.device = device

        self.model.to(device)
        self.model.eval()

    @classmethod
    def init_component(cls,
                       model_path: str,
                       device: Any,
                       max_len: Optional[int] = None,
                       model_architecture: Any = BERTResidualGATv2ContextGatedFusion,
                       **kwargs):
        return cls(*CueBertInferenceGAT.load_model_and_tokenizer(model_path, model_architecture), device=device)

    def run(self, batch_tokens: List[List[str]], original_input: Optional[List[List[str]]] = None):
        # Preprocess input
        inputs = CueBertInferenceGAT.preprocess_input(batch_tokens, original_input, self.tokenizer, self.spacy_model, self.max_length)

        # Perform inference
        batch_predictions = CueBertInferenceGAT.predict(self.model,
                                                        self.tokenizer,
                                                        batch_tokens,
                                                        inputs["input_ids"],
                                                        inputs["attention_mask"],
                                                        inputs["pos_ids"],
                                                        inputs["dep_ids"],
                                                        inputs["edge_index"],
                                                        inputs["word_ids"],
                                                        inputs["word_count"],
                                                        self.id2label,
                                                        self.max_length,
                                                        self.device)
        return batch_predictions

    @staticmethod
    def majority_label(labels: List[str]) -> str:
        """Return the majority label from a list of labels ('X' or 'C')."""
        if not labels:
            return "X"  # Default to "X" if no labels
        count_c = labels.count("C")
        count_x = labels.count("X")
        return "C" if count_c >= count_x else "X"

    @staticmethod
    def predict(model: torch.nn.Module,
                tokenizer: AutoTokenizer,
                input_tokens,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                pos_ids: torch.Tensor,
                dep_ids: torch.Tensor,
                edge_index: List[torch.Tensor],
                word_ids: List[int],
                word_count: List[int],
                id2label: dict,
                max_length: int = 128,
                device: str = "cuda:0") -> List[List[Dict]]:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        pos_ids = pos_ids.to(device)
        dep_ids = dep_ids.to(device)
        for idx, edge in enumerate(edge_index):
            edge_index[idx] = edge.to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pos_ids=pos_ids,
                            dep_ids=dep_ids,
                            edge_index=edge_index,
                            word_ids=word_ids,
                            word_count=word_count,
                            )
            logits = outputs.logits  # Shape: (batch_size, sequence_length, num_labels)

        # Convert logits to predictions
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()  # Shape: (batch_size, sequence_length)

        # Process each sequence in the batch
        batch_results = []
        idx = 0
        for sent in input_tokens:
            # Merge subtoken predictions to original tokens
            results = []
            for token in sent:
                results.append({"token": token, "label": id2label[f"{int(predictions[idx])}"]})
                idx += 1
            batch_results.append(results)

        return batch_results



class ScopeBertInferenceGAT(NegBertInferenceGAT):
    special_tokens = {"S": "[SCO]"}

    def __init__(self,
                 model: Optional[Any] = None,
                 tokenizer: Optional[Any] = None,
                 spacy_model: Optional[Any] = None,
                 max_length: Optional[int] = None,
                 id2label: Optional[dict] = None,
                 device: Optional[str] = None
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.spacy_model = spacy_model
        self.max_length = max_length
        self.id2label = id2label
        self.device = device
        self.model.eval()
        self.model.to(device)


    @classmethod
    def init_component(cls,
                       model_path: str,
                       device: Any,
                       max_len: Optional[int] = None,
                       model_architecture: Any = BERTResidualGATv2ContextGatedFusion,
                       **kwargs):
        return cls(*ScopeBertInferenceGAT.load_model_and_tokenizer(model_path, model_architecture), device=device)

    def run(self, batch_tokens: List[List[str]], original_input: Optional[List[List[str]]] = None):
        # Preprocess input
        inputs = ScopeBertInferenceGAT.preprocess_input(batch_tokens, original_input, self.tokenizer, self.spacy_model, self.max_length)
        # print(batch_tokens)
        # Perform inference
        batch_predictions = ScopeBertInferenceGAT.predict(model=self.model,
                                                        tokenizer=self.tokenizer,
                                                        input_tokens=batch_tokens,
                                                        input_ids=inputs["input_ids"],
                                                        attention_mask=inputs["attention_mask"],
                                                        pos_ids=inputs["pos_ids"],
                                                        dep_ids=inputs["dep_ids"],
                                                        edge_index=inputs["edge_index"],
                                                        word_ids=inputs["word_ids"],
                                                        word_count=inputs["word_count"],
                                                        id2label=self.id2label,
                                                        max_length=self.max_length,
                                                        device=self.device)
        return batch_predictions

    @staticmethod
    def predict(model: torch.nn.Module,
                tokenizer: AutoTokenizer,
                input_tokens,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                pos_ids: torch.Tensor,
                dep_ids: torch.Tensor,
                edge_index: List[torch.Tensor],
                word_ids: List[int],
                word_count: List[int],
                id2label: dict,
                max_length: int = 128,
                device: str = "cuda:0") -> List[List[Dict]]:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        pos_ids = pos_ids.to(device)
        dep_ids = dep_ids.to(device)
        for idx, edge in enumerate(edge_index):
            edge_index[idx] = edge.to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pos_ids=pos_ids,
                            dep_ids=dep_ids,
                            edge_index=edge_index,
                            word_ids=word_ids,
                            word_count=word_count,
                            )
            logits = outputs.logits  # Shape: (batch_size, sequence_length, num_labels)

        # Convert logits to predictions
        predictions = torch.argmax(logits, dim=1).cpu().numpy()  # Shape: (batch_size, sequence_length)
        # Process each sequence in the batch
        batch_results = []
        idx = 0
        for sent in input_tokens:
            # Merge subtoken predictions to original tokens
            results = []
            for token in sent:
                results.append({"token": token, "label": id2label[f"{int(predictions[idx])}"]})
                idx += 1
            batch_results.append(results)

        return batch_results


class Pipeline:
    def __init__(self,
                 components: List[BasicInference],
                 model_paths: List[str],
                 device: str = "cuda:0",
                 max_length: int = 128,
                 model_architecture: Any = BERTResidualGATv2ContextGatedFusion):
        self.components = []
        for component, model_path in zip(components, model_paths):
            self.components.append(component.init_component(model_path=model_path, device=device, max_len=max_length, model_architecture=model_architecture))

        self.special_tokens = {value[1]:value[0] for comp in self.components for value in comp.special_tokens.items()}

    @classmethod
    def from_language(cls,
                      lang: Literal["en", "de", "hi", "zh", "jap", "ru", "nl", "es", "fr", "it", "ar"] = "en",
                      mode: Literal["base", "gat"] = "base",
                      ds: Literal["conan", "sfu", "pb_foc", "bioscope_abstracts", "bioscope_full", "socc", "dt_neg"] = "conan",
                      device: str = "cuda:0",
                      max_length: int = 128):
        if mode == "base":
            components = [CueBertInference, ScopeBertInference]
            models = [f"Lelon/cue-{lang}-{ds}", f"Lelon/scope-{lang}-{ds}"]
        else:
            components = [CueBertInferenceGAT, ScopeBertInferenceGAT]
            models = [f"Lelon/cue-gat-{lang}-{ds}", f"Lelon/scope-gat-{lang}-{ds}"]

        return cls(components=components,
                   model_paths=models,
                   device=device,
                   max_length=max_length,
                   model_architecture=BERTResidualGATv2ContextGatedFusion)


    def run(self, batch_tokens: List[List[str]]) -> dict:
        base_layer = copy.deepcopy(batch_tokens)
        parts = {"cue": [[], [], []],
                 "scope": [[], [], []],
                 # "focus": (scope_layer, []),
                 # "event": (scope_layer, [])
                 }
        new_mapping = [*range(len(base_layer))]
        batch_seq = base_layer
        for idx, component in enumerate(self.components):
            part_id = list(parts.keys())[idx]
            parts[part_id][1] = copy.deepcopy(new_mapping)
            parts[part_id][0] = copy.deepcopy(batch_seq)
            batch_predictions = component.run(parts[part_id][0], [base_layer[orig] for orig in parts[part_id][1]])
            new_mapping = []
            split_sequences = []
            if part_id == "cue":
                for seq_idx, predictions in enumerate(batch_predictions):
                    for special_label in component.special_tokens:
                        new_sequences = splitter(predictions, special_label)
                        split_sequences.extend(new_sequences)
                        for seq in new_sequences:
                            new_mapping.append(parts[part_id][1][seq_idx])
            else:
                split_sequences = batch_predictions
                new_mapping = parts[part_id][1]
            batch_seq = []
            for split_sequence in split_sequences:
                target = []
                for word_idx, result in enumerate(split_sequence):
                    if result['label'] in component.special_tokens:
                        target.append(component.special_tokens[result['label']])
                    else:
                        target.append(result['token'])
                batch_seq.append(target)
            parts[part_id][2] = batch_seq

        # print(*parts.items())
        result = {i: [] for i in range(len(batch_tokens))}
        for idx, seq in enumerate(parts["scope"][2]):
            result[parts["scope"][1][idx]].append(seq)

        result["original"] = batch_tokens
        # print(*result.items(), sep="\n")
        return result

    @staticmethod
    def pretty_print(result: dict) -> None:
        mappings = {"[SCO]": "S", "[CUE]": "C"}
        original = result["original"]
        for i in range(len(original)):
            for items in zip(*[sent for sent in [original[i]] + result[i]]):
                s = f"{items[0]:<{30}}"
                for item in items[1:]:
                    item = "X" if item not in mappings else mappings[item]
                    s += f" {item:<{5}}"
                print(s)
            print()


