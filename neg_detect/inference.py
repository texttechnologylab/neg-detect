import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, PreTrainedTokenizer
from typing import List, Dict, Any, Tuple, Optional
import os


BP = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../.."))


class NegBertInference:
    special_tokens = ...

    @staticmethod
    def load_model_and_tokenizer(model_path: str, bert_model: str) -> Tuple[Any, Any]:
        """
        Load the fine-tuned model and tokenizer.
        :param model_path:
        :param bert_model:
        :return:
        """
        tokenizer = AutoTokenizer.from_pretrained(bert_model)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        return model, tokenizer

    @staticmethod
    def preprocess_input(tokens: List[List[str]], tokenizer: PreTrainedTokenizer, max_length: int = 128) -> Tuple[Any, Any]:
        """
        Tokenize batched pre-split input tokens and return word IDs for merging subtokens.
        :param tokens:
        :param tokenizer:
        :param max_length:
        :return:
        """
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
    def predict(model: AutoModelForTokenClassification,
                tokenizer: AutoTokenizer,
                input_tokens,
                tokenized_inputs,
                word_ids: List[int],
                max_length: int = 128,
                device: str = "cuda"):
        ...

class CueBertInference(NegBertInference):
    special_tokens = {"C": "[CUE]"}

    @staticmethod
    def majority_label(labels: List[str]) -> str:
        """
        Return the majority label from a list of labels ('X' or 'C').
        :param labels:
        :return:
        """
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
        """
        Perform prediction on batched inputs, merging subtoken predictions by majority vote.
        :param model:
        :param tokenizer:
        :param input_tokens:
        :param tokenized_inputs:
        :param word_ids:
        :param max_length:
        :param device:
        :return:
        """
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

    @staticmethod
    def main(model_path: str, tok_path: str, device: str = "cuda:0", max_length: int = 128) -> None:
        # Example batched input
        batch_tokens = [
            "Your sample input doesn't go here , i live in the prestreetlondon .".split(" "),
            "This is another test sentence .".split(" ")
        ]

        # Load model and tokenizer
        model, tokenizer = CueBertInference.load_model_and_tokenizer(model_path, tok_path)
        model = model.to(device)
        model.eval()  # Set model to evaluation mode

        # Preprocess input
        tokenized_inputs, word_ids = CueBertInference.preprocess_input(batch_tokens, tokenizer, max_length)

        # Perform inference
        batch_predictions = CueBertInference.predict(model, tokenizer, batch_tokens, tokenized_inputs, word_ids, max_length, device)

        # Print results
        print("Inference Results:")
        for seq_idx, predictions in enumerate(batch_predictions):
            print(f"\nSequence {seq_idx + 1}:")
            for result in predictions:
                print(f"Token: {result['token']}\nLabel: {result['label']}")


class ScopeBertInference(NegBertInference):
    special_tokens = {"S": "[SCO]"}

    @staticmethod
    def majority_label(labels: List[str]) -> str:
        """
        Return the majority label from a list of labels ('S' or 'C').
        :param labels:
        :return:
        """
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
        """
        Perform inference on batched inputs, merging subtoken predictions by majority vote.
        :param model:
        :param tokenizer:
        :param input_tokens:
        :param tokenized_inputs:
        :param word_ids:
        :param max_length:
        :param device:
        :return:
        """
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

    @staticmethod
    def main(model_path: str, tok_path: str, device: str = "cuda:0", max_length: int = 128) -> None:
        # Example batched input
        batch_tokens = [
            "Your sample input does [CUE] n't go here , i live in the prestreetlondon .".split(" "),
            "This is not another test sentence .".split(" ")
        ]

        # Load model and tokenizer
        model, tokenizer = ScopeBertInference.load_model_and_tokenizer(model_path, tok_path)
        model = model.to(device)
        model.eval()  # Set model to evaluation mode

        # Preprocess input
        tokenized_inputs, word_ids = ScopeBertInference.preprocess_input(batch_tokens, tokenizer, max_length)

        # Perform inference
        batch_predictions = ScopeBertInference.predict(model, tokenizer, batch_tokens, tokenized_inputs, word_ids, max_length, device)

        # Print results
        print("Inference Results:")
        for seq_idx, predictions in enumerate(batch_predictions):
            print(f"\nSequence {seq_idx + 1}:")
            for result in predictions:
                print(f"Token: {result['token']}\nLabel: {result['label']}")


class Pipeline:
    def __init__(self, components: Optional[List[NegBertInference]] = None, models: Optional[List[Any]] = None, tokenizers: Optional[List[Any]] = None):
        comp_dict = {CueBertInference: "Lelon/8449368577", ScopeBertInference: "Lelon/5556020097"}
        if components is None:
            components = [CueBertInference, ScopeBertInference]
            print("No components provided, using default components (cue + scope).")
        else:
            print("Using provided components.")

        self.components = components
        if models is None or tokenizers is None:
            models, tokenizers = [], []
            print("No models or tokenizers provided, using default models and tokenizers.")
            for component in components:
                model, tokenizer = component.load_model_and_tokenizer(comp_dict[component], comp_dict[component])
                models.append(model)
                tokenizers.append(tokenizer)
                print(f"Loaded {component.__name__} model and tokenizer.")
        else:
            print("Using provided models and tokenizers.")
            self.models = models
            self.tokenizers = tokenizers
        self.special_tokens = {value[1]:value[0] for comp in self.components for value in comp.special_tokens.items()}
        assert len(components) == len(models) == len(tokenizers), "Provide the same number of components, models, and tokenizers."

    def run(self, batch_tokens: List[List[str]], device: str = "cuda:0", max_length: int = 128) -> List[Tuple[List[str], List[str]]]:

        for idx, component in enumerate(self.components):
            tokenizer = self.tokenizers[idx]
            model = self.models[idx]
            model = model.to(device)
            model.eval()
            tokenized_inputs, word_ids = component.preprocess_input(batch_tokens,
                                                                    tokenizer,
                                                                    max_length)
            batch_predictions = component.predict(model,
                                                  tokenizer,
                                                  batch_tokens,
                                                  tokenized_inputs,
                                                  word_ids,
                                                  max_length,
                                                  device)
            batch_tokens = []
            for seq_idx, predictions in enumerate(batch_predictions):
                batch_seq = []
                for result in predictions:
                    if result['label'] in component.special_tokens:
                        batch_seq.append(component.special_tokens[result['label']])
                    batch_seq.append(result['token'])
                batch_tokens.append(batch_seq)

        result = []
        for seq_idx, predictions in enumerate(batch_tokens):
            clean_seq = []
            labels = []
            next_label = "X"
            for token in predictions:
                if token in self.special_tokens:
                    next_label = self.special_tokens[token]
                else:
                    clean_seq.append(token)
                    labels.append(next_label)
                    next_label = "X"

            """print()
            for item1, item2 in zip(clean_seq, labels):
                print(f"{str(item1):<{15}} {str(item2):<{15}}")
            print(predictions)"""
            result.append((clean_seq, labels))

        return result

    @staticmethod
    def pretty_print(result: List[Tuple[List[str], List[str]]]) -> None:
        for res in result:
            for item1, item2 in zip(res[0], res[1]):
                print(f"{str(item1):<{15}} {str(item2):<{15}}")
            print()

    @staticmethod
    def main():
        mcue_path = "Lelon/8449368577"
        mscope_path = "Lelon/5556020097"
        model_cue, tokenizer_cue = CueBertInference.load_model_and_tokenizer(mcue_path, mcue_path)
        model_scope, tokenizer_scope = ScopeBertInference.load_model_and_tokenizer(mscope_path, mscope_path)
        pipe = Pipeline(components=[CueBertInference, ScopeBertInference],
                        models=[model_cue, model_scope],
                        tokenizers=[tokenizer_cue, tokenizer_scope])

        batch_tokens = [
            "Your sample input does n't go here , i live in the prestreetlondon .".split(" "),
            "This is not another test sentence .".split(" "),
            ['Second', ',', 'T', 'cells', ',', 'which', 'lack', 'CD45', 'and', 'can', 'not', 'signal', 'via', 'the', 'TCR', ',', 'supported', 'higher', 'levels', 'of', 'viral', 'replication', 'and', 'gene', 'expression', '.'],
            ['Our', 'results', 'indicate', 'that', 'I', 'kappa', 'b', 'beta', ',', 'but', '[CUE]', 'not', 'I', 'kappa', 'B', 'alpha', ',', 'is', 'required', 'for', 'the', 'signal', '-', 'dependent', 'activation', 'of', 'NF', '-', 'kappa', 'B', 'in', 'fibroblasts', '.']
        ]

        Pipeline.pretty_print(pipe.run(batch_tokens))

if __name__ == "__main__":
    Pipeline.main()
