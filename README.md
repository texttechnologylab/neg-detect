# neg-detect

`neg-detect` is a Python package for detecting negation cues and their scopes in text using fine-tuned BERT models. It provides a pipeline to process batched text inputs, identify negation cues (e.g., "not", "n't"), and determine the scope of negation within sentences. The package leverages the Hugging Face Transformers library and PyTorch for efficient inference.

## Features
- **Negation Cue Detection**: Identifies negation cues (e.g., "not", "n't") using the `CueBertInference` class.
- **Negation Scope Detection**: Determines the scope of negation in text using the `ScopeBertInference` class.
- **Pipeline Processing**: Combines cue and scope detection in a single pipeline for streamlined processing.
- **Batch Processing**: Supports batched inputs for efficient inference.
- **GPU Support**: Utilizes CUDA for accelerated inference on compatible hardware.
- **TODO**: In the future there will be negation event and focus detection components added to the Pipeline.

## Installation

### Prerequisites
- Python 3.6 or higher
- PyTorch
- Hugging Face Transformers
- CUDA-enabled GPU (optional, for faster inference)

### Install via PyPI
```bash
pip install neg-detect
```

### Install Dependencies
Ensure dependencies are installed:
```bash
pip install torch transformers
```

## Usage

### Basic Example
The following example demonstrates how to use the `Pipeline` class to detect negation cues and scopes in a batch of sentences.

```python
from neg_detect import Pipeline

# Define input sentences
batch_tokens = [
    "Your sample input does n't go here .".split(" "),
    "This is not another test sentence .".split(" ")
]

# Initialize pipeline with default models
pipe = Pipeline()

# Run inference
results = pipe.run(batch_tokens)

# Pretty print results
Pipeline.pretty_print(results)
```

**Output**:
```
Your            X
sample          X
input           X
does            X
n't             C
go              S
here            S
,               X
i               X
live            X
in              X
Germany         X
.               X

This            X
is              X
not             C
another         S
test            S
sentence        S
.               X
```

### Advanced Usage
For custom models or tokenizers, you can initialize the pipeline with specific components:

```python
from neg_detect import Pipeline, CueBertInference, ScopeBertInference

# Load custom models and tokenizers
mcue_path = "Lelon/8449368577"
mscope_path = "Lelon/5556020097"
model_cue, tokenizer_cue = CueBertInference.load_model_and_tokenizer(mcue_path, mcue_path)
model_scope, tokenizer_scope = ScopeBertInference.load_model_and_tokenizer(mscope_path, mscope_path)

# Initialize pipeline with custom components
pipe = Pipeline(
    components=[CueBertInference, ScopeBertInference],
    models=[model_cue, model_scope],
    tokenizers=[tokenizer_cue, tokenizer_scope]
)

# Define input
batch_tokens = [
    "This is not another test sentence .".split(" ")
]

# Run inference
results = pipe.run(batch_tokens, device="cuda:0", max_length=128)

# Print results
Pipeline.pretty_print(results)
```

## Package Structure
- **`CueBertInference`**: Detects negation cues (labeled as "C" for cues, "X" otherwise).
- **`ScopeBertInference`**: Identifies the scope of negation (labeled as "S" for scope, "X" otherwise).
- **`Pipeline`**: Combines `CueBertInference` and `ScopeBertInference` for end-to-end negation detection.
- **Special Tokens**:
  - `[CUE]`: Marks negation cues.
  - `[SCO]`: Marks negation scope.

## Requirements
See `requirements.txt` for a full list of dependencies. Key dependencies include:
- `torch>=1.9.0`
- `transformers>=4.9.0`

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please open an issue on the [GitHub repository](https://github.com/LeonHammerla/neg-detect).