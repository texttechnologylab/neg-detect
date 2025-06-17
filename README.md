# neg-detect

[![PyPI version](https://img.shields.io/pypi/v/neg-detect.svg)](https://pypi.org/project/neg-detect/)
[![PyPI downloads](https://img.shields.io/pypi/dm/neg-detect.svg)](https://pypi.org/project/neg-detect/)
[![PyPI Python versions](https://img.shields.io/pypi/pyversions/neg-detect.svg)](https://pypi.org/project/neg-detect/)

`neg-detect` is a Python package for detecting negation cues and their scopes in text using fine-tuned BERT models. It provides a pipeline to process batched text inputs, identify negation cues (e.g., "not", "n't"), and determine the scope of negation within sentences. The package leverages the Hugging Face Transformers library, PyTorch-Geometric, and PyTorch for efficient inference.

## Features
- **Negation Cue Detection**: Identifies negation cues (e.g., "not", "n't") using the `CueBertInference` class.
- **Negation Scope Detection**: Determines the scope of negation in text using the `ScopeBertInference` class.
- **Pipeline Processing**: Combines cue and scope detection in a single pipeline for streamlined processing.
- **Batch Processing**: Supports batched inputs for efficient inference.
- **GPU Support**: Utilizes CUDA for accelerated inference on compatible hardware.
- **TODO**: In the future there will be negation event and focus detection components added to the Pipeline.
- 🌟✴️🌟 **German Language Support**: The pipeline now supports negation detection in **German** as well as English.
- 🌟✴️🌟 **Multi Language Support**: The pipeline now supports negation detection for 10 additional Languages: **German**, **Italian**, **Spanish**, **French**, **Dutch**, **Chinese**, **Japanese**, **Russian**, **Hindi**, **Arabic**
- We trained around 300 models, which can all accessed via this package (and can be found on huggingface)

### Prerequisites
- Python 3.6 or higher
- PyTorch 
- PyTorch Geometric
- Scikit-Learn
- UD-Pipe
- Spacy 
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
from neg_detect import PipelineTests, Pipeline
pipe = Pipeline.from_language()
batch_tokens = [
    "This is not an example for testing, it is also not an example for multi negation testing and i never ate spinach .".split(
        " "),
    ['In', 'contrast', 'to', 'anti-CD3/IL-2-activated', 'LN', 'cells', ',', 'adoptive', 'transfer', 'of',
     'freshly', 'isolated', 'tumor-draining', 'LN', 'T', 'cells', 'has', 'no', 'therapeutic', 'activity',
     '.'],
]
res = pipe.run(batch_tokens)
Pipeline.pretty_print(res)

```
```
Results in:

This                           S     X     X    
is                             S     X     X    
not                            C     X     X    
an                             S     X     X    
example                        X     X     X    
for                            S     X     X    
testing,                       X     X     X    
it                             X     S     X    
is                             X     S     X    
also                           X     S     X    
not                            X     C     X    
an                             X     S     X    
example                        X     X     X    
for                            X     S     X    
multi                          X     S     X    
negation                       X     X     X    
testing                        X     X     X    
and                            X     X     X    
i                              X     S     S    
never                          X     X     C    
ate                            X     X     S    
spinach                        X     X     S    
.                              X     X     X    

In                             X    
contrast                       X    
to                             X    
anti-CD3/IL-2-activated        X    
LN                             X    
cells                          X    
,                              X    
adoptive                       X    
transfer                       X    
of                             X    
freshly                        X    
isolated                       X    
tumor-draining                 X    
LN                             X    
T                              X    
cells                          X    
has                            X    
no                             C    
therapeutic                    X    
activity                       X    
.                              X  
```

### Advanced Usage
For custom models or tokenizers, you can initialize the pipeline with specific components:

```python
from neg_detect import Pipeline, CueBertInference, ScopeBertInference

# Load custom models and tokenizers
mcue_path = "Lelon/cue-de-sfu"
mscope_path = "Lelon/scope-de-sfu"

# Initialize pipeline with custom components
pipe = Pipeline(
    components=[CueBertInference, ScopeBertInference],
    model_paths=[mcue_path, mscope_path]
)

# Define input
batch_tokens = [
    "Das ist nicht ein Testsatz .".split(" ")
]

# Run inference
results = pipe.run(batch_tokens)

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
