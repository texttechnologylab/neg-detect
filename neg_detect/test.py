import os

from .gat_model import BERTResidualGATv2ContextGatedFusion
from .inference import Pipeline, CueBertInference, ScopeBertInference, ScopeBertInferenceGAT, CueBertInferenceGAT


BP = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../.."))


class PipelineTests:
    @staticmethod
    def pipeline_english_test():
        print("English Inference Baseline")
        mcue_path = "Lelon/cue-en-conan"
        mscope_path = "Lelon/scope-en-bioscope_abstracts"
        pipe = Pipeline(components=[CueBertInference, ScopeBertInference],
                        model_paths=[mcue_path, mscope_path],
                        device="cuda:0",
                        max_length=128)

        """batch_tokens = [
            ['In', 'contrast', 'to', 'anti-CD3/IL-2-activated', 'LN', 'cells', ',', 'adoptive', 'transfer', 'of',
             'freshly', 'isolated', 'tumor-draining', 'LN', 'T', 'cells', 'has', 'no', 'therapeutic', 'activity',
             '.'],
            ['The', 'majority', 'of', 'these', 'TCC', 'exhibited', 'a', 'strongly', 'polarized', 'Th2', 'cytokine',
             'profile', ',', 'and', 'the', 'production', 'of', 'IFN-gamma', 'could', 'not', 'be', 'induced', 'by',
             'exogenous', 'IL-12', '.']
        ]"""
        batch_tokens = [
            "This is not an example for testing, it is also not an example for multi negation testing and i never ate spinach .".split(" "),
            ['In', 'contrast', 'to', 'anti-CD3/IL-2-activated', 'LN', 'cells', ',', 'adoptive', 'transfer', 'of',
             'freshly', 'isolated', 'tumor-draining', 'LN', 'T', 'cells', 'has', 'no', 'therapeutic', 'activity',
             '.'],
        ]
        res = pipe.run(batch_tokens)
        Pipeline.pretty_print(res)

        mcue_path = "Lelon/cue-gat-en-socc"
        mscope_path = "Lelon/scope-gat-en-bioscope_abstracts"
        pipe = Pipeline(components=[CueBertInferenceGAT, ScopeBertInferenceGAT],
                        model_paths=[mcue_path, mscope_path],
                        device="cuda:0",
                        max_length=128,
                        model_architecture=BERTResidualGATv2ContextGatedFusion)

        res = pipe.run(batch_tokens)
        Pipeline.pretty_print(res)

        return res

    @staticmethod
    def pipeline_german_test():
        mcue_path = "Lelon/cue-de-conan"
        mscope_path = "Lelon/scope-de-bioscope_abstracts"
        pipe = Pipeline(components=[CueBertInference, ScopeBertInference],
                        model_paths=[mcue_path, mscope_path],
                        device="cuda:0",
                        max_length=128)

        batch_tokens = [
            "Ich werde heute nicht mehr nach Hause fahren .".split(" "),
            "Ich sage dir nicht , dass du nicht nett bist , aber ich umarme dich auch nicht .".split(" ")
        ]

        res = pipe.run(batch_tokens)
        Pipeline.pretty_print(res)

        mcue_path = "Lelon/cue-gat-de-conan"
        mscope_path = "Lelon/scope-gat-de-bioscope_abstracts"
        pipe = Pipeline(components=[CueBertInferenceGAT, ScopeBertInferenceGAT],
                        model_paths=[mcue_path, mscope_path],
                        device="cuda:0",
                        max_length=128,
                        model_architecture=BERTResidualGATv2ContextGatedFusion)

        res = pipe.run(batch_tokens)
        Pipeline.pretty_print(res)
        return res

    @staticmethod
    def cue_baseline_test(model_path: str, tok_path: str, device: str = "cuda:0", max_length: int = 128) -> None:
        # Example batched input
        batch_tokens = [
            ['The', 'majority', 'of', 'these', 'TCC', 'exhibited', 'a', 'strongly', 'polarized', 'Th2', 'cytokine',
             'profile', ',', 'and', 'the', 'production', 'of', 'IFN-gamma', 'could', 'not', 'be', 'induced', 'by',
             'exogenous', 'IL-12', '.']
        ]
        original_input = [
            ['The', 'majority', 'of', 'these', 'TCC', 'exhibited', 'a', 'strongly', 'polarized', 'Th2', 'cytokine',
             'profile', ',', 'and', 'the', 'production', 'of', 'IFN-gamma', 'could', 'not', 'be', 'induced', 'by',
             'exogenous', 'IL-12', '.']
        ]

        cb_inf = CueBertInference.init_component(model_path, device, max_length)
        res = cb_inf.run(batch_tokens)

        cb_inf.pretty_print(res)

    @staticmethod
    def scope_baseline_test(model_path: str, tok_path: str, device: str = "cuda:0", max_length: int = 128) -> None:
        # Example batched input
        batch_tokens = [
            ['The', 'majority', 'of', 'these', 'TCC', 'exhibited', 'a', 'strongly', 'polarized', 'Th2', 'cytokine',
             'profile', ',', 'and', 'the', 'production', 'of', 'IFN-gamma', 'could', '[CUE]', 'be', 'induced', 'by',
             'exogenous', 'IL-12', '.'],
            ['In', 'contrast', 'to', 'anti-CD3/IL-2-activated', 'LN', 'cells', ',', 'adoptive', 'transfer', 'of',
             'freshly', 'isolated', 'tumor-draining', 'LN', 'T', 'cells', 'has', '[CUE]', 'therapeutic', 'activity',
             '.']
        ]
        original_input = [
            ['The', 'majority', 'of', 'these', 'TCC', 'exhibited', 'a', 'strongly', 'polarized', 'Th2', 'cytokine',
             'profile', ',', 'and', 'the', 'production', 'of', 'IFN-gamma', 'could', 'not', 'be', 'induced', 'by',
             'exogenous', 'IL-12', '.'],
            ['In', 'contrast', 'to', 'anti-CD3/IL-2-activated', 'LN', 'cells', ',', 'adoptive', 'transfer', 'of',
             'freshly', 'isolated', 'tumor-draining', 'LN', 'T', 'cells', 'has', 'no', 'therapeutic', 'activity',
             '.']
        ]

        cb_inf = ScopeBertInference.init_component(model_path, device, max_length)
        res = cb_inf.run(batch_tokens)
        cb_inf.pretty_print(res)

    @staticmethod
    def cue_gat_test(model_path: str, device: str = "cuda:0", max_length: int = 128) -> None:
        # Example batched input
        batch_tokens = [
            ['The', 'majority', 'of', 'these', 'TCC', 'exhibited', 'a', 'strongly', 'polarized', 'Th2', 'cytokine',
             'profile', ',', 'and', 'the', 'production', 'of', 'IFN-gamma', 'could', 'not', 'be', 'induced', 'by',
             'exogenous', 'IL-12', '.']
        ]
        original_input = [
            ['The', 'majority', 'of', 'these', 'TCC', 'exhibited', 'a', 'strongly', 'polarized', 'Th2', 'cytokine',
             'profile', ',', 'and', 'the', 'production', 'of', 'IFN-gamma', 'could', 'not', 'be', 'induced', 'by',
             'exogenous', 'IL-12', '.']
        ]

        cb_inf = CueBertInferenceGAT.init_component(model_path, device, max_length)
        batch_predictions = cb_inf.run(batch_tokens, original_input)

        cb_inf.pretty_print(batch_predictions)

    @staticmethod
    def scope_gat_test(model_path: str, device: str = "cuda:0", max_length: int = 128) -> None:
        # Example batched input
        batch_tokens = [
            ['The', 'majority', 'of', 'these', 'TCC', 'exhibited', 'a', 'strongly', 'polarized', 'Th2', 'cytokine',
             'profile', ',', 'and', 'the', 'production', 'of', 'IFN-gamma', 'could', '[CUE]', 'be', 'induced', 'by',
             'exogenous', 'IL-12', '.']
        ]
        original_input = [
            ['The', 'majority', 'of', 'these', 'TCC', 'exhibited', 'a', 'strongly', 'polarized', 'Th2', 'cytokine',
             'profile', ',', 'and', 'the', 'production', 'of', 'IFN-gamma', 'could', 'not', 'be', 'induced', 'by',
             'exogenous', 'IL-12', '.']
        ]

        cb_inf = ScopeBertInferenceGAT.init_component(model_path, device)
        batch_predictions = cb_inf.run(batch_tokens, original_input)

        cb_inf.pretty_print(batch_predictions)