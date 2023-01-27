"""
Evaluation routines for MadMixture shared latent space multimodal audio model

Authors
 * Artem Ploujnikov 2023
"""

from .madmixture import ModalityType
from speechbrain.utils.metric_stats import ErrorRateStats

DEFAULT_OUTPUT = "./output"

# TODO: Work in progress
class MadMixtureEvaluator:
    """The high-level evaluation wrapper"""
    def __init__(self, model, output_path=None):
        if not output_path:
            output_path = DEFAULT_OUTPUT
        self.model = model
        self.output_path = output_path

    def evaluate(self):
        pass


class ModalityTransferEvaluator:
    """An evaluator module that attempts a cross-transfer for all
    of the available modalities and evaluates the outputs using
    available techniques
    """
    def __init__(self, model):
        self.model = model

    def evaluate(self, inputs, lengths):
        # Select the source modality
        primary_modalities = [
            modality.name
            for modality in self.model.modalities
            if modality.mod_type == ModalityType.PRIMARY
        ]
        for src in primary_modalities:
            self.evaluate_modality(self, inputs, lengths, src)

    def evaluate_modality(self, inputs, lengths, src):
        rec, latents, alignments, enc_out = self.model.transfer(
            inputs=inputs,
            lengths=lengths,
            src=src
        )
        

class TokenSequenceEvaluator:
    def __init__(self):
        self.error_stats = ErrorRateStats(
            mode="generic"
        )