"""
Evaluation routines for MadMixture shared latent space multimodal audio model

Authors
 * Artem Ploujnikov 2023
"""

import os
import json
import torch
from speechbrain.utils.metric_stats import ErrorRateStats
from speechbrain.dataio.encoder import TextEncoder
from speechbrain.decoders.seq2seq import S2SBaseSearcher
from speechbrain.utils.data_utils import undo_padding
from functools import partial

DEFAULT_OUTPUT = "./output"

# TODO: Work in progress. The interfaces will change - this will
# be updated to also support Tensorboard
class MadMixtureEvaluator:
    """The high-level evaluation wrapper for the MadMixture model.
    Evaluation may consist of multiple tasks, each of which may 
    produce a series of outputs and reports"""
    def __init__(self, model, tasks):
        self.model = model
        self.tasks = tasks
        for task in self.tasks.values():
            task.bind(model)

    def append(self, ids, inputs, lengths, targets):
        """Adds a data sample to the evaluation, for
        all available evaluation tasks
        
        Arguments
        ---------
        ids: list
            the IDs of samples, for reporting purposes, in
            the same order as samples in the batch

        inputs: dict
            all available input features, for each modality
        
        lengths: dict
            all available output features, for each modality

        targets: dict
            ground truths for each modality

        """
        for task in self.tasks.values():
            task.append(ids, inputs, lengths, targets)

    def report(self, path):
        for key, task in self.tasks.items():
            task.report(os.path.join(path, key))


class EvaluationTask:
    """A superclass for evaluation tasks
    
    Arguments
    ---------
    model: speechbrain.lobes.models.madmixture.MadMixture
        a MadMixture model
    """
    def __init__(self, model=None):
        if model is not None:
            self.bind(model)

    def bind(self, model):
        """Binds this evaluation task to a model
        
        Arguments
        ---------
        model: speechbrain.lobes.models.madmixture.MadMixture
            a MadMixture model
        """
        self.model = model

    def append(self, ids, inputs, lengths, targets):
        """Adds a data sample to the evaluation
        
        Arguments
        ---------
        ids: list
            the IDs of samples, for reporting purposes, in
            the same order as samples in the batch

        inputs: dict
            all available input features, for each modality
        
        lengths: dict
            all available output features, for each modality

        targets: dict
            ground truths for each modality
        """
        raise NotImplementedError()

    def report(self, path):
        raise NotImplementedError()



class ModalityTransferTask(EvaluationTask):
    """An evaluator module that attempts a cross-transfer for all
    of the available modalities and evaluates the outputs using
    available techniques
    """
    def __init__(self, evaluators, model=None):
        self.evaluator_fn = evaluators
        super().__init__(model)

    def bind(self, model):
        super().bind(model)
        
        # Possible transfers consist of all available pairs
        # of modalities.
        # Supported transfers refer to transfers for which
        # an evaluator has been registered.
        self.supported_transfers = [
            (src, tgt)
            for src, tgt in self.model.possible_transfers
            if tgt in self.evaluator_fn
        ]
        self.targets = list(self.evaluator_fn.keys())

        self.evaluators = {
            (src, tgt): self.evaluator_fn[tgt]()
            for src, tgt in self.supported_transfers
        }


    def append(self, ids, inputs, lengths, targets):
        """Adds a data sample to the evaluation
        
        Arguments
        ---------
        ids: list
            the IDs of samples, for reporting purposes, in
            the same order as samples in the batch

        inputs: dict
            all available input features, for each modality
        
        lengths: dict
            all available output features, for each modality

        targets: dict
            ground truths for each modality
        """        
        # Run evaluation for all primary modalities
        for src in self.model.primary_modalities:
            self.append_modality(ids, inputs, lengths, targets, src)

    def append_modality(self, ids, inputs, lengths, targets, src):
        """Adds a data sample to the evaluation, for a single source
        modality identified by the `src` parameter
        
        Arguments
        ---------
        ids: list
            the IDs of samples, for reporting purposes, in
            the same order as samples in the batch

        inputs: dict
            all available input features, for each modality
        
        lengths: dict
            all available output features, for each modality

        targets: dict
            ground truths for each modality
        
        src: str
            the source modality key
        """        
        # A single transfer can produce latents from one modality
        # (running an encoder once with its aligner) and then
        # decode to all available targets
        rec, latents, alignments, enc_out = self.model.transfer(
            inputs=inputs,
            lengths=lengths,
            src=src,
            tgt=self.targets
        )
        # Run evaluators for all items decoded, comparing to ground
        # truths and producing metrics
        for tgt, tgt_rec in rec.items():
            self.evaluators[src, tgt].append(
                ids=ids,
                predict=tgt_rec,
                target=targets[tgt],
                latent=latents[src],
                src_lengths=lengths[src],
                tgt_lengths=lengths[tgt]
            )

    def report(self, path):
        """Outputs reports for the modality transfer task
        
        Arguments
        ---------
        path: str
            the report path
        """
        for src, tgt in self.supported_transfers:
            tgt_path = os.path.join(path, f"{src}_to_{tgt}")
            self.evaluators[src, tgt].report(tgt_path)


    
TOKEN_SEQUENCE_SUMMARY_REPORT = "summary.json"
TOKEN_SEQUENCE_ALIGNMENT_REPORT = "alignment.txt"

class OutputEvaluator:
    """A superclass for output evaluators. Each output
    evaluator is responsible for taking raw outputs,
    comparing them to the targets and producing a report.
    In the case of modalities that are difficult to evaluate
    automatically, it may be sufficient to record
    samples."""
    def append(self, ids, predict, target, latent, src_lengths, tgt_lengths):
        """Remembers predictions, targets and any associated
        metrics for future reporting
        
        Arguments
        ---------
        ids: list
            the IDs of samples, for reporting purposes, in
            the same order as samples in the batch
        predict: torch.Tensor
            predictions corresponding to samples in the batch
        target: torch.Tensor
            ground truths
        target: torch.Tensor
            latent representations - useful when using non-trivial
            decoding methods, such as a beam search over autoregressive
            models
        src_lengths: torch.Tensor
            source sequence lengths
        tgt_lengths: torch.Tensor:
            target sequence lengths

        lengths: torch.Tensor
            the relative lengths of the ground truths
        """
        raise NotImplementedError()
    
    def report(self, path):
        """Outputs the relevant reports
        
        Arguments
        ---------
        path: str
            the path to output reports, samples, etc
        """
        raise NotImplementedError()
    

class TokenSequenceEvaluator(OutputEvaluator):
    """An evaluator for targets that can be interpreted
    as sequences of tokens (words, characters, etc)
    
    decoder: speechbrain.dataio.encoder.TextEncoder|callable
        a TextEncoder instance or a custom function that
        converts raw predictions to labels

    hyp: callable
        a callable function that converts raw predictions 
        (e.g. estimated probabilities) to token indexes.
        If not provided, a simple argmax over the last
        dimension will be used 

    src_lengths: torch.Tensor
        source sequence lengths

    tgt_lengths: torch.Tensor:
        target sequence lengths
        
    """
    def __init__(self, decoder=None, hyp=None):
        self.decoder = decoder
        if isinstance(self.decoder, TextEncoder):
            decoder_fn = self.decoder.decode_ndim
        elif callable(self.decoder):
            decoder_fn = self.decoder
        else:
            raise ValueError(f"Invalid decoder {decoder_fn}")
        self.decoder_fn = decoder_fn
        self.error_stats = ErrorRateStats(
            mode="generic"
        )
        if hyp is None:
            hyp = hyp_argmax
        elif isinstance(hyp, S2SBaseSearcher):
            hyp = partial(hyp_s2s_search, searcher=hyp)

        self.hyp = hyp

    def append(self, ids, predict, target, latent, src_lengths, tgt_lengths):
        """Appends a batch of predictions to the
        evaluator
        """
        hyps = self.hyp(predict, src_lengths, latent)
        self.error_stats.append(
            ids=ids,
            predict=hyps,
            target=target,
            target_len=tgt_lengths,
            ind2lab=self.decoder_fn
        )

    def report(self, path):
        """Outputs the relevant reports
        
        Arguments
        ---------
        path: str
            the path to output reports, samples, etc
        """
        if not os.path.exists(path):
            os.makedirs(path)
        self.report_summary(path)
        self.report_detail(path)

    def report_summary(self, path):
        """Outputs a summary report
        
        Arguments
        ---------
        path: str
            the path to output reports, samples, etc
        """        
        summary = self.error_stats.summarize()
        file_name = os.path.join(path, TOKEN_SEQUENCE_SUMMARY_REPORT)
        with open(file_name, "w") as report_file:
            json.dump(summary, report_file)
    
    def report_detail(self, path):
        """Outputs a detail report
        
        Arguments
        ---------
        path: str
            the path to output reports, samples, etc
        """
        file_name = os.path.join(path, TOKEN_SEQUENCE_ALIGNMENT_REPORT)
        with open(file_name, "w") as report_file:
            self.error_stats.write_stats(report_file)


def hyp_argmax(probs, lengths=None, latent=None, eos_index=None):
    """A simple hypothesis function selecting the most probable
    token. It can be used for training or debugging but it is not
    recommended for final use

    Arguments
    ---------
    probs: torch.Tensor
        a (batch x length x token) tensor of probabilities
    lengths: torch.Tensor
        relative lengths of inputs - not used for this method
    latents: torch.Tensor
        latent representations - not used for this method
    eos_token: int
        the index of the EOS token. If set to None, all sequences
        will be expected to be complete

    Returns
    -------
    hyps: list
         nested list of decoded sequences
    """
    hyps = probs.argmax(-1)
    if eos_index is None:
        lengths = torch.ones(hyps.size(0)).float()
    else:
        lengths_abs = (hyps == eos_index).argmax(dim=-1)
        lengths = lengths_abs / hyps.size(1)
    return undo_padding(hyps, lengths)

def hyp_s2s_search(probs, lengths, latent, searcher):
    """A hypothesis function that uses an S2S searcher, such
    as a beam search
    
    Arguments
    ----------
    probs: torch.Tensor
        a (batch x length x token) tensor of probabilities
    lengths: torch.Tensor
        relative lengths of inputs - not used for this method
    latents: torch.Tensor
        latent representations - not used for this method
    searcher: S2SBaseSearcher
        a sequence-to-sequence searcher

    Returns
    -------
    hyps: list
         nested list of decoded sequences

    """
    hyps, _ = searcher(latent, lengths)
    return hyps
