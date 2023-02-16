"""
Evaluation routines for MadMixture shared latent space multimodal audio model

Authors
 * Artem Ploujnikov 2023
"""

import os
import json
import torch
import logging
from speechbrain.utils.metric_stats import ErrorRateStats
from speechbrain.dataio.encoder import TextEncoder
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.dataio.dataio import write_audio
from speechbrain.decoders.seq2seq import S2SBaseSearcher
from speechbrain.utils.data_utils import undo_padding, undo_padding_tensor
from torch import nn
from torchaudio import functional as AF
from functools import partial

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = "./output"

# TODO: Work in progress. The interfaces will change - this will
# be updated to also support Tensorboard
class MadMixtureEvaluator:
    """The high-level evaluation wrapper for the MadMixture model.
    Evaluation may consist of multiple tasks, each of which may 
    produce a series of outputs and reports
    
    Arguments
    ---------
    model: speechbrain.lobes.models.madmixture.MadMixture
        a MadMixture model instance
    tasks: dict
        a str -> callable dictionary
    vis_sample: set
        s selection of IDs to be used for visualization samples
    """
    def __init__(self, model, tasks, vis_sample=None):
        self.model = model
        self.tasks = {key: task() for key, task in tasks.items()}
        for task in self.tasks.values():
            task.bind(model)
        self.use_vis_sample(vis_sample)

    def append(self, ids, inputs, latents, alignments, lengths, targets, out_context):
        """Adds a data sample to the evaluation, for
        all available evaluation tasks
        
        Arguments
        ---------
        ids: list
            the IDs of samples, for reporting purposes, in
            the same order as samples in the batch

        inputs: dict
            all available input features, for each modality

        latents: dict
            latent representations, for each modality

        alignments: dict
            alignment matrices, for each modality
        
        lengths: dict
            all available output features, for each modality

        targets: dict
            ground truths for each modality
        
        out_context: dict
            the output context (decoder-specific tensors, may contain
            alignments, gates, etc)

        """
        for task in self.tasks.values():
            task.append(ids, inputs, latents, alignments, lengths, targets, out_context)

    def report(self, path):
        """Outputs evaluation reports
        
        Arguments
        ---------
        path: str
            the report path
        """
        for key, task in self.tasks.items():
            task.report(os.path.join(path, key))

    def use_vis_sample(self, data_ids):
        """Uses the specified selection of IDs as a visualization sample
        
        Arguments
        ---------
        data_ids: enumerable
            A collection of data IDs
        """
        if not data_ids:
            data_ids = []
        self.vis_sample = set(data_ids)
        for task in self.tasks.values():
            task.use_vis_sample(data_ids)

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

    def append(self, ids, inputs, latent, alignments, lengths, targets, out_context):
        """Adds a data sample to the evaluation
        
        Arguments
        ---------
        ids: list
            the IDs of samples, for reporting purposes, in
            the same order as samples in the batch

        inputs: dict
            all available input features, for each modality
        
        latents: dict
            latent representations, for each modality

        alignments: dict
            alignment matrices, for each modality            
        
        lengths: dict
            all available output features, for each modality

        targets: dict
            ground truths for each modality

        out_context: dict
            the output context (decoder-specific tensors, may contain
            alignments, gates, etc)

        """
        raise NotImplementedError()

    def report(self, path):
        """Outputs reports for the modality transfer task
        
        Arguments
        ---------
        path: str
            the report path
        """        
        raise NotImplementedError()
    
    def use_vis_sample(self, data_ids):
        """Uses the specified selection of IDs as a visualization sample
        
        Arguments
        ---------
        data_ids: enumerable
            A collection of data IDs
        """
        if not data_ids:
            data_ids = []
        self.vis_sample = set(data_ids)


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


    def append(self, ids, inputs, latents, alignments, lengths, targets, out_context):
        """Adds a data sample to the evaluation
        
        Arguments
        ---------
        ids: list
            the IDs of samples, for reporting purposes, in
            the same order as samples in the batch

        inputs: dict
            all available input features, for each modality
        
        latents: dict
            latent representations, for each modality

        alignments: dict
            alignments, for each modalities
        
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
        rec, latents, alignments, enc_out, out_context = self.model.transfer(
            inputs=inputs,
            lengths=lengths,
            src=src,
            tgt=self.targets
        )
        # Run evaluators for all items decoded, comparing to ground
        # truths and producing metrics
        for tgt, tgt_rec in rec.items():
            evaluator = self.evaluators[src, tgt]
            src_latents = latents[src]
            _, tgt_targets = targets[src], targets[tgt]
            src_lengths, tgt_lengths = lengths[src], lengths[tgt]
            if evaluator.vis_samples_only:
                (
                    eval_ids,
                    eval_tgt_rec,
                    eval_tgt_targets,
                    eval_src_lengths,
                    eval_src_latents
                ) = filter_sample_ids(
                    ids,
                    self.vis_sample,
                    tgt_rec,
                    tgt_targets,
                    src_lengths,
                    src_latents
                )
            else:

                (
                    eval_ids,
                    eval_tgt_rec,
                    eval_tgt_targets,
                    eval_src_lengths,
                    eval_src_latents
                ) = (
                    ids,
                    tgt_rec,
                    tgt_targets,
                    src_lengths,
                    src_latents
                )
            evaluator.append(
                ids=eval_ids,
                predict=eval_tgt_rec,
                target=eval_tgt_targets,
                latent=eval_src_latents,
                src_lengths=eval_src_lengths,
                tgt_lengths=tgt_lengths,
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
            os.makedirs(tgt_path, exist_ok=True)
            self.evaluators[src, tgt].report(tgt_path)


LATENT_DEFAULT_FIGSIZE = (16, 16)
ALGNMENT_DEFAULT_FIGSIZE = (16, 8)


class LatentSpaceAnalysisTask(EvaluationTask):
    """A task that performs latent space analysis. Currently it will
    output latent spaces and alignments for each modality"""
    def __init__(
            self,
            model=None,
            context_alignment_keys=None,
            figsize_latent=LATENT_DEFAULT_FIGSIZE,
            figsize_aligment=ALGNMENT_DEFAULT_FIGSIZE
        ):
        super().__init__(model)
        self.ids = []
        self.latents = {}
        self.alignments = {}
        self.lengths = []
        self.context_alignments = {}
        if context_alignment_keys is None:
            context_alignment_keys = []
        self.context_alignment_keys = set(context_alignment_keys)
        self.figsize_latent = figsize_latent
        self.figsize_alignment = figsize_aligment

        self.plt = _get_matplotlib()

    """Computes and outputs latent representations for analysis"""
    def append(self, ids, inputs, latents, alignments, lengths, targets, out_context):
        """Adds a data sample to the evaluation
        
        Arguments
        ---------
        ids: list
            the IDs of samples, for reporting purposes, in
            the same order as samples in the batch

        inputs: dict
            all available input features, for each modality

        inputs: dict
            all available input features, for each modality            

        latents: dict
            latent representations, for each modality

        alignments: dict
            alignments, for each modalities
                    
        lengths: dict
            all available output features, for each modality

        targets: dict
            ground truths for each modality

        out_context: dict
            the output context
        """
        ids, lengths, latents, alignments, out_context = filter_sample_ids(
            ids, self.vis_sample, lengths, latents, alignments, out_context
        )
        if ids:
            self.ids.extend(ids)
            self.lengths.extend(lengths)
            self._extend_dict(self.latents, latents)
            self._extend_dict(self.alignments, alignments)
            context_alignments = {
                key: value for key, value in out_context.items()
                if key in self.context_alignment_keys
            }
            self._extend_dict(
                self.context_alignments, context_alignments)

    def _extend_dict(self, target, values):
        """Adds per-modality values to a tracking dictionary
        
        Arguments
        ---------
        target: dict
            a dictionary
        values: dict
            a key -> tensor dictionary (e.g. per-modality values)
        """
        for key, modality_values in values.items():
            if key not in target:
                target[key] = []
            target[key].extend(
                modality_values.detach().cpu()
            )

    def report(self, path):
        """Outputs reports for the modality transfer task
        
        Arguments
        ---------
        path: str
            the report path
        """
        os.makedirs(path, exist_ok=True)
        self.save_raw(path)
        if self.plt is not None:
            self.save_images(path)

    def save_raw(self, path):
        """Saves raw tensors"""
        data = {
            "ids": self.ids,
            "latents": self.latents,
            "alignments": self.alignments,
            "lengths": self.lengths,
            "context_alignments": self.context_alignments
        }
        file_name = os.path.join(path, "raw.pt")
        torch.save(data, file_name)
    
    def save_images(self, path):
        for idx, sample_id in enumerate(self.ids):
            sample_latents = {
                key: modality_latents[idx]
                for key, modality_latents in self.latents.items()
            }
            sample_alignments = {
                key: modality_alignments[idx]
                for key, modality_alignments in self.alignments.items()
            }
            file_name = os.path.join(path, f"{sample_id}.png")
            fig = self.plot_latent(sample_id, sample_latents, sample_alignments)
            try:
                fig.savefig(file_name)
            finally:
                self.plt.close(fig)
            
            sample_context_alignments = {
                key: context_alignments[idx]
                for key, context_alignments in self.context_alignments.items()
            }
            file_name = os.path.join(path, f"context-{sample_id}.png")
            fig = self.plot_alignment(sample_id, sample_context_alignments)
            try:
                fig.savefig(file_name)
            finally:
                self.plt.close(fig)


    def plot_latent(self, sample_id, sample_latents, sample_alignments):
        """Plots a latent space with the corresponding alignments
        
        Arguments
        ---------
        sample_id: str
            the sample identifier
        sample_latents: torch.Tensor
            the latent space context
        sample_alignments: torch.Tensor
            the corresponding alignments
        """
        fig = self.plt.figure(figsize=self.figsize_latent)
        try:
            modality_count = len(sample_latents)
            fig.suptitle(f"Latent: {sample_id}")
            for idx, (key, sample_latent) in enumerate(sample_latents.items()):
                ax = fig.add_subplot(2, modality_count, idx + 1)
                ax.set_title(key)
                ax.set_ylabel("Features")
                ax.set_xlabel("Time")
                im = ax.imshow(sample_latent.t(), aspect="auto", origin="lower")
                fig.colorbar(im, orientation="vertical")
            for idx, (key, sample_alignment) in enumerate(sample_alignments.items()):
                ax = fig.add_subplot(2, modality_count, modality_count + idx + 1)
                ax.set_title(f"align: {key}")
                ax.set_ylabel("Outputs")
                ax.set_xlabel("Inputs")
                im = ax.imshow(sample_alignment, aspect="auto", origin="lower")
                fig.colorbar(im, orientation="vertical")                   
            fig.tight_layout()
            return fig
        except:
            self.plt.close(fig)
            raise

    def plot_alignment(self, sample_id, alignments):
        """Plots simple alignments
        
        Arguments
        ---------
        sample_id: str
            the sample ID
        alignments: dict
            a key -> alignment dictionary
        """
        fig = self.plt.figure(figsize=self.figsize_alignment)
        try:
            count = len(alignments)
            fig.suptitle(f"Context: {sample_id}")
            for idx, (key, alignment) in enumerate(alignments.items()):
                ax = fig.add_subplot(1, count, idx + 1)
                ax.set_title(f"{key}", fontsize=10)
                im = ax.imshow(alignment, aspect="auto", origin="lower")
                fig.colorbar(im, orientation="vertical")
            return fig
        except:
            self.plt.close()
            raise


    
TOKEN_SEQUENCE_SUMMARY_REPORT = "summary.json"
TOKEN_SEQUENCE_ALIGNMENT_REPORT = "alignment.txt"

class OutputEvaluator:
    """A superclass for output evaluators. Each output
    evaluator is responsible for taking raw outputs,
    comparing them to the targets and producing a report.
    In the case of modalities that are difficult to evaluate
    automatically, it may be sufficient to record
    samples."""
    vis_samples_only = False

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

    ignore_tokens: enumerable
        a collection of tokens (in decoded form) that
        will be removed from sequences
        
    """
    def __init__(self, decoder=None, hyp=None, ignore_tokens=None):
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
        self.ignore_tokens = set(ignore_tokens) if ignore_tokens else None

    def append(self, ids, predict, target, latent, src_lengths, tgt_lengths):
        """Updates sequence metrics with the data samples provided
        
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
        hyps = self.hyp(predict, src_lengths, latent)
        hyps_clean = self._clean(hyps)
        targets_list = undo_padding(target, tgt_lengths)
        targets_clean = self._clean(targets_list)
        self.error_stats.append(
            ids=ids,
            predict=hyps_clean,
            target=targets_clean,
            ind2lab=self.decoder_fn
        )

    def _clean(self, hyps):
        """Removes any ignored tokens from the hypothesis list"""
        return [
            [item  for item in batch
             if item not in self.ignore_tokens]
            for batch in hyps
        ] if self.ignore_tokens else hyps

    def report(self, path):
        """Outputs the relevant reports
        
        Arguments
        ---------
        path: str
            the path to output reports, samples, etc
        """
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


SPECTROGRAM_DEFAULT_FIGURE_SIZE = (16, 10)
class SpectrogramEvaluator(OutputEvaluator):
    """An evaluator that outputs audio spectrogram samples

    Arguments
    ---------
    figsize: tuple
        the Matplotlib figure size to be used for spectrograms
        (defaults to (16, 10))

    vocoder: callable
        a vocoder that can convert spectrograms to raw
        waveform audio

    normalization: torch.nn.ModuleList
        A list of modules implementing the denormalize method
        (reversible normalizations). If provided, they will be applied
        in reverse order

    """
    vis_samples_only = True
    def __init__(
            self,
            figsize=None,
            vocoder=None,
            vocoder_pre=None,
            vocoder_batch_size=16,
            sample_rate=16000,
            normalization=None,
            spec_db=True,
            spec_ref=10.,
            spec_power=1.
        ):
        self.device = None
        self.predict = []
        self.target = []
        self.ids = []
        if figsize is None:
            figsize = SPECTROGRAM_DEFAULT_FIGURE_SIZE
        self.figsize = figsize
        self.normalization = normalization
        self.vocoder_fn = vocoder
        self._vocoder = None
        if vocoder_pre is None:
            vocoder_pre = nn.Identity()
        self.vocoder_pre = vocoder_pre
        self.vocoder_batch_size = vocoder_batch_size
        self.sample_rate = sample_rate
        self.spec_db = spec_db
        self.spec_ref = spec_ref
        self.spec_power = spec_power
        self.plt = _get_matplotlib()   
    
    def append(self, ids, predict, target, latent, src_lengths, tgt_lengths):
        """Remembers raw outputs for spectrogram plotting
        
        Arguments
        ---------
        ids: list
            the IDs of samples, for reporting purposes, in
            the same order as samples in the batch
        predict: torch.Tensor
            predictions corresponding to samples in the batch
        latent: torch.Tensor
            ground truths
        target: torch.Tensor
            latent representations - useful when using non-trivial
            decoding methods, such as a beam search over autoregressive
            models
        src_lengths: torch.Tensor
            source sequence lengthsaten
        tgt_lengths: torch.Tensor:
            target sequence lengths

        lengths: torch.Tensor
            the relative lengths of the ground truths
        """
        self.ids.extend(ids)
        if self.device is None:
            self.device = predict.device
        self.predict.extend(
            undo_padding_tensor(predict.detach().cpu(), tgt_lengths)
        )
        self.target.extend(
            undo_padding_tensor(target.detach().cpu(), tgt_lengths)
        )

    def report(self, path):
        """Outputs all accumulated spectrograms
        
        Arguments
        ---------
        path: str
            the target path
        """
        self.save_spectrograms_raw(path)
        if self.plt is not None:
            self.save_spectrograms_image(path)
        if self.vocoder_fn is not None:
            self.save_audio(path)
    

    def save_spectrograms_image(self, path):
        """Saves all spectrograms as images, using Matplotlib

        Arguments
        ---------
        path: str
            the target path            
        
        """
        for sample_id, predict_sample, target_sample in zip(self.ids, self.predict, self.target):
            self.save_spectrogram_image(path, sample_id, predict_sample, target_sample)
    
    def save_spectrograms_raw(self, path):
        """Saves all spectrograms as a raw PyTorch file
        
        Arguments
        ---------
        path: str
            the target path            
        """
        file_name = os.path.join(path, "raw.pt")
        data = {
            "ids": self.ids,
            "predict": self.predict,
            "target": self.target,        
        }
        torch.save(data, file_name)

    def save_spectrogram_image(self, path, sample_id, predict_sample, target_sample):
        """Saves a single spectrogram image
        
        Arguments
        ---------
        path: str
            the target path
        predict: torch.Tensor
            the prediction (a single spectrogram)
        target: torch.Tensor
            the target (a single spectrogram)        
        """
        fig = self.plot_comparison(sample_id, predict_sample, target_sample)
        try:
            file_name = os.path.join(path, f"spec_{sample_id}.png")
            fig.savefig(file_name)
        finally:
            self.plt.close(fig)

    def plot_comparison(self, sample_id, predict, target):
        """Plots a single prediction/target spectrogram comparison
        
        Arguments
        ---------
        sample_id: str
            the ID of the sample
        predict: torch.Tensor
            the prediction (a single spectrogram)
        target: torch.Tensor
            the target (a single spectrogram)
        
        Returns
        -------
        fig: matplotlib.figure.Figure
            a Matplotlib figure
        """
        fig = self.plt.figure()
        predict_plot = predict[:target.size(0), :].t()
        target_plot = target.t()
        try:
            ax = fig.add_subplot(1, 2, 1)
            fig.suptitle(f"Spectrogram: {sample_id}")
            ax.set_title("Prediction")
            ax.set_ylabel("Features")
            ax.set_xlabel("Time")
            im = ax.imshow(predict_plot, aspect="auto", origin="lower")
            fig.colorbar(im, orientation="vertical")
            ax = fig.add_subplot(1, 2, 2)
            ax.set_title("Target")
            ax.set_xlabel("Time")
            im = ax.imshow(target_plot, aspect="auto", origin="lower")
            fig.colorbar(im, orientation="vertical")
            fig.tight_layout()
            return fig
        except:
            self.plt.close(fig)
            raise
    @property
    def vocoder(self):
        if self._vocoder is None:
            self._vocoder = self.vocoder_fn(run_opts={"device": self.device})
        return self._vocoder

    def save_audio(self, path):
        """Generates and saves raw audio samples from spectrograms
        using the provided vocoder"""
        batches = batchify(
            sample_ids=self.ids,
            predict=self.predict, target=self.target,
            batch_size=self.vocoder_batch_size
        )
        for batch in batches:
            batch = batch.to(self.device)
            spec_target = batch.target.data.transpose(-1, -2)
            spec_predict = batch.predict.data.transpose(-1, -2)
            spec_predict = spec_predict[:, :, :spec_target.size(-1)]
            wav_target = self.generate_audio(spec_target)
            wav_predict = self.generate_audio(spec_predict)
            self.save_audio_batch(
                sample_ids=batch.sample_ids,
                wav=wav_predict,
                lengths=batch.predict.lengths,
                prefix="predict",
                path=path
            )
            self.save_audio_batch(
                sample_ids=batch.sample_ids,
                wav=wav_target,
                lengths=batch.target.lengths,
                prefix="target",
                path=path
            )

    def generate_audio(self, spec):
        with torch.no_grad():
            if self.normalization is not None:
                for norm in reversed(self.normalization):
                    spec = norm.denormalize(spec)
            if self.spec_db:
                spec = AF.DB_to_amplitude(
                    spec,
                    ref=self.spec_ref,
                    power=self.spec_power
                )
            spec = self.vocoder_pre(spec)
            return self.vocoder(spec)
    
    def save_audio_batch(self, path, sample_ids, wav, lengths, prefix):
        lengths_abs = (wav.size(-1) * lengths).int()
        wav = wav.squeeze(1)
        for sample_id, wav_item, wav_length in zip(sample_ids, wav, lengths_abs):
            file_name = os.path.join(path, f"{prefix}_{sample_id}.wav")
            write_audio(
                file_name,
                wav_item[:wav_length].cpu(),
                samplerate=self.sample_rate
            )


def _get_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        logger.warn("matplotlib is not available - cannot log figures")
        return None


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


def filter_sample_ids(ids, sample_ids, *args):
    """Filters the arguments to include only items
    included in the sample
    
    Arguments
    ---------
    ids: list
        a list of item IDs
    sample_ids: set
        the IDs to be included
    args: list
        tensors ordered by id"""
    if sample_ids is None:
        return ids, *args
    filtered = [
        (item_id, idx) for idx, item_id in enumerate(ids)
        if item_id in sample_ids
    ]
    filtered_ids = [item_id for item_id, _ in filtered]
    indexes = [idx for _, idx in filtered]
    return filtered_ids, *(
        _filter_by_index(item, indexes) for item in args)


def _filter_by_index(items, indexes):
    """Filteres a sequence by index. If a tensor is provided,
    slicing is used. Otherwise, it will use a simple (but slow)
    list comprehension
    
    Arguments
    ---------
    items: torch.Tensor|list
        a sequence of items
    indexes: list
        the indexes to select"""
    if torch.is_tensor(items):
        filtered_items = items[indexes]
    elif isinstance(items, dict):
        filtered_items = {
            key: _filter_by_index(value, indexes)
            for key, value in items.items()
        }
    else:
        filtered_items = [items[idx] for idx in indexes]
    return filtered_items


def batchify(batch_size, **kwargs):
    """Converts lists of tensors or tensor-like objects to
    a generator of PaddedBatch objects"""
    first_key = next(iter(kwargs.keys()))
    total_length = len(kwargs[first_key])
    for idx in range(0, total_length, batch_size):
        batch_data_items = [
            {key: value[idx + batch_idx]
             for key, value in kwargs.items()}
             for batch_idx in range(batch_size)
            if idx + batch_idx < total_length
        ]
        batch = PaddedBatch(batch_data_items)
        yield batch
