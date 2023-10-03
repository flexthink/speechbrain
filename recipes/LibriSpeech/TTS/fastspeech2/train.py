"""
Recipe for training the FastSpeech2 Text-To-Speech model
Instead of using pre-extracted phoneme durations from MFA,
This recipe trains an internal alignment from scratch, as introduced in:
https://arxiv.org/pdf/2108.10447.pdf (One TTS Alignment To Rule Them All)
To run this recipe, do the following:
# python train_internal_alignment.py hparams/train_internal_alignment.yaml

Authors
* Yingzhi Wang 2023
"""

import os
import sys
import torch
import logging
import torchaudio
import numpy as np
import speechbrain as sb
from speechbrain.pretrained import HIFIGAN
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.data_utils import scalarize

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)


class FastSpeech2Brain(sb.Brain):
    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp and initializes statistics"""
        self.hparams.progress_sample_logger.reset()
        self.last_epoch = 0
        self.last_batch = None
        self.last_loss_stats = {}
        return super().on_fit_start()

    def compute_forward(self, batch, stage):
        """Computes the forward pass
        Arguments
        ---------
        batch: str
            a single batch
        stage: speechbrain.Stage
            the training stage
        Returns
        -------
        the model output
        """
        inputs, _ = self.batch_to_device(batch)
        return self.hparams.model(*inputs)

    def fit_batch(self, batch):
        """Fits a single batch
        Arguments
        ---------
        batch: tuple
            a training batch
        Returns
        -------
        loss: torch.Tensor
            detached loss
        """
        result = super().fit_batch(batch)
        self.hparams.noam_annealing(self.optimizer)
        return result

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.
        Arguments
        ---------
        predictions : torch.Tensor
            The model generated spectrograms and other metrics from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        x, y, metadata = self.batch_to_device(batch, return_metadata=True)
        self.last_batch = [x[0], y[-1], y[-2], predictions[0], *metadata]
        self._remember_sample([x[0], *y, *metadata], predictions)
        loss = self.hparams.criterion(
            predictions, y, self.hparams.epoch_counter.current
        )
        self.last_loss_stats[stage] = scalarize(loss)
        return loss["total_loss"]

    def _remember_sample(self, batch, predictions):
        """Remembers samples of spectrograms and the batch for logging purposes
        Arguments
        ---------
        batch: tuple
            a training batch
        predictions: tuple
            predictions (raw output of the FastSpeech2
             model)
        """
        (
            phoneme_padded,
            mel_padded,
            pitch,
            energy,
            output_lengths,
            input_lengths,
            labels,
            wavs,
        ) = batch

        (
            mel_post,
            postnet_mel_out,
            predict_durations,
            predict_pitch,
            average_pitch,
            predict_energy,
            average_energy,
            predict_mel_lens,
            alignment_durations,
            alignment_soft,
            alignment_logprob,
            alignment_mas,
        ) = predictions
        self.hparams.progress_sample_logger.remember(
            target=self.process_mel(mel_padded, output_lengths),
            output=self.process_mel(postnet_mel_out, output_lengths),
            raw_batch=self.hparams.progress_sample_logger.get_batch_sample(
                {
                    "tokens": phoneme_padded,
                    "input_lengths": input_lengths,
                    "mel_target": mel_padded,
                    "mel_out": postnet_mel_out,
                    "mel_lengths": predict_mel_lens,
                    "durations": alignment_durations,
                    "predict_durations": predict_durations,
                    "labels": labels,
                    "wavs": wavs,
                }
            ),
        )

    def process_mel(self, mel, len, index=0):
        """Converts a mel spectrogram to one that can be saved as an image
        sample  = sqrt(exp(mel))
        Arguments
        ---------
        mel: torch.Tensor
            the mel spectrogram (as used in the model)
        len: int
            length of the mel spectrogram
        index: int
            batch index
        Returns
        -------
        mel: torch.Tensor
            the spectrogram, for image saving purposes
        """
        assert mel.dim() == 3
        return torch.sqrt(torch.exp(mel[index][: len[index]]))

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # At the end of validation, we can write
        if stage == sb.Stage.VALID:
            # Update learning rate
            self.last_epoch = epoch
            lr = self.hparams.noam_annealing.current_lr

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(  # 1#2#
                stats_meta={"Epoch": epoch, "lr": lr},
                train_stats=self.last_loss_stats[sb.Stage.TRAIN],
                valid_stats=self.last_loss_stats[sb.Stage.VALID],
            )
            output_progress_sample = (
                self.hparams.progress_samples
                and epoch % self.hparams.progress_samples_interval == 0
                and epoch >= self.hparams.progress_samples_min_run
            )

            if output_progress_sample:
                logger.info("Saving predicted samples")
                inference_mel, mel_lens = self.run_inference()
                self.hparams.progress_sample_logger.save(epoch)
                self.run_vocoder(inference_mel, mel_lens)
            # Save the current checkpoint and delete previous checkpoints.
            # UNCOMMENT THIS
            self.checkpointer.save_and_keep_only(
                meta=self.last_loss_stats[stage], min_keys=["total_loss"],
            )
        # We also write statistics about test data spectogramto stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=self.last_loss_stats[sb.Stage.TEST],
            )

    def run_inference(self):
        """Produces a sample in inference mode with predicted durations."""
        if self.last_batch is None:
            return
        tokens, *_ = self.last_batch

        (
            _,
            postnet_mel_out,
            _,
            _,
            _,
            _,
            _,
            predict_mel_lens,
            _,
            _,
            _,
            _,
        ) = self.hparams.model(tokens)
        self.hparams.progress_sample_logger.remember(
            infer_output=self.process_mel(
                postnet_mel_out, [len(postnet_mel_out[0])]
            )
        )
        return postnet_mel_out, predict_mel_lens

    def run_vocoder(self, inference_mel, mel_lens):
        """Uses a pretrained vocoder to generate audio from predicted mel
        spectogram. By default, uses speechbrain hifigan.
        Arguments
        ---------
        inference_mel: torch.Tensor
            predicted mel from fastspeech2 inference
        mel_lens: torch.Tensor
            predicted mel lengths from fastspeech2 inference
            used to mask the noise from padding
        """
        if self.last_batch is None:
            return
        *_, wavs = self.last_batch

        inference_mel = inference_mel[: self.hparams.progress_batch_sample_size]
        mel_lens = mel_lens[0 : self.hparams.progress_batch_sample_size]
        assert (
            self.hparams.vocoder == "hifi-gan"
            and self.hparams.pretrained_vocoder is True
        ), "Specified vocoder not supported yet"
        logger.info(
            f"Generating audio with pretrained {self.hparams.vocoder_source} vocoder"
        )
        hifi_gan = HIFIGAN.from_hparams(
            source=self.hparams.vocoder_source,
            savedir=self.hparams.vocoder_download_path,
        )
        waveforms = hifi_gan.decode_batch(
            inference_mel.transpose(2, 1), mel_lens, self.hparams.hop_length
        )
        for idx, wav in enumerate(waveforms):

            path = os.path.join(
                self.hparams.progress_sample_path,
                str(self.last_epoch),
                f"pred_{Path(wavs[idx]).stem}.wav",
            )
            torchaudio.save(path, wav, self.hparams.sample_rate)

    def batch_to_device(self, batch, return_metadata=False):
        """Transfers the batch to the target device
        Arguments
        ---------
        batch: tuple
            the batch to use
        Returns
        -------
        batch: tuple
            the batch on the correct device
        """

        (
            phoneme_padded,
            input_lengths,
            mel_padded,
            pitch_padded,
            energy_padded,
            output_lengths,
            # len_x,
            labels,
            wavs,
        ) = batch

        # durations = durations.to(self.device, non_blocking=True).long()
        phonemes = phoneme_padded.to(self.device, non_blocking=True).long()
        input_lengths = input_lengths.to(self.device, non_blocking=True).long()
        spectogram = mel_padded.to(self.device, non_blocking=True).float()
        pitch = pitch_padded.to(self.device, non_blocking=True).float()
        energy = energy_padded.to(self.device, non_blocking=True).float()
        mel_lengths = output_lengths.to(self.device, non_blocking=True).long()
        x = (phonemes, spectogram, pitch, energy)
        y = (spectogram, pitch, energy, mel_lengths, input_lengths)
        metadata = (labels, wavs)
        if return_metadata:
            return x, y, metadata
        return x, y


def dataio_prepare(hparams):
    "Creates the datasets and their data processing pipelines."
    # Load lexicon
    lexicon = read_token_list(hparams["token_list_file"])
    input_encoder = hparams.get("input_encoder")

    # add a dummy symbol for idx 0 - used for padding.
    lexicon = ["@@"] + lexicon
    input_encoder.update_from_iterable(lexicon, sequence_input=False)
    input_encoder.add_unk()

    # load audio, text and durations on the fly; encode audio and text.
    @sb.utils.data_pipeline.takes("id", "wav", "wrd", "pitch")
    @sb.utils.data_pipeline.provides("label", "mel_text_pair")
    def audio_pipeline(label, wav, tokens, pitch):
        input_seq = input_encoder.encode_sequence_torch(tokens).int()

        audio, fs = torchaudio.load(wav)
        audio = audio.squeeze()
        mel, energy = hparams["mel_spectrogram"](audio=audio)

        pitch = np.load(pitch)
        pitch = torch.from_numpy(pitch)
        pitch = pitch[: mel.shape[-1]]
        mel_text_pair = (input_seq, mel, pitch, energy, len(input_seq), len(mel))
        return label, mel_text_pair

    # define splits and load it as sb dataset
    datasets = {}

    for dataset in hparams["splits"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[f"{dataset}_csv"],
            replacements={
                "data_root": hparams["data_folder"],
                "prepared_data_root": hparams["prepared_data_folder"]
            },
            dynamic_items=[audio_pipeline],
            output_keys=["mel_text_pair", "label", "wav", "pitch"],
        )
    return datasets


def read_token_list(file_name):
    """Reads a simple text file with tokens (e.g. characters or phonemes) listed
    one per line

    Arguments
    ---------
    file_name: str
        the file name

    Returns
    -------
    result: list
        a list of tokens
    """
    if not os.path.isfile(file_name):
        raise ValueError(f"Token file {file_name} not found")
    with open(file_name) as token_file:
        return [line.strip("\r\n") for line in token_file if line]


def main():
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    sb.utils.distributed.ddp_init_group(run_opts)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    select_n_sentences = None
    if hparams["use_subset"]:
        select_n_sentences = hparams["select_n_sentences"]

    from librispeech_prepare import prepare_librispeech

    sb.utils.distributed.run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "pitch_enabled": True,
            "pitch_n_fft": hparams["n_fft"],
            "pitch_hop_length": hparams["hop_length"],
            "pitch_min_f0": hparams["min_f0"],
            "pitch_max_f0": hparams["max_f0"],            
            "select_n_sentences": select_n_sentences,
            "sample_rate": hparams["sample_rate"],
            "normalize": hparams["audio_normalize"],
            "use_relative_paths": hparams["use_relative_paths"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    datasets = dataio_prepare(hparams)

    # Brain class initialization
    fastspeech2_brain = FastSpeech2Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    # Training
    fastspeech2_brain.fit(
        fastspeech2_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )


if __name__ == "__main__":
    main()
