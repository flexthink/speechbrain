"""
An approximation of the SpecAugment algorithm, carried out in the time domain.

Authors
 * Peter Plantinga 2020
"""
import torch
from speechbrain.processing.speech_augmentation import (
    SpeedPerturb,
    DropFreq,
    DropChunk,
)


class TimeDomainSpecAugment(torch.nn.Module):
    """A time-domain approximation of the SpecAugment algorithm.

    Arguments
    ---------
    perturb_prob : float from 0 to 1
        The probability that a batch will have speed peturbation applied.
    drop_freq_prob : float from 0 to 1
        The probability that a batch will have frequencies dropped.
    drop_chunk_prob : float from 0 to 1
        The probability that a batch will have chunks dropped.
    speeds : list of ints
        A set of different speeds to use to perturb each batch.
        See `speechbrain.processing.speech_augmentation.SpeedPerturb`
    sample_rate : int
        Sampling rate of the input waveforms.
    drop_freq_count_low : int
        Lowest number of frequencies that could be dropped.
    drop_freq_count_high : int
        Highest number of frequencies that could be dropped.
    drop_chunk_count_low : int
        Lowest number of chunks that could be dropped.
    drop_chunk_count_high : int
        Highest number of chunks that could be dropped.
    drop_chunk_length_low : int
        Lowest length of chunks that could be dropped.
    drop_chunk_length_high : int
        Highest length of chunks that could be dropped.

    Example
    -------
    >>> inputs = torch.randn([10, 16000])
    >>> feature_maker = TimeDomainSpecAugment(speeds=[80])
    >>> feats = feature_maker(inputs, torch.ones(10), init_params=True)
    >>> feats.shape
    torch.Size([10, 12800])
    """

    def __init__(
        self,
        perturb_prob=1.0,
        drop_freq_prob=1.0,
        drop_chunk_prob=1.0,
        speeds=[95, 100, 105],
        sample_rate=16000,
        drop_freq_count_low=0,
        drop_freq_count_high=3,
        drop_chunk_count_low=0,
        drop_chunk_count_high=5,
        drop_chunk_length_low=1000,
        drop_chunk_length_high=2000,
    ):
        super().__init__()
        self.speed_perturb = SpeedPerturb(
            perturb_prob=perturb_prob, orig_freq=sample_rate, speeds=speeds,
        )
        self.drop_freq = DropFreq(
            drop_prob=drop_freq_prob,
            drop_count_low=drop_freq_count_low,
            drop_count_high=drop_freq_count_high,
        )
        self.drop_chunk = DropChunk(
            drop_prob=drop_chunk_prob,
            drop_count_low=drop_chunk_count_low,
            drop_count_high=drop_chunk_count_high,
            drop_length_low=drop_chunk_length_low,
            drop_length_high=drop_chunk_length_high,
        )

    def forward(self, waveforms, lengths, init_params=False):
        """Returns the distorted waveforms.

        Arguments
        ---------
        waveforms : torch.Tensor
            The waveforms to distort
        """
        # Augmentation
        if self.training:
            waveforms = self.speed_perturb(waveforms)
            waveforms = self.drop_freq(waveforms)
            waveforms = self.drop_chunk(waveforms, lengths)

        return waveforms
