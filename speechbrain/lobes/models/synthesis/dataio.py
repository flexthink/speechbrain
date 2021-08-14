"""Pipeline transofmrtaion functions that can be useful for
multiple speech synthesis models (wrapped Torchaudio transforms,
embeddings, etc)

Authors
* Artem Ploujnikov 2021
"""

import speechbrain as sb
from torchaudio import transforms
from speechbrain.dataio.encoder import CategoricalEncoder
from speechbrain.pretrained.interfaces import SpeakerRecognition
import re
import os


def wrap_transform(transform_type, takes=None, provides=None):
    """Wraps a Torch transform for the pipeline, returning a
    decorator

    Arguments
    ---------
    transform_type: torch.nn.Module
        a Torch transform (from torchaudio.Transforms) to be wrapped
    takes: str
        the name of the pipeline input
    provides: str
        the name of the pipeline output

    Arguments
    ---------
    takes: str
        the name of the pipeline input
    provides: str
        the name of the pipeline output

    Returns
    -------
    result: DynamicItem
      A wrapped transformation function
    """
    default_takes = takes
    default_provides = provides

    def decorator(takes=None, provides=None, *args, **kwargs):
        transform = transform_type(*args, **kwargs)

        @sb.utils.data_pipeline.takes(takes or default_takes)
        @sb.utils.data_pipeline.provides(provides or default_provides)
        def f(*args, **kwargs):
            return transform.to(args[0].device)(*args, **kwargs)

        return f

    return decorator


SPEAKER_EMBEDDINGS_DEFAULT_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
RE_NON_ALPHA = "[^A-Za-z]"


def _compute_default_savedir(source):
    """Computes the default save directory based on the repo name

    Arguments
    ---------
    source: str
        the repo name

    Returns
    -------
        the target directory
    """
    source_norm = re.sub(RE_NON_ALPHA, "_", source)
    return f"pretrained_{source_norm}"


def pretrained_speaker_embeddings(
    takes=["sig", "sig_lengths"],
    provides="speaker_embed",
    source=None,
    savedir=None,
):
    """
    A pipeline function that obtains pretrained speaker embeddings

    Arguments
    ---------
    takes: str
        The name of the pipeline input
    provides: str
        The name of the pipeline output
    source: str
        The name of the pretrained HuggingFace Hub repository
        to obtain the pretrained embeddings from (the default is
        speechbrain/spkrec-ecapa-voxceleb)
    savedir: str
        the directory to which the downloaded pretrained embeddings
        will be saved

    Returns
    -------
    result: DynamicItem
      A wrapped transformation function
    """
    if not source:
        source = SPEAKER_EMBEDDINGS_DEFAULT_SOURCE
    if not savedir:
        savedir = os.path.join(".", _compute_default_savedir(source))

    verification = SpeakerRecognition.from_hparams(
        source=source, savedir=savedir
    )

    @sb.utils.data_pipeline.takes(*takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(sig, sig_lengths):
        embedding = verification.encode_batch(sig, sig_lengths.float())
        if len(embedding.shape) == 3:
            embedding = embedding.squeeze(1)
        return embedding

    return f


def categorical(takes, provides, encoder=None):
    """A pipeline function that encodes categorical information,
    such as speaker IDs

    Arguments
    ---------
    takes: str
        the name of the pipeline input
    provides: str
        the name of the pipeline output
    encoder: CategoricalEncoder
        the categorical encoder to be used (particularly)
        useful if its state needs to be saved

    Returns
    -------
    result: DynamicItem
      A wrapped transformation function
    """
    if not encoder:
        encoder = CategoricalEncoder()

    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(label):
        encoder.ensure_label(label)
        return encoder.ensure_label(label)

    return f


@sb.utils.data_pipeline.takes("wav")
@sb.utils.data_pipeline.provides("sig", "sig_lengths")
def audio_pipeline(file_name: str):
    """A pipeline function that reads a single file and reads
    audio from it into a tensor
    """
    audio = sb.dataio.dataio.read_audio(file_name)
    return audio, len(audio)


def transpose_spectrogram(takes, provides):
    """A pipeline function that transposes a spectrogram along the
    last two axes

    Arguments
    ---------
    takes: str
        the name of the pipeline input
    provides: str
        the name of the pipeline output

    Returns
    -------
    result: DymamicItem
    """

    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(spectrogram):
        return spectrogram.transpose(-1, -2)

    return f


resample = wrap_transform(
    transforms.Resample, takes="sig", provides="sig_resampled"
)
mel_spectrogram = wrap_transform(
    transforms.MelSpectrogram, takes="sig", provides="mel"
)
spectrogram = wrap_transform(
    transforms.Spectrogram, takes="sig", provides="spectrogram"
)
inverse_spectrogram = wrap_transform(
    transforms.GriffinLim, takes="spectrogram", provides="sig"
)
inverse_mel = wrap_transform(
    transforms.InverseMelScale, takes="mel", provides="linear"
)


def load_datasets(hparams, dataset_prep):
    """A convenience function to load multiple datasets, from hparams

    Arguments
    ---------
    hparams: dict
        a hyperparameters file
    dataset_prep: callable
        a function taking two parameters: (dataset, hparams) that

    """
    result = {}
    for name, dataset_params in hparams["datasets"].items():
        loader = dataset_params["loader"]
        dataset = loader(dataset_params["path"])
        filter_file_name = dataset_params.get("filter")
        if filter_file_name:
            dataset = filter_by_id_list(dataset, filter_file_name)
        result[name] = dataset_prep(dataset, hparams)
    return result


def filter_by_id_list(dataset, file_name):
    """Filters a dataset by selecting IDs from a text file - used to
    "freeze" predetermined splits

    Arguments
    ---------
    dataset: DynamicItemDataSet
        a dataset
    file_name: str
        the name of the filter file
    """
    with open(file_name) as filter_file:
        item_ids = set(line.strip() for line in filter_file)

    return dataset.filtered_sorted(
        key_test={"id": lambda item_id: item_id in item_ids}
    )


def pretrained_vocoder(vocoder, takes="mel", provides="wav"):
    """Creates a pipeline element for a pretrained vocoder

    Arguments
    ---------
    vocoder: speechbrain.pretrained.interfaces.VocoderWrapper
        a pretrained vocoder instance
    takes: str
        the name of the pipeline input
    provides: str
        the name of the pipeline output
    """

    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(mel):
        mel = mel.to(vocoder.device)
        return vocoder.synthesize(mel)

    return f
