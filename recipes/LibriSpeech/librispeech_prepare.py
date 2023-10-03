"""
Data preparation.

Download: http://www.openslr.org/12

Author
------
Mirco Ravanelli, Ju-Chieh Chou, Loren Lugosch 2020
"""

import os
import csv
import random
import torchaudio
import numpy as np
from collections import Counter
from dataclasses import dataclass
import functools
import logging
from speechbrain.utils.data_utils import download_file, get_all_files
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
    merge_csvs,
    read_audio_info,
)
from speechbrain.utils.parallel import parallel_map
from pathlib import Path

logger = logging.getLogger(__name__)
OPT_FILE = "opt_librispeech_prepare.pkl"
SAMPLERATE = 16000


def prepare_librispeech(
    data_folder,
    save_folder,
    tr_splits=[],
    dev_splits=[],
    te_splits=[],
    select_n_sentences=None,
    merge_lst=[],
    merge_name=None,
    create_lexicon=False,
    pitch_enabled=False,
    pitch_n_fft=1024,
    pitch_hop_length=256,
    pitch_min_f0=65,
    pitch_max_f0=2093,
    sample_rate=16000,
    normalize=False,
    use_relative_paths=False,
    skip_prep=False,
):
    """
    This class prepares the csv files for the LibriSpeech dataset.
    Download link: http://www.openslr.org/12

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original LibriSpeech dataset is stored.
    tr_splits : list
        List of train splits to prepare from ['test-others','train-clean-100',
        'train-clean-360','train-other-500'].
    dev_splits : list
        List of dev splits to prepare from ['dev-clean','dev-others'].
    te_splits : list
        List of test splits to prepare from ['test-clean','test-others'].
    save_folder : str
        The directory where to store the csv files.
    select_n_sentences : int
        Default : None
        If not None, only pick this many sentences.
    merge_lst : list
        List of librispeech splits (e.g, train-clean, train-clean-360,..) to
        merge in a singe csv file.
    merge_name: str
        Name of the merged csv file.
    create_lexicon: bool
        If True, it outputs csv files containing mapping between grapheme
        to phonemes. Use it for training a G2P system.
    normalize: bool
        where or not to apply simple volume normalization (dividing by the
        maximum)
    use_relative_paths: bool
        whether or not to use relative paths. This is useful on shared clusters
        where the target path might not be fixed
    skip_prep: bool
        If True, data preparation is skipped.


    Example
    -------
    >>> data_folder = 'datasets/LibriSpeech'
    >>> tr_splits = ['train-clean-100']
    >>> dev_splits = ['dev-clean']
    >>> te_splits = ['test-clean']
    >>> save_folder = 'librispeech_prepared'
    >>> prepare_librispeech(data_folder, save_folder, tr_splits, dev_splits, te_splits)
    """

    if skip_prep:
        return
    data_folder = data_folder
    splits = tr_splits + dev_splits + te_splits
    save_folder = Path(save_folder)
    select_n_sentences = select_n_sentences
    conf = {
        "select_n_sentences": select_n_sentences,
    }

    # Other variables
    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_opt = os.path.join(save_folder, OPT_FILE)

    process_pitch_fn = None
    pitch_folder = None
    # Create the pitch folder if it does not already exist
    if pitch_enabled:
        pitch_folder = save_folder / "pitch"
        pitch_folder.mkdir(parents=True, exist_ok=True)
        process_pitch_fn = functools.partial(
            torchaudio.functional.compute_kaldi_pitch,
            sample_rate=SAMPLERATE,
            frame_length=(pitch_n_fft / SAMPLERATE * 1000),
            frame_shift=(pitch_hop_length / SAMPLERATE * 1000),
            min_f0=pitch_min_f0,
            max_f0=pitch_max_f0,
        )

    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, conf):
        logger.info("Skipping preparation, completed in previous run.")
        return
    else:
        logger.info("Data_preparation...")

    # Additional checks to make sure the data folder contains Librispeech
    check_librispeech_folders(data_folder, splits)

    # create csv files for each split
    all_texts = {}
    for split_index in range(len(splits)):

        split = splits[split_index]

        wav_lst = get_all_files(
            os.path.join(data_folder, split), match_and=[".flac"], exclude_and=["._"]
        )

        text_lst = get_all_files(
            os.path.join(data_folder, split), match_and=["trans.txt"], exclude_and=["._"]
        )

        text_dict = text_to_dict(text_lst)
        all_texts.update(text_dict)

        if select_n_sentences is not None:
            n_sentences = select_n_sentences[split_index]
        else:
            n_sentences = len(wav_lst)

        process_audio_fn = None
        wav_folder = None
        resample = SAMPLERATE != sample_rate
        if resample or normalize:
            process_audio_fn = functools.partial(
                process_audio,
                resample=resample,
                sample_rate=sample_rate,
                normalize=normalize,
            )
            wav_folder = save_folder / "wav"
            wav_folder.mkdir(parents=True, exist_ok=True)

        path_placeholders = None
        if use_relative_paths:
            path_placeholders = {
                "data_root": Path(data_folder),
                "prepared_data_root": Path(save_folder),
            }

        create_csv(
            save_folder,
            wav_lst,
            text_dict,
            split,
            n_sentences,
            process_pitch_fn,
            pitch_folder,
            process_audio_fn,
            wav_folder,
            path_placeholders
        )

    # Merging csv file if needed
    if merge_lst and merge_name is not None:
        merge_files = [split_libri + ".csv" for split_libri in merge_lst]
        merge_csvs(
            data_folder=save_folder, csv_lst=merge_files, merged_csv=merge_name,
        )

    # Create lexicon.csv and oov.csv
    if create_lexicon:
        create_lexicon_and_oov_csv(all_texts, data_folder, save_folder)

    # saving options
    save_pkl(conf, save_opt)


def create_lexicon_and_oov_csv(all_texts, data_folder, save_folder):
    """
    Creates lexicon csv files useful for training and testing a
    grapheme-to-phoneme (G2P) model.

    Arguments
    ---------
    all_text : dict
        Dictionary containing text from the librispeech transcriptions
    data_folder : str
        Path to the folder where the original LibriSpeech dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    Returns
    -------
    None
    """
    # If the lexicon file does not exist, download it
    lexicon_url = "http://www.openslr.org/resources/11/librispeech-lexicon.txt"
    lexicon_path = os.path.join(save_folder, "librispeech-lexicon.txt")

    if not os.path.isfile(lexicon_path):
        logger.info(
            "Lexicon file not found. Downloading from %s." % lexicon_url
        )
        download_file(lexicon_url, lexicon_path)

    # Get list of all words in the transcripts
    transcript_words = Counter()
    for key in all_texts:
        transcript_words.update(all_texts[key].split("_"))

    # Get list of all words in the lexicon
    lexicon_words = []
    lexicon_pronunciations = []
    with open(lexicon_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            word = line.split()[0]
            pronunciation = line.split()[1:]
            lexicon_words.append(word)
            lexicon_pronunciations.append(pronunciation)

    # Create lexicon.csv
    header = "ID,duration,char,phn\n"
    lexicon_csv_path = os.path.join(save_folder, "lexicon.csv")
    with open(lexicon_csv_path, "w") as f:
        f.write(header)
        for idx in range(len(lexicon_words)):
            separated_graphemes = [c for c in lexicon_words[idx]]
            duration = len(separated_graphemes)
            graphemes = " ".join(separated_graphemes)
            pronunciation_no_numbers = [
                p.strip("0123456789") for p in lexicon_pronunciations[idx]
            ]
            phonemes = " ".join(pronunciation_no_numbers)
            line = (
                ",".join([str(idx), str(duration), graphemes, phonemes]) + "\n"
            )
            f.write(line)
    logger.info("Lexicon written to %s." % lexicon_csv_path)

    # Split lexicon.csv in train, validation, and test splits
    split_lexicon(save_folder, [98, 1, 1])


def split_lexicon(data_folder, split_ratio):
    """
    Splits the lexicon.csv file into train, validation, and test csv files

    Arguments
    ---------
    data_folder : str
        Path to the folder containing the lexicon.csv file to split.
    split_ratio : list
        List containing the training, validation, and test split ratio. Set it
        to [80, 10, 10] for having 80% of material for training, 10% for valid,
        and 10 for test.

    Returns
    -------
    None
    """
    # Reading lexicon.csv
    lexicon_csv_path = os.path.join(data_folder, "lexicon.csv")
    with open(lexicon_csv_path, "r") as f:
        lexicon_lines = f.readlines()
    # Remove header
    lexicon_lines = lexicon_lines[1:]

    # Shuffle entries
    random.shuffle(lexicon_lines)

    # Selecting lines
    header = "ID,duration,char,phn\n"

    tr_snts = int(0.01 * split_ratio[0] * len(lexicon_lines))
    train_lines = [header] + lexicon_lines[0:tr_snts]
    valid_snts = int(0.01 * split_ratio[1] * len(lexicon_lines))
    valid_lines = [header] + lexicon_lines[tr_snts : tr_snts + valid_snts]
    test_lines = [header] + lexicon_lines[tr_snts + valid_snts :]

    # Saving files
    with open(os.path.join(data_folder, "lexicon_tr.csv"), "w") as f:
        f.writelines(train_lines)
    with open(os.path.join(data_folder, "lexicon_dev.csv"), "w") as f:
        f.writelines(valid_lines)
    with open(os.path.join(data_folder, "lexicon_test.csv"), "w") as f:
        f.writelines(test_lines)


@dataclass
class LSRow:
    snt_id: str
    spk_id: str
    duration: float
    file_path: str
    words: str
    pitch_file_path: str


def process_line(
    wav_file,
    text_dict,
    process_pitch_fn=None,
    process_audio_fn=None,
    wav_folder=None,
    pitch_folder=None,
    path_placeholders=None
) -> LSRow:
    snt_id = wav_file.split("/")[-1].replace(".flac", "")
    spk_id = "-".join(snt_id.split("-")[0:2])
    wrds = text_dict[snt_id]
    wrds = " ".join(wrds.split("_"))

    info = read_audio_info(wav_file)
    duration = info.num_frames / info.sample_rate
    pitch_file_path = None
    if process_audio_fn is not None:
        wav_file = process_audio_fn(wav_file, wav_folder)
    if process_pitch_fn is not None:
        pitch_file_path = process_pitch(
            wav_file, process_pitch_fn, pitch_folder
        )
    if path_placeholders is not None:
        wav_file = relativize_path(
            wav_file,
            path_placeholders
        )
        pitch_file_path = relativize_path(
            pitch_file_path,
            path_placeholders
        )

    return LSRow(
        snt_id=snt_id,
        spk_id=spk_id,
        duration=duration,
        file_path=wav_file,
        words=wrds,
        pitch_file_path=pitch_file_path
    )


def relativize_path(file_name, path_placeholders):
    """Converts an absolute path to a relative path
    
    Arguments
    ---------
    file_name: str
        the file name (which might be absolute)
    path_placeholders: dict
        possible path replacements
        
    Returns
    -------
    result: str
        the relativized file name"""
    file_path = Path(file_name)
    for key, path in path_placeholders.items():
        if file_path.is_relative_to(path):
            relative_path = file_path.relative_to(path)
            file_name = f"${key}/{relative_path}"
            break
    return file_name


def process_pitch(
    wav_file,
    process_pitch_fn,
    pitch_folder
):
    """Computes and saves pitch information

    Arguments
    ---------
    wav_file: str|pathlib.Path
        the source wave file
    process_pitch_fn: callable
        the function to compute pitch information
    pitch_folder: pathlib.Path
        the destination path
    """
    audio, _ = torchaudio.load(wav_file)
    pitch_file_name = Path(wav_file).stem + ".npy"
    pitch_file_path = pitch_folder / pitch_file_name
    pitch = process_pitch_fn(audio)
    pitch = pitch[0, :, 0]
    np.save(pitch_file_path, pitch)
    return pitch_file_path


def create_csv(
    save_folder,
    wav_lst,
    text_dict,
    split,
    select_n_sentences,
    process_pitch_fn,
    pitch_folder,
    process_audio_fn,
    wav_folder,
    path_placeholders,
):
    """
    Create the dataset csv file given a list of wav files.

    Arguments
    ---------
    save_folder : str
        Location of the folder for storing the csv.
    wav_lst : list
        The list of wav files of a given data split.
    text_dict : list
        The dictionary containing the text of each sentence.
    split : str
        The name of the current data split.
    select_n_sentences : int, optional
        The number of sentences to select.
    process_pitch_fn: callable
        The function to compute pitch information
    pitch_folder: str
        The path where pitch files will be saved
    process_audio_fn: callable
        The function to process the audio file
    path_placeholders: dict
        Paths that can be relativized using placehodlers

    Returns
    -------
    None
    """
    # Setting path for the csv file
    csv_file = os.path.join(save_folder, split + ".csv")
    if os.path.exists(csv_file):
        logger.info("Csv file %s already exists, not recreating." % csv_file)
        return

    # Preliminary prints
    msg = "Creating csv lists in  %s..." % (csv_file)
    logger.info(msg)
    header = ["ID", "duration", "wav", "spk_id", "wrd"]
    if process_audio_fn is not None:
        header.append("pitch")
    csv_lines = [header]

    snt_cnt = 0
    line_processor = functools.partial(
        process_line,
        text_dict=text_dict,
        process_pitch_fn=process_pitch_fn,
        pitch_folder=pitch_folder,
        process_audio_fn=process_audio_fn,
        wav_folder=wav_folder,
        path_placeholders=path_placeholders
    )
    # Processing all the wav files in wav_lst
    # FLAC metadata reading is already fast, so we set a high chunk size
    # to limit main thread CPU bottlenecks
    for row in parallel_map(line_processor, wav_lst, chunk_size=2):
        csv_line = [
            row.snt_id,
            str(row.duration),
            row.file_path,
            row.spk_id,
            row.words,
        ]
        if row.pitch_file_path is not None:
            csv_line.append(row.pitch_file_path)

        # Appending current file to the csv_lines list
        csv_lines.append(csv_line)

        snt_cnt = snt_cnt + 1

        # parallel_map guarantees element ordering so we're OK
        if snt_cnt == select_n_sentences:
            break

    # Writing the csv_lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final print
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)


def process_audio(
    file_name,
    destination_folder,
    resample=False,
    sample_rate=16000,
    normalize=False
):
    """Processes an audio file

    Arguments
    ---------
    file_name: str|pathlib.Path
        the file name of the original wave file
    destination_folder: pathlib.Path
        the path to which wav files will be saved
    resample: bool
        whether the audio needs to be resampled
    sample_rate: bool
        the target sample rate
    normalize: bool
        whether the samples should be re-normalized
    """
    wav, original_sample_rate = torchaudio.load(file_name)
    if resample:
        wav = torchaudio.functional.resample(
            wav,
            orig_freq=original_sample_rate,
            new_freq=sample_rate
        )
    if normalize:
        wav = wav / wav.abs().max()

    destination_file_name = (
        destination_folder / Path(file_name).name
    )
    torchaudio.save(
        destination_file_name, wav, sample_rate=sample_rate
    )
    return destination_file_name


def skip(splits, save_folder, conf):
    """
    Detect when the librispeech data prep can be skipped.

    Arguments
    ---------
    splits : list
        A list of the splits expected in the preparation.
    save_folder : str
        The location of the seave directory
    conf : dict
        The configuration options to ensure they haven't changed.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking csv files
    skip = True

    for split in splits:
        if not os.path.isfile(os.path.join(save_folder, split + ".csv")):
            skip = False

    #  Checking saved options
    save_opt = os.path.join(save_folder, OPT_FILE)
    if skip is True:
        if os.path.isfile(save_opt):
            opts_old = load_pkl(save_opt)
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False

    return skip


def text_to_dict(text_lst):
    """
    This converts lines of text into a dictionary-

    Arguments
    ---------
    text_lst : str
        Path to the file containing the librispeech text transcription.

    Returns
    -------
    dict
        The dictionary containing the text transcriptions for each sentence.

    """
    # Initialization of the text dictionary
    text_dict = {}
    # Reading all the transcription files is text_lst
    for file in text_lst:
        with open(file, "r") as f:
            # Reading all line of the transcription file
            for line in f:
                line_lst = line.strip().split(" ")
                text_dict[line_lst[0]] = "_".join(line_lst[1:])
    return text_dict


def check_librispeech_folders(data_folder, splits):
    """
    Check if the data folder actually contains the LibriSpeech dataset.

    If it does not, an error is raised.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If LibriSpeech is not found at the specified path.
    """
    # Checking if all the splits exist
    for split in splits:
        split_folder = os.path.join(data_folder, split)
        if not os.path.exists(split_folder):
            err_msg = (
                "the folder %s does not exist (it is expected in the "
                "Librispeech dataset)" % split_folder
            )
            raise OSError(err_msg)
