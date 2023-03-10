"""Loggers for experiment monitoring.

Authors
 * Peter Plantinga 2020
"""
import logging
import ruamel.yaml
import torch
import csv
import os
import numpy as np

logger = logging.getLogger(__name__)


class TrainLogger:
    """Abstract class defining an interface for training loggers."""

    def log_stats(
        self,
        stats_meta,
        train_stats=None,
        valid_stats=None,
        test_stats=None,
        verbose=False,
    ):
        """Log the stats for one epoch.

        Arguments
        ---------
        stats_meta : dict of str:scalar pairs
            Meta information about the stats (e.g., epoch, learning-rate, etc.).
        train_stats : dict of str:list pairs
            Each loss type is represented with a str : list pair including
            all the values for the training pass.
        valid_stats : dict of str:list pairs
            Each loss type is represented with a str : list pair including
            all the values for the validation pass.
        test_stats : dict of str:list pairs
            Each loss type is represented with a str : list pair including
            all the values for the test pass.
        verbose : bool
            Whether to also put logging information to the standard logger.
        """
        raise NotImplementedError


class FileTrainLogger(TrainLogger):
    """Text logger of training information.

    Arguments
    ---------
    save_file : str
        The file to use for logging train information.
    precision : int
        Number of decimal places to display. Default 2, example: 1.35e-5.
    summary_fns : dict of str:function pairs
        Each summary function should take a list produced as output
        from a training/validation pass and summarize it to a single scalar.
    """

    def __init__(self, save_file, precision=2):
        self.save_file = save_file
        self.precision = precision

    def _item_to_string(self, key, value, dataset=None):
        """Convert one item to string, handling floats"""
        if isinstance(value, float) and 1.0 < value < 100.0:
            value = f"{value:.{self.precision}f}"
        elif isinstance(value, float):
            value = f"{value:.{self.precision}e}"
        if dataset is not None:
            key = f"{dataset} {key}"
        return f"{key}: {value}"

    def _stats_to_string(self, stats, dataset=None):
        """Convert all stats to a single string summary"""
        return ", ".join(
            [self._item_to_string(k, v, dataset) for k, v in stats.items()]
        )

    def log_stats(
        self,
        stats_meta,
        train_stats=None,
        valid_stats=None,
        test_stats=None,
        verbose=True,
    ):
        """See TrainLogger.log_stats()"""
        string_summary = self._stats_to_string(stats_meta)
        for dataset, stats in [
            ("train", train_stats),
            ("valid", valid_stats),
            ("test", test_stats),
        ]:
            if stats is not None:
                string_summary += " - " + self._stats_to_string(stats, dataset)

        with open(self.save_file, "a") as fout:
            print(string_summary, file=fout)
        if verbose:
            logger.info(string_summary)


class CsvTrainLogger:
    """A train logger that outputs details as CSV
    
    Arguments
    ---------
    path: str
        the target path

    prefix: str
        the prefix prepended to the filename
    """

    def __init__(self, path, prefix=None):
        self.path = path
        if prefix is None:
            prefix = ""
        self.prefix = prefix
        self.file_names = {
            key: os.path.join(path, f"{prefix}{key}.csv")
            for key in ["train", "valid", "test"]
        }
        self.files = {}
        self.writers = {}

    def log_stats(
        self,
        stats_meta,
        train_stats=None,
        valid_stats=None,
        test_stats=None,
        verbose=False,
    ):
        """Log the stats for one epoch.

        Arguments
        ---------
        stats_meta : dict of str:scalar pairs
            Meta information about the stats (e.g., epoch, learning-rate, etc.).
        train_stats : dict of str:list pairs
            Each loss type is represented with a str : list pair including
            all the values for the training pass.
        valid_stats : dict of str:list pairs
            Each loss type is represented with a str : list pair including
            all the values for the validation pass.
        test_stats : dict of str:list pairs
            Each loss type is represented with a str : list pair including
            all the values for the test pass.
        verbose : bool
            Whether to also put logging information to the standard logger.
        """
        if train_stats is not None:
            self._write_csv_stats("train", stats_meta, train_stats)
            if verbose:
                logger.info("Train Stats: %s - %s", stats_meta, train_stats)
        if valid_stats is not None:
            self._write_csv_stats("valid", stats_meta, valid_stats)
            if verbose:
                logger.info("Valid Stats: %s - %s", stats_meta, valid_stats)
        if test_stats is not None:
            self._write_csv_stats("test", stats_meta, test_stats)
            if verbose:
                logger.info("Test Stats: %s - %s", stats_meta, test_stats)

    def _write_csv_stats(self, key, stats_meta, stats):
        """Outputs the stats for the specified key
        
        Arguments
        ---------
        key: str
            the stats key ("train", "valid" or "test")
        stats_meta: dict
            statistics metadata
        stats: dict
            statistics details
        """
        writer = self._get_writer(key, stats_meta, stats)
        full_stats = {**stats_meta, **stats}
        writer.writerow(full_stats)
        self.files[key].flush()

    def _get_writer(self, key, stats_meta, stats):
        """Gets a CSV writer for the specified key, creating
        a new file if necessary        
        
        Arguments
        ---------
        key: str
            the stats key ("train", "valid" or "test")
        stats_meta: dict
            statistics metadata
        stats: dict
            statistics details
        
        Returns
        -------
        writer: csv.DictWriter
            a CSV writer
        """
        if key not in self.writers:
            fields = [*stats_meta.keys(), *stats.keys()]
            self._create_writer(key, fields)
        return self.writers[key]

    def _create_writer(self, key, fields):
        """Creates a new CSV writer
        
        Arguments
        ---------
        key: str
            the stats key ("train", "valid" or "test")
        
        fields: list
            a list of field names
        """
        file_name = self.file_names[key]
        existing_file = os.path.exists(file_name)
        if existing_file:
            fields = self._read_fields(file_name)
        csv_file = open(file_name, "a+")
        self.files[key] = csv_file
        writer = csv.DictWriter(
            csv_file, fieldnames=fields, extrasaction="ignore"
        )
        self.writers[key] = writer
        if not existing_file:
            writer.writeheader()
        return writer

    def _read_fields(self, file_name):
        """Reads the list of fields from a CSV file
        
        Arguments
        ---------
        file_name: str
            the name of the CSV file
        
        Returns
        -------
        fields: list
            the list of fields
        """
        with open(file_name, "r") as csv_file:
            reader = csv.reader(csv_file)
            fields = next(reader)
        return fields

    def close(self):
        """Closes all files"""
        for csv_file in self.files.values():
            csv_file.close()
        self.files = {}
        self.writers = {}

    def trim(self, criteria=None, **kwargs):
        """Remove entries matching the specified criteria. Useful
        when restarting an epoch
        
        Arguments
        ---------
        criteria: dict|callable
            a function or a dictionary to determine which
            records to remove

            Criteria can also be passed as keywords arguments
        """
        if criteria is None:
            criteria = kwargs
        if isinstance(criteria, dict):
            criteria_ = criteria
            criteria = lambda item: all(
                item.get(key) == str(value) for key, value in criteria_.items()
            )
        for key in self.file_names:
            self._trim_file(key, criteria)

    def _trim_file(self, key, criteria):
        """Removes all lines matching the specified criteria
        from the specified file
        
        Arguments
        ---------
        key: str
            the stats stage key ("train", "valid" or "test")

        """
        file_name = self.file_names[key]
        if not os.path.exists(file_name):
            return
        if key in self.files:
            self.files[key].close()
            del self.files[key]
            del self.writers[key]
        tmp_file_name = os.path.join(
            self.path, "_" + os.path.basename(file_name)
        )
        with open(file_name, "r") as src_file, open(
            tmp_file_name, "w"
        ) as dest_file:
            reader = csv.DictReader(src_file)
            writer = csv.DictWriter(dest_file, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if not criteria(row):
                    writer.writerow(row)
        os.unlink(file_name)
        os.rename(tmp_file_name, file_name)


class TensorboardLogger(TrainLogger):
    """Logs training information in the format required by Tensorboard.

    Arguments
    ---------
    save_dir : str
        A directory for storing all the relevant logs.

    Raises
    ------
    ImportError if Tensorboard is not installed.
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir

        # Raises ImportError if TensorBoard is not installed
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(self.save_dir)
        self.global_step = {"train": {}, "valid": {}, "test": {}, "meta": 0}

    def log_stats(
        self,
        stats_meta,
        train_stats=None,
        valid_stats=None,
        test_stats=None,
        verbose=False,
    ):
        """See TrainLogger.log_stats()"""
        self.global_step["meta"] += 1
        for name, value in stats_meta.items():
            self.writer.add_scalar(name, value, self.global_step["meta"])

        for dataset, stats in [
            ("train", train_stats),
            ("valid", valid_stats),
            ("test", test_stats),
        ]:
            if stats is None:
                continue
            for stat, value_list in stats.items():
                if stat not in self.global_step[dataset]:
                    self.global_step[dataset][stat] = 0
                tag = f"{stat}/{dataset}"

                # Both single value (per Epoch) and list (Per batch) logging is supported
                if isinstance(value_list, list):
                    for value in value_list:
                        new_global_step = self.global_step[dataset][stat] + 1
                        self.writer.add_scalar(tag, value, new_global_step)
                        self.global_step[dataset][stat] = new_global_step
                else:
                    value = value_list
                    new_global_step = self.global_step[dataset][stat] + 1
                    self.writer.add_scalar(tag, value, new_global_step)
                    self.global_step[dataset][stat] = new_global_step

    def log_audio(self, name, value, sample_rate):
        """Add audio signal in the logs."""
        self.writer.add_audio(
            name, value, self.global_step["meta"], sample_rate=sample_rate
        )

    def log_figure(self, name, value):
        """Add a figure in the logs."""
        fig = plot_spectrogram(value)
        if fig is not None:
            self.writer.add_figure(name, fig, self.global_step["meta"])


class WandBLogger(TrainLogger):
    """Logger for wandb. To be used the same way as TrainLogger. Handles nested dicts as well.
    An example on how to use this can be found in recipes/Voicebank/MTL/CoopNet/"""

    def __init__(self, *args, **kwargs):
        try:
            yaml_file = kwargs.pop("yaml_config")
            with open(yaml_file, "r") as yaml_stream:
                # Read yaml with ruamel to ignore bangs
                config_dict = ruamel.yaml.YAML().load(yaml_stream)
            self.run = kwargs.pop("initializer", None)(
                *args, **kwargs, config=config_dict
            )
        except Exception as e:
            raise e("There was an issue with the WandB Logger initialization")

    def log_stats(
        self,
        stats_meta,
        train_stats=None,
        valid_stats=None,
        test_stats=None,
        verbose=False,
    ):
        """See TrainLogger.log_stats()"""

        logs = {}
        for dataset, stats in [
            ("train", train_stats),
            ("valid", valid_stats),
            ("test", test_stats),
        ]:
            if stats is None:
                continue
            logs[dataset] = stats

        step = stats_meta.get("epoch", None)
        if step is not None:  # Useful for continuing runs that crashed
            self.run.log({**logs, **stats_meta}, step=step)
        else:
            self.run.log({**logs, **stats_meta})


def _get_image_saver():
    """Returns the TorchVision image saver, if available
    or None if it is not - optional dependency"""
    try:
        import torchvision

        return torchvision.utils.save_image
    except ImportError:
        logger.warn("torchvision is not available - cannot save figures")
        return None


class ProgressSampleLogger:
    """A logger that outputs samples during training progress, used primarily in speech synthesis but customizable, reusable and applicable to any other generative task

    Natively, this logger supports images and raw PyTorch output.
    Other custom formats can be added as needed.

    Example:

    In hparams.yaml
    progress_sample_logger: !new:speechbrain.utils.train_logger.ProgressSampleLogger
        output_path: output/samples
        progress_batch_sample_size: 3
        format_defs:
            foo:
                extension: bar
                saver: !speechbrain.dataio.mystuff.save_my_format
                kwargs:
                    baz: qux
        formats:
            foobar: foo



    In the brain:

    Run the following to "remember" a sample (e.g. from compute_objectives)

    self.hparams.progress_sample_logger.remember(
        target=spectrogram_target,
        output=spectrogram_output,
        alignments=alignments_output,
        my_output=
        raw_batch={
            "inputs": inputs,
            "spectrogram_target": spectrogram_target,
            "spectrogram_output": spectrorgram_outputu,
            "alignments": alignments_output
        }
    )

    Run the following at the end of the epoch (e.g. from on_stage_end)
    self.progress_sample_logger.save(epoch)



    Arguments
    ---------
    output_path: str
        the filesystem path to which samples will be saved
    formats: dict
        a dictionary with format identifiers as keys and dictionaries with
        handler callables and extensions as values. The signature of the handler
        should be similar to torch.save

        Example:
        {
            "myformat": {
                "extension": "myf",
                "saver": somemodule.save_my_format
            }
        }
    batch_sample_size: int
        The number of items to retrieve when extracting a batch sample
    """

    _DEFAULT_FORMAT_DEFS = {
        "raw": {"extension": "pth", "saver": torch.save, "kwargs": {}},
        "image": {
            "extension": "png",
            "saver": _get_image_saver(),
            "kwargs": {},
        },
    }
    DEFAULT_FORMAT = "image"

    def __init__(
        self, output_path, formats=None, format_defs=None, batch_sample_size=1
    ):
        self.progress_samples = {}
        self.formats = formats or {}
        self.format_defs = dict(self._DEFAULT_FORMAT_DEFS)
        if format_defs is not None:
            self.format_defs.update(format_defs)
        self.batch_sample_size = batch_sample_size
        self.output_path = output_path

    def reset(self):
        """Initializes the collection of progress samples"""
        self.progress_samples = {}

    def remember(self, **kwargs):
        """Updates the internal dictionary of snapshots with the provided
        values

        Arguments
        ---------
        kwargs: dict
            the parameters to be saved with
        """
        self.progress_samples.update(
            {key: detach(value) for key, value in kwargs.items()}
        )

    def get_batch_sample(self, value):
        """Obtains a sample of a batch for saving. This can be useful to
        monitor raw data (both samples and predictions) over the course
        of training

        Arguments
        ---------
        value: dict|torch.Tensor|list
            the raw values from the batch

        Returns
        -------
        result: object
            the same type of object as the provided value
        """
        if isinstance(value, dict):
            result = {
                key: self.get_batch_sample(item_value)
                for key, item_value in value.items()
            }
        elif isinstance(value, (torch.Tensor, list)):
            result = value[: self.batch_sample_size]
        else:
            result = value
        return result

    def save(self, epoch):
        """Saves all items previously saved with remember() calls

        Arguments
        ---------
        epoch: int
            The epoch number
        """
        for key, data in self.progress_samples.items():
            self.save_item(key, data, epoch)

    def save_item(self, key, data, epoch):
        """Saves a single sample item

        Arguments
        ---------
        key: str
            the key/identifier of the item
        data: torch.Tensor
            the  data to save
        epoch: int
            the epoch number (used in file path calculations)
        """
        target_path = os.path.join(self.output_path, str(epoch))
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        format = self.formats.get(key, self.DEFAULT_FORMAT)
        format_def = self.format_defs.get(format)
        if format_def is None:
            raise ValueError("Unsupported format {format}")
        file_name = f"{key}.{format_def['extension']}"
        effective_file_name = os.path.join(target_path, file_name)
        saver = format_def.get("saver")
        if saver is not None:
            saver(data, effective_file_name, **format_def["kwargs"])


def plot_spectrogram(spectrogram, ap=None, fig_size=(16, 10), output_fig=False):
    """Returns the matplotlib sprctrogram if available
    or None if it is not - optional dependency"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    except ImportError:
        logger.warn("matplotlib is not available - cannot log figures")
        return None

    spectrogram = spectrogram.detach().cpu().numpy().squeeze()
    fig = plt.figure(figsize=fig_size)
    plt.imshow(spectrogram, aspect="auto", origin="lower")
    plt.colorbar()
    plt.tight_layout()
    if not output_fig:
        plt.close()
    return fig


def detach(value):
    """Detaches the specified object from the graph, which can be a
    single tensor or a dictionary of tensors. Dictionaries of tensors are
    converted recursively

    Arguments
    ---------
    value: torch.Tensor|dict
        a tensor or a dictionary of tensors

    Returns
    -------
    result: torch.Tensor|dict
        a tensor of dictionary of tensors
    """
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu()
    elif isinstance(value, dict):
        result = {key: detach(item_value) for key, item_value in value.items()}
    else:
        result = value
    return result


class TensorLogger:
    """A logger that stores a sequence of raw tensors in a binary
    file. One  possible use is to save latent representations.
    
    Arguments
    ---------
    file_name: str
        the file name
    """

    def __init__(self, file_name):
        self.file_name = file_name
        self.tensor_file = None

    def open(self):
        """Opens the file"""
        self.tensor_file = open(self.file_name, "ab+")

    def ensure_open(self):
        """Opens the file if it is not already open"""
        if self.tensor_file is None:
            self.open()

    def append(self, value):
        """Appends a tensor

        Arguments
        ---------
        value: torch.Tensor
            the value to append
        """
        self.ensure_open()
        value_np = value.detach().cpu().numpy()
        np.save(self.tensor_file, value_np)

    def close(self):
        """Closes the file"""
        if self.tensor_file is not None:
            self.tensor_file.close()
            self.tensor_file = None
