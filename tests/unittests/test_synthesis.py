from speechbrain.pretrained.interfaces import GraphemeToPhoneme
import torch
from speechbrain.pretrained import SpeechSynthesizer
from speechbrain.utils.data_pipeline import takes, provides
from speechbrain.dataio.encoder import TextEncoder
from torch import nn

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ.,!-"

TEST_HPARAMS = """
model: !new:speechbrain.lobes.models.synthesis.framework.TestModel
model_input_keys: ['input']
model_output_keys: ['output']
"""


class TestSynthesisModel(nn.Module):
    def forward(self, input):
        return input + 1.0


class TestG2PModel(nn.Module):
    def forward(self, input):
        return torch.tensor(
            [
                [[0.9, 0.05, 0.05], [0.1, 0.2, 0.7], [0.2, 0.6, 0.2]],
                [[0.1, 0.1, 0.8], [0.1, 0.7, 0.2], [0.7, 0.1, 0.2]],
            ]
        )


def test_synthesizer():
    # Note: the unit test is done with a fake model
    encoder = TextEncoder()
    encoder.update_from_iterable(ALPHABET)
    encoder.add_unk()

    @takes("txt")
    @provides("txt_encoded")
    def encode_text(txt):
        return encoder.encode_sequence_torch(txt.upper())

    @takes("txt_encoded")
    @provides("input")
    def model_input(txt_encoded):
        print(txt_encoded)
        assert txt_encoded.size(-1) == 4
        return torch.tensor([[1.0, 2.0, 3.0]])

    @takes("output")
    @provides("wav")
    def decode_waveform(model_output):
        return model_output + torch.tensor([1.0, 2.0, 3.0])

    test_hparams = {
        "model": TestSynthesisModel(),
        "encode_pipeline": {
            "batch": False,
            "steps": [encode_text, model_input],
            "output_keys": ["input"],
        },
        "decode_pipeline": {"steps": [decode_waveform]},
        "model_input_keys": ["input"],
        "model_output_keys": ["output"],
    }

    synthesizer = SpeechSynthesizer(hparams=test_hparams)
    output = synthesizer("test")
    assert torch.isclose(output, torch.tensor([3.0, 5.0, 7.0])).all()


def test_g2p():
    encoder = TextEncoder()
    encoder.update_from_iterable(ALPHABET)

    phonemes = ["EY", "BEE", "SEE"]

    @takes("txt")
    @provides("txt_encoded")
    def encode_text(txt):
        return encoder.encode_sequence_torch(txt.upper())

    @takes("txt_encoded")
    @provides("input")
    def model_input(txt_encoded):
        assert txt_encoded.size(-1) == 3
        return torch.tensor([[1.0, 2.0, 3.0]])

    @takes("output")
    @provides("phonemes")
    def decode_g2p(model_output):
        indices = model_output.argmax(dim=-1)
        return [[phonemes[index] for index in batch] for batch in indices]

    test_hparams = {
        "model": TestG2PModel(),
        "phonemes": phonemes,
        "encode_pipeline": {
            "batch": False,
            "steps": [encode_text, model_input],
            "output_keys": ["input"],
        },
        "decode_pipeline": {"steps": [decode_g2p], "batch": True},
        "model_input_keys": ["input"],
        "model_output_keys": ["output"],
    }
    g2p = GraphemeToPhoneme(hparams=test_hparams)
    assert g2p.phonemes == ["EY", "BEE", "SEE"]
    result = g2p(["ACB", "CBA"])
    ref_phonemes = [["EY", "SEE", "BEE"], ["SEE", "BEE", "EY"]]
    assert ref_phonemes == result
