import torch


def test_madmixture_basic():
    from speechbrain.lobes.models.madmixture import (
        MadMixture,
        Modality,
        AttentionalAligner,
    )
    from speechbrain.nnet.CNN import Conv1d, ConvTranspose1d
    from speechbrain.nnet.embedding import Embedding
    from speechbrain.nnet.containers import LengthsCapableSequential

    mock_text_emb = Embedding(26, 32)
    mock_text_conv = Conv1d(
        in_channels=32, out_channels=16, kernel_size=3, padding="same",
    )

    mock_text_enc = LengthsCapableSequential(mock_text_emb, mock_text_conv)
    mock_text_dec = ConvTranspose1d(
        in_channels=16, out_channels=32, kernel_size=3, padding=1,
    )
    latent_size = 16

    mock_audio_enc = LengthsCapableSequential(
        Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding="same",),
        Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding="same",),
    )

    mock_audio_dec = LengthsCapableSequential(
        ConvTranspose1d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1,
        ),
        ConvTranspose1d(
            in_channels=32, out_channels=1, kernel_size=3, padding=1,
        ),
    )
    mock_audio_align = AttentionalAligner(dim=16, scale=0.5)

    madmixture = MadMixture(
        modalities=[
            Modality(name="text", encoder=mock_text_enc, decoder=mock_text_dec),
            Modality(
                name="audio",
                encoder=mock_audio_enc,
                decoder=mock_audio_dec,
                aligner=mock_audio_align,
            ),
        ],
        anchor_name="audio",
        latent_size=latent_size,
    )

    mock_text = torch.tensor([[2, 1, 4, 0], [3, 1, 20, 19]])
    mock_audio = torch.randn(2, 200, 1)
    mock_text_length = torch.tensor([1.0, 1.0])
    mock_audio_length = torch.tensor([1.0, 1.0])

    inputs = {"text": mock_text, "audio": mock_audio}
    length = {"text": mock_text_length, "audio": mock_audio_length}
    latents, alignments, enc_out = madmixture.latent(inputs, length)
    assert len(latents) == 2
    assert set(latents.keys()) == {"text", "audio"}

    # NOTE: Ensure all latent representations are encoded into
    # tensors with the correct / expected shape
    for modality_latent in latents.values():
        batch_size, _, feature_size = modality_latent.shape
        assert batch_size == 2
        assert feature_size == 16

    assert set(modality_latent.keys()) == {"text", "audio"}
    assert set(enc_out.keys()) == {"text", "audio"}
