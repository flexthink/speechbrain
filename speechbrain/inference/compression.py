"""Pretrained inference interfaces for compression models

Authors
 * Artem Ploujnikov 2024
"""

from speechbrain.inference.interfaces import Pretrained


class TokenSquish(Pretrained):
    def compress(self, tokens, length=None):
        """Compresses the token sequence

        Arguments
        ---------
        tokens : torch.Tensor
            A token sequence
        length : torch.Tensor, optional
            Relative lengths. They are not required because the implementation simply
            leaves every scale_factorth token, removing the others. It is provided in
            for future compatibility with other approaches

        Result
        ------
        compressed_tokens : torch.Tensor
            The compressed tensor
        """
        return self.mods.model.compress(tokens, length)

    def decompress(self, tokens, length=None):
        """Decompresses a compressed token sequence

        Arguments
        ---------
        tokens : torch.Tensor
            A compressed token sequence
        length : torch.Tensor, optional
            Relative lengths.

        Result
        ------
        decompressed_tokens : torch.Tensor
            The decompressed token sequence
        """
        return self.mods.model.decompress(tokens, length)
