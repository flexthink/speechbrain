"""
The MAdmixture model loss

This is the implementation of the MAdmixture multimodal
speech model loss, which requires multiple weighted loss
components to be computed

- Reconstruction losses
- Attention losses
- Latent space structure losses

Authors
* Artem Ploujnikov 2023
"""

import torch
from torch import nn

class MadMixtureLoss(nn.Module):
    """The MAdmixture model loss

    Arguments
    ---------
    modalities: list
        a list of modality keys
    rec_loss_weight: float
        the weight of the sum of recreation losses
    rec_loss_fn: dict
        the reconstruction loss function to be used for
        each modality
    align_attention_loss_weight: float
        the weight of the alignment attention loss
    modality_weights: float
        the weights of individual modalities. If ommitted,
        modalities will be equally weighted
    """
    def __init__(
            self,
            modalities,
            rec_loss_fn,
            rec_loss_weight=1.,
            align_attention_loss_weight=0.25,
            modality_weights=None,
        ):
        super().__init__()
        self.modalities = modalities
        self.rec_loss_weight = rec_loss_weight
        self.rec_loss_fn = rec_loss_fn
        self.align_attention_loss_weight = align_attention_loss_weight
        if modality_weights is None:
            modality_weights = {key: 1. for key in modalities}
        self.modality_weights = modality_weights

    def forward(self, inputs, length, latents, alignments, rec):
        details = self.details(inputs, length, latents, alignments, rec)
        return details["loss"]

    def details(self, inputs, length, latents, alignments, rec):
        """Computes the MadMixture loss with the detailed breakdown
        of all loss components
        
        Arguments
        ---------
        inputs: dict
            the orignal inputs to the MadMixture model
        length: dict
            the length tensor
        latents: dict
            latent representation
        alignments: dict
            alignments
        rec: dict
            reconstructions in each modality

        Returns
        -------
        loss_details: dict
            a complete breakdown of each loss, per modality, useful
            for tracking in Tensorboard, etc
        """
        rec_loss, modality_rec_loss, weighted_modality_weight_loss = (
            self.compute_rec_loss(inputs, rec, length)
        )
        loss_details = {
            "rec_loss": rec_loss
        }
        modality_rec_details = self._modality_expand(
            "rec", modality_rec_loss)
        modality_rec_weighted_details = self._modality_expand(
            "rec_weighted", weighted_modality_weight_loss)
        loss_details.update(modality_rec_details)
        loss_details.update(modality_rec_weighted_details)
        loss = self.rec_loss_weight * rec_loss
        loss_details["loss"] = loss
        return loss_details
    
    def _modality_expand(self, prefix, loss_dict):
        return {
            f"{prefix}_{key}_loss": value
            for key, value in loss_dict.items()
        }
    
    def compute_rec_loss(self, inputs, rec, lengths):
        """Computes the recreation losses
        
        Arguments
        ---------
        inputs: dict
            modality inputs
        rec: dict
            modality reconstructions
        lengths: dict
            relative lengths in each modality
        
        Returns
        -------
        rec_loss: dict
            the total reconstruction loss
        modality_rec_loss: dict
            the raw outputs of the loss functions
            for each modality
        weighted_modality_rec_loss: dict
            the modality reconstruction losses, with
            weights applied
        """
        modality_rec_loss = {
            key: self.rec_loss_fn[key](
                rec[key],            
                inputs[key],
                length=lengths[key],
            )
            for key in self.modalities
        }
        weighted_modality_losses = {
            key: self.modality_weights[key] * loss_value
            for key, loss_value in modality_rec_loss.items()
        }
        rec_loss = torch.stack(
            list(weighted_modality_losses.values())
        ).sum()
        return rec_loss, modality_rec_loss, weighted_modality_losses
