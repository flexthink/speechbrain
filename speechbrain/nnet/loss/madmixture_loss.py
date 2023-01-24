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
    align_attention_loss_fn: callable
        the loss function used to compute alignment losses
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
            align_attention_loss_fn=None,
            rec_loss_weight=1.,
            align_attention_loss_weight=0.25,
            modality_weights=None,
        ):
        super().__init__()
        self.modalities = modalities
        self.rec_loss_weight = rec_loss_weight
        self.rec_loss_fn = rec_loss_fn
        self.align_attention_loss_fn = align_attention_loss_fn
        self.align_attention_loss_weight = align_attention_loss_weight
        if modality_weights is None:
            modality_weights = {key: 1. for key in modalities}
        self.modality_weights = modality_weights

    def forward(self, inputs, length, latents, alignments, rec, reduction="mean"):
        details = self.details(inputs, length, latents, alignments, rec, reduction)
        return details["loss"]

    def details(self, inputs, length, latents, alignments, rec, reduction="mean"):
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
        reduction : str
            Options are 'mean', 'batch', 'batchmean', 'sum'.
            See pytorch for 'mean', 'sum'. The 'batch' option returns
            one loss per item in the batch, 'batchmean' returns sum / batch size.


        Returns
        -------
        loss_details: dict
            a complete breakdown of each loss, per modality, useful
            for tracking in Tensorboard, etc
        """
        rec_loss, modality_rec_loss, weighted_modality_rec_loss = (
            self.compute_rec_loss(inputs, rec, length, reduction)
        )
        loss_details = {
            "rec_loss": rec_loss
        }
        modality_rec_details = self._modality_expand(
            "rec", modality_rec_loss
        )
        modality_rec_weighted_details = self._modality_expand(
            "rec_weighted",
            weighted_modality_rec_loss
        )
        loss_details.update(modality_rec_details)
        loss_details.update(modality_rec_weighted_details)
        if self.align_attention_loss_fn is not None:
            alignment_loss, modality_alignment_loss = self.compute_alignment_loss(
                alignments, length, reduction
            )
            modality_alignment_loss_details = self._modality_expand(
                "align",
                modality_alignment_loss
            )
            loss_details.update(modality_alignment_loss_details)
            
        else:
            alignment_loss = 0.
        loss = (
            self.rec_loss_weight * rec_loss 
            + self.align_attention_loss_weight * alignment_loss
        )
        loss_details["loss"] = loss
        return loss_details
    
    def _modality_expand(self, prefix, loss_dict):
        """Adds a modality prefix to the specified loss dictionary
        
        Arguments
        ---------
        prefix: str
            the modality prefix
        loss_dict
            the raw loss values for the modality
            
        Arguments
        ---------
        result: dict
            a new dictionary with keys rewritten as "{prefix}_{key}_loss
        """
        return {
            f"{prefix}_{key}_loss": value
            for key, value in loss_dict.items()
        }
    
    def compute_rec_loss(self, inputs, rec, lengths, reduction="mean"):
        """Computes the recreation losses
        
        Arguments
        ---------
        inputs: dict
            modality inputs
        rec: dict
            modality reconstructions
        lengths: dict
            relative lengths in each modality
        reduction : str
            Options are 'mean', 'batch', 'batchmean', 'sum'.
            See pytorch for 'mean', 'sum'. The 'batch' option returns
            one loss per item in the batch, 'batchmean' returns sum / batch size.
        
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
                reduction=reduction
            )
            for key in self.modalities
        }
        rec_loss, weighted_modality_rec_loss = self._weighted_modality_loss(
            modality_rec_loss
        )
        return rec_loss, modality_rec_loss, weighted_modality_rec_loss
    
    def _weighted_modality_loss(self, modality_loss):
        weighted_modality_loss = {
            key: self.modality_weights[key] * loss_value
            for key, loss_value in modality_loss.items()
        }
        total_loss = torch.stack(
            list(weighted_modality_loss.values())
        ).sum(dim=0)
        return total_loss, weighted_modality_loss


    def compute_alignment_loss(self, alignments, lengths, reduction):
        alignments_with_lengths = [
            (key, alignment, alignment.size(2) * lengths[key], alignment.size(1) * lengths[key])
            for key, alignment in alignments.items()
        ]
        modality_alignment_loss = {
            key: self.align_attention_loss_fn(
                attention=modality_alignment, 
                input_lengths=input_lengths,
                target_lengths=target_lengths,
                reduction=reduction
            )
            for key, modality_alignment, input_lengths, target_lengths
            in alignments_with_lengths
        }
        alignment_loss, modality_alignment_loss = self._weighted_modality_loss(
            modality_alignment_loss
        )
        return alignment_loss, modality_alignment_loss
