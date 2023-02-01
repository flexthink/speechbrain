"""
The MAdmixture model loss

This is the implementation of the MAdmixture multimodal
speech model loss, which requires multiple weighted loss
components to be computed

- Reconstruction losses
- Attention losses
- Latent space structure losses

The approach to tying embeddings is borrowed from the
Amazon Alexa research paper titled Tie Your Embeddings Down

https://arxiv.org/pdf/2011.09044.pdf

Authors
* Artem Ploujnikov 2023
"""

import torch
from torch import nn
from speechbrain.nnet.losses import truncate

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
    anchor: str
        the "anchor" modality used for distance loss calculations,
        i.e. the modality to which latent representations for other
        modalities are compared
    latent_distance_loss_fn: callable
        the distance function used to tie embeddings together
        in latent space
    latent_distance_loss_weight: float
        the relative weight of the latent distance loss
        
    """
    def __init__(
            self,
            modalities,
            rec_loss_fn,
            align_attention_loss_fn=None,
            rec_loss_weight=1.,
            align_attention_loss_weight=0.25,
            modality_weights=None,
            anchor=None,
            latent_distance_loss_fn=None,
            latent_distance_loss_weight=1.
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
        self.anchor = anchor
        self.latent_distance_loss_fn = latent_distance_loss_fn
        self.latent_distance_loss_weight = latent_distance_loss_weight
    

    def forward(self, inputs, length, latents, alignments, rec, transfer_rec, reduction="mean"):
        details = self.details(inputs, length, latents, alignments, rec, transfer_rec, reduction)
        return details["loss"]

    def details(self, inputs, length, latents, alignments, rec, transfer_rec, reduction="mean"):
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
            Options are 'mean', 'batch'
            See pytorch for 'mean' The 'batch' option returns
            one loss per item in the batch


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
        alignment_loss, modality_alignment_loss = self.compute_alignment_loss(
            alignments, length, reduction
        )
        modality_alignment_loss_details = self._modality_expand(
            "align",
            modality_alignment_loss
        )
        loss_details.update(modality_alignment_loss_details)

        latent_distance_loss, modality_distance_loss = self.compute_latent_distance_loss(
            latents=latents,
            lengths=length,
            reduction=reduction
        )
        modality_distance_loss_details = self._modality_expand(
            "distance",
            modality_distance_loss
        )
        loss_details.update(modality_distance_loss_details)
        if transfer_rec is not None:
            transfer_loss, modality_transfer_loss, weighted_modality_transfer_loss = self.compute_transfer_loss(
                inputs=inputs,
                transfer_rec=transfer_rec, 
                lengths=length,
                reduction=reduction
            )
            modality_transfer_loss_details = self._modality_expand_transfer(
                "transfer", modality_transfer_loss
            )
            weighted_modality_transfer_loss_details = self._modality_expand_transfer(
                "transfer_weighted", weighted_modality_transfer_loss
            )
            loss_details.update(modality_transfer_loss_details)
            loss_details.update(weighted_modality_transfer_loss_details)
        else:
            transfer_loss = torch.tensor(0.).to(rec_loss.device)

        loss = (
            self.rec_loss_weight * rec_loss 
            + self.align_attention_loss_weight * alignment_loss
            + self.latent_distance_loss_weight * latent_distance_loss
            + transfer_loss
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
    
    def _modality_expand_transfer(self, prefix, loss_dict):
        """A version of _modality_expand for the transfer
        loss
        
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
            f"{prefix}_{src}_to_{tgt}_loss": value
            for (src, tgt), value in loss_dict.items()
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
    

    def compute_transfer_loss(self, inputs, transfer_rec, lengths, reduction="mean"):
        modality_transfer_loss = {
            (src, tgt): self.rec_loss_fn[tgt](
                modality_transfer_rec,
                inputs[tgt],
                length=lengths[tgt],
                reduction=reduction
            )
            for (src, tgt), modality_transfer_rec 
            in transfer_rec.items()
        }
        weighted_modality_transfer_loss = {
            (src, tgt): self.modality_weights[tgt] * loss
            for (src, tgt), loss in modality_transfer_loss.items()
        }
        transfer_loss = torch.stack(
            list(weighted_modality_transfer_loss.values())
        ).sum(dim=0)
        return transfer_loss, modality_transfer_loss, weighted_modality_transfer_loss
        

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
        """Computes the loss on alignment matrices output by aligner modules,
        such as a guided attention loss
        
        Arguments
        ---------
        alignments: dict
            an str -> tensor dictionary with alignments for all aligned
            modalities
        
        lengths: dict
            an str -> tensor dictionary with relative lengths
            for all aligned modalities
        
        reduction: str
            reduction mode: "batch" or "mean"
            when set to "mean", a single value is returned
            when set to "batch", a value is returned for each batch
            
        Returns
        -------
        alignment_loss: torch.Tensor
            the total alignment loss
        modality_alignment_loss: dict
            an str -> tensor dictionary with alignment loss values
            for all aligned modalities

        """
        if self.align_attention_loss_fn is None:
            # NOTE: In this case, there is nothing to compute, the loss is 0
            first_alignment = next(iter(alignments.values()))
            return torch.tensor(0.).to(first_alignment.device), {}
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
    
    def compute_latent_distance_loss(self, latents, lengths, reduction="mean"):
        """Computes a distance loss across modalities, which is
        used to ensure that the latent representations of a single
        sample are similar in different modalities
        
        Arguments
        ---------
        latents: torch.Tensor
            an str -> tensor dictionary with latent representations
            for each modality
        lengths: dict
            an str -> tensor dictionary with relative lengths
            for all aligned modalities
        """
        if self.latent_distance_loss_fn is None:
            # NOTE: In this case, there is nothing to compute, the loss is 0
            first_alignment = next(iter(latents.values()))
            return torch.tensor(0.).to(first_alignment.device), {}

        anchor_latent = latents[self.anchor]
        anchor_length = lengths[self.anchor]

        sec_latent = dict(latents)
        sec_lengths = dict(lengths)
        del sec_latent[self.anchor]
        del sec_lengths[self.anchor]

        modality_latent_distance_loss = {
            key: self.compute_modality_latent_distance_loss(
                anchor_latent, anchor_length, sec_latent[key], sec_lengths[key],
                reduction=reduction)
            for key in sec_latent            
        }
        total_loss = torch.stack(
            list(modality_latent_distance_loss.values())
        ).sum(dim=0)

        return total_loss, modality_latent_distance_loss

    def compute_modality_latent_distance_loss(
        self,
        anchor_latent,
        anchor_length,
        sec_latent,
        sec_length,
        reduction
    ):
        # It is important to ensure the lentgths match, and only
        # unpadded locations are used for loss calculations
        # Find the max length of anchor and secondary modalities
        sec_latent_trunc, anchor_latent_trunc = truncate(
            sec_latent, anchor_latent, torch.inf
        )
        anchor_max_length = anchor_latent_trunc.size(1)
        sec_max_length = sec_latent.size(1)
        max_len_trunc = sec_latent_trunc.size(1)
        # Recalculate the lengths post-truncation
        length = torch.minimum(
            anchor_length * anchor_max_length / max_len_trunc, 
            sec_length * sec_max_length / max_len_trunc
        ).clamp(0., 1.)
        return self.latent_distance_loss_fn(
            sec_latent_trunc,
            anchor_latent_trunc,
            length=length,
            reduction=reduction
        )
