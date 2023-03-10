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
from torch.nn import functional as F
from speechbrain.nnet.losses import truncate, distance_diff_loss, reduce_loss
from speechbrain.utils.data_utils import concat_padded_features
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.nnet.loss.guidedattn_loss import GuidedAttentionLoss


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
    context_loss_fn: callable
        the loss function to apply to the context
    length_loss_wight: float
        the relative weight of the length loss
    length_loss_fn: callable
        the length loss function or module
    eos_loss_fn: callable
        the end-of-sequence loss function
    eos_loss_weight: int
        the relative weight of the end-of-sequence loss
    modality_enabled: dict
        a str->bool dictionary indicating which modalities
        are enabled
    """

    def __init__(
        self,
        modalities,
        rec_loss_fn,
        align_attention_loss_fn=None,
        rec_loss_weight=1.0,
        align_attention_loss_weight=0.25,
        transfer_loss_weight=1.0,
        modality_weights=None,
        anchor=None,
        latent_distance_loss_fn=None,
        latent_distance_loss_weight=1.0,
        context_loss_weight=1.0,
        context_loss_fn=None,
        length_loss_weight=1.0,
        length_loss_fn=None,
        eos_loss_fn=None,
        eos_loss_weight=1.0,
        modality_enabled=None,
    ):
        super().__init__()
        self.modalities = modalities
        self.modality_enabled = modality_enabled
        if modality_enabled is not None:
            self.modalities = [
                mod for mod in self.modalities if modality_enabled[mod]
            ]
        self.rec_loss_weight = rec_loss_weight
        self.rec_loss_fn = rec_loss_fn
        self.align_attention_loss_fn = align_attention_loss_fn
        self.align_attention_loss_weight = align_attention_loss_weight
        if modality_weights is None:
            modality_weights = {key: 1.0 for key in modalities}
        self.modality_weights = modality_weights
        self.anchor = anchor
        self.latent_distance_loss_fn = latent_distance_loss_fn
        self.latent_distance_loss_weight = latent_distance_loss_weight
        self.transfer_loss_weight = transfer_loss_weight
        self.context_loss_weight = context_loss_weight
        self.context_loss_fn = context_loss_fn
        self.length_loss_weight = length_loss_weight
        self.length_loss_fn = length_loss_fn
        self.eos_loss_fn = eos_loss_fn
        self.eos_loss_weight = eos_loss_weight

    def forward(
        self,
        targets,
        length,
        latents,
        alignments,
        rec,
        transfer_rec,
        out_context=None,
        length_preds=None,
        length_latent=None,
        length_input=None,
        latents_neg=None,
        length_latent_neg=None,
        reduction="mean",
    ):
        """Computes the MadMixture loss with the detailed breakdown
        of all loss components
        
        Arguments
        ---------
        targets: dict
            the targets to which reconstructions will be compared
        length: dict
            the length tensor
        latents: dict
            latent representation
        alignments: dict
            alignments
        rec: dict
            a str->tensor dictionary with reconstructions in each modality 
        transfer_rec: dict
            a (str,str)->tensor dictionary with transfer reconstructions
        length_preds: dict
            a str->tensor dictionary with length predictions for each modality
        length_input: dict
            a str->tensor dictionary with ground truths for lengths
        latents_neg: dict
            a str->tensor dictionary for negative examples
        length_latent_neg: dict
            negative example lengths
        reduction : str
            Options are 'mean', 'batch'
            See pytorch for 'mean' The 'batch' option returns
            one loss per item in the batch

        Returns
        -------
        loss: torch.Tensor
            the total loss value
        """
        details = self.details(
            targets,
            length,
            latents,
            alignments,
            rec,
            transfer_rec,
            out_context,
            length_preds,
            length_latent,
            length_input,
            latents_neg,
            length_latent_neg,
            reduction,
        )
        return details["loss"]

    def details(
        self,
        targets,
        length,
        latents,
        alignments,
        rec,
        transfer_rec,
        out_context=None,
        length_preds=None,
        length_latent=None,
        length_input=None,
        latents_neg=None,
        length_latent_neg=None,
        reduction="mean",
    ):
        """Computes the MadMixture loss with the detailed breakdown
        of all loss components
        
        Arguments
        ---------
        targets: dict
            the targets to which reconstructions will be compared
        length: dict
            the length tensor
        latents: dict
            latent representation
        alignments: dict
            alignments
        rec: dict
            a str->tensor dictionary with reconstructions in each modality 
        transfer_rec: dict
            a (str,str)->tensor dictionary with transfer reconstructions
        length_preds: dict
            a str->tensor dictionary with length predictions for each modality
        length_input: dict
            a str->tensor dictionary with ground truths for lengths
        latents_neg: dict
            a str->tensor dictionary for negative examples
        length_latent_neg: dict
            negative example lengths
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
        (
            rec_loss,
            modality_rec_loss,
            weighted_modality_rec_loss,
        ) = self.compute_rec_loss(targets, rec, length, reduction)
        loss_details = {"rec_loss": rec_loss}
        modality_rec_details = with_prefix("rec", modality_rec_loss)
        modality_rec_weighted_details = with_prefix(
            "rec_weighted", weighted_modality_rec_loss
        )
        loss_details.update(modality_rec_details)
        loss_details.update(modality_rec_weighted_details)
        alignment_loss, modality_alignment_loss = self.compute_alignment_loss(
            alignments, length_latent, length_input, reduction
        )
        modality_alignment_loss_details = with_prefix(
            "align", modality_alignment_loss
        )
        loss_details.update(modality_alignment_loss_details)

        (
            latent_distance_loss,
            modality_distance_loss,
        ) = self.compute_latent_distance_loss(
            latents=latents,
            lengths=length_latent,
            latents_neg=latents_neg,
            lengths_latent_neg=length_latent_neg,
            reduction=reduction,
        )
        modality_distance_loss_details = with_prefix(
            "distance", modality_distance_loss
        )
        loss_details.update(modality_distance_loss_details)
        if transfer_rec is not None:
            (
                transfer_loss,
                modality_transfer_loss,
                weighted_modality_transfer_loss,
            ) = self.compute_transfer_loss(
                targets=targets,
                transfer_rec=transfer_rec,
                lengths=length,
                reduction=reduction,
            )
            modality_transfer_loss_details = with_prefix_transfer(
                "transfer", modality_transfer_loss
            )
            weighted_modality_transfer_loss_details = with_prefix_transfer(
                "transfer_weighted", weighted_modality_transfer_loss
            )
            loss_details.update(modality_transfer_loss_details)
            loss_details.update(weighted_modality_transfer_loss_details)
        else:
            transfer_loss = torch.tensor(0.0).to(rec_loss.device)

        if self.context_loss_fn is not None:
            (
                context_loss,
                component_context_loss,
                weighted_context_loss,
            ) = self.context_loss_fn(
                out_context=out_context,
                latent_lengths=length_latent,
                target_lengths=length,
                reduction=reduction,
            )
            context_loss_details = with_prefix(
                "context", component_context_loss
            )
            weighted_context_loss_details = with_prefix(
                "context_weighted", weighted_context_loss
            )

            loss_details.update(context_loss_details)
            loss_details.update(weighted_context_loss_details)
        else:
            context_loss = torch.tensor(0.0).to(rec_loss.device)

        if self.length_loss_fn is not None and length_preds is not None:
            (
                length_loss,
                modality_length_loss,
                weighted_modality_length_loss,
            ) = self.compute_length_loss(
                length_preds, latents, length, reduction=reduction
            )
            length_loss_details = with_prefix("length", modality_length_loss)
            weighted_length_loss_details = with_prefix(
                "length_weighted", weighted_modality_length_loss
            )
            loss_details.update(length_loss_details)
            loss_details.update(weighted_length_loss_details)
            loss_details["length_loss"] = length_loss
        else:
            length_loss = torch.tensor(0.0).to(rec_loss.device)

        if self.eos_loss_fn is not None:
            (
                eos_loss,
                modality_eos_loss,
                weighted_modality_eos_loss,
            ) = self.compute_eos_loss(latents, length_latent, reduction)
            eos_loss_details = with_prefix("eos", modality_eos_loss)
            weighted_eos_loss_details = with_prefix(
                "weighted_eos", weighted_modality_eos_loss
            )
            loss_details.update(eos_loss_details)
            loss_details.update(weighted_eos_loss_details)
        else:
            eos_loss = torch.tensor(0.0).to(rec_loss.device)

        loss = (
            self.rec_loss_weight * rec_loss
            + self.align_attention_loss_weight * alignment_loss
            + self.latent_distance_loss_weight * latent_distance_loss
            + self.transfer_loss_weight * transfer_loss
            + self.context_loss_weight * context_loss
            + self.length_loss_weight * length_loss
            + self.eos_loss_weight * eos_loss
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
            f"{prefix}_{key}_loss": value for key, value in loss_dict.items()
        }

    def compute_rec_loss(self, targets, rec, lengths, reduction="mean"):
        """Computes the recreation losses
        
        Arguments
        ---------
        targets: dict
            modality targets
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
                rec[key], targets[key], length=lengths[key], reduction=reduction
            )
            for key in self.modalities
        }
        rec_loss, weighted_modality_rec_loss = self._weighted_modality_loss(
            modality_rec_loss
        )
        return rec_loss, modality_rec_loss, weighted_modality_rec_loss

    def compute_transfer_loss(
        self, targets, transfer_rec, lengths, reduction="mean"
    ):
        modality_transfer_loss = {
            (src, tgt): self.rec_loss_fn[tgt](
                modality_transfer_rec,
                targets[tgt],
                length=lengths[tgt],
                reduction=reduction,
            )
            for (src, tgt), modality_transfer_rec in transfer_rec.items()
        }
        weighted_modality_transfer_loss = {
            (src, tgt): self.modality_weights[tgt] * loss
            for (src, tgt), loss in modality_transfer_loss.items()
        }
        if modality_transfer_loss:
            transfer_loss = torch.stack(
                list(weighted_modality_transfer_loss.values())
            ).sum(dim=0)
        else:
            first_target = next(iter(targets.values()))
            transfer_loss = torch.tensor(0.0, device=first_target.device)
        return (
            transfer_loss,
            modality_transfer_loss,
            weighted_modality_transfer_loss,
        )

    def _weighted_modality_loss(self, modality_loss):
        return weighted_loss(modality_loss, self.modality_weights)

    def compute_alignment_loss(
        self, alignments, lengths_latent, lengths_input, reduction
    ):
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

        lengths_latent: dict
            an str -> tensor dictionary with absolute lengths of
            the latent dimension

        lengths_input: dict
            an str -> tensor dictionary with absolute lengths of
            the attention inputs
        
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
            return torch.tensor(0.0).to(first_alignment.device), {}
        alignments_with_lengths = [
            (key, alignment, lengths_input[key], lengths_latent[key])
            for key, alignment in alignments.items()
        ]
        modality_alignment_loss = {
            key: self.align_attention_loss_fn(
                attention=mod_alignment,
                input_lengths=mod_length_input.float(),
                target_lengths=mod_length_target.float(),
                reduction=reduction,
            )
            for key, mod_alignment, mod_length_input, mod_length_target in alignments_with_lengths
        }
        alignment_loss, modality_alignment_loss = self._weighted_modality_loss(
            modality_alignment_loss
        )
        return alignment_loss, modality_alignment_loss

    def compute_latent_distance_loss(
        self,
        latents,
        lengths,
        latents_neg,
        lengths_latent_neg,
        reduction="mean",
    ):
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
        latents_neg: dict
            an str -> tensor dictionary with negative example latents
        lengths_latent_neg: dict
            an str->tensor dictionary with negative example length
        reduction : str
            Options are 'mean', 'batch', ''sum'.
        """
        if self.latent_distance_loss_fn is None:
            # NOTE: In this case, there is nothing to compute, the loss is 0
            first_alignment = next(iter(latents.values()))
            return torch.tensor(0.0).to(first_alignment.device), {}

        anchor_latent = latents[self.anchor]
        anchor_length = lengths[self.anchor]

        sec_latent = dict(latents)
        sec_lengths = dict(lengths)
        del sec_latent[self.anchor]
        del sec_lengths[self.anchor]
        if not sec_latent:
            first_alignment = next(iter(latents.values()))
            return torch.tensor(0.0).to(first_alignment.device), {}

        if latents_neg is None:
            modality_latent_distance_loss = {
                key: self.compute_modality_latent_distance_loss(
                    anchor_latent,
                    anchor_length,
                    sec_latent[key],
                    sec_lengths[key],
                    reduction=reduction,
                )
                for key in sec_latent
            }
        else:
            modality_latent_distance_loss = {
                key: self.compute_modality_latent_distance_loss_with_neg(
                    anchor_latent,
                    anchor_length,
                    sec_latent[key],
                    sec_lengths[key],
                    latents_neg[key],
                    lengths_latent_neg[key],
                    reduction=reduction,
                )
                for key in sec_latent
            }
        total_loss = torch.stack(
            list(modality_latent_distance_loss.values())
        ).sum(dim=0)

        return total_loss, modality_latent_distance_loss

    def compute_modality_latent_distance_loss(
        self, anchor_latent, anchor_length, sec_latent, sec_length, reduction
    ):
        """Computes the latent distance loss for a single modality - used for 2-point
        losses, such as MSE or cosine distance
        
        Arguments
        ---------
        anchor_latent: torch.Tensor
            the latent representation in the anchor modality
        anchor_length: torch.Tensor
            absolute latent lengths in the anchor modality
        sec_latent: torch.Tensor
            the latent representation in a secondary modality
        sec_length: torch.Tensor
            absolute latent lengths in a secondary modality
        reduction: str
            Options are 'mean', 'batch', ''sum'.
        """
        sec_latent_trunc, anchor_latent_trunc = truncate(
            sec_latent, anchor_latent, torch.inf
        )
        anchor_max_length = anchor_latent_trunc.size(1)
        sec_max_length = sec_latent.size(1)
        max_len_trunc = sec_latent_trunc.size(1)
        # Recalculate the lengths post-truncation
        length = torch.minimum(
            anchor_length * anchor_max_length / max_len_trunc,
            sec_length * sec_max_length / max_len_trunc,
        )
        return self.latent_distance_loss_fn(
            sec_latent_trunc,
            anchor_latent_trunc,
            length=length,
            reduction=reduction,
        )

    def compute_modality_latent_distance_loss_with_neg(
        self,
        anchor_latent,
        anchor_length,
        sec_latent,
        sec_length,
        sec_latent_neg,
        sec_length_neg,
        reduction,
    ):
        """Computes the latent distance loss for a single modality - used for 2-point
        losses, such as MSE or cosine distance
        
        Arguments
        ---------
        anchor_latent: torch.Tensor
            the latent representation in the anchor modality
        anchor_length: torch.Tensor
            absolute latent lengths in the anchor modality
        sec_latent: torch.Tensor
            the latent representation in a secondary modality
        sec_length: torch.Tensor
            absolute latent lengths in a secondary modality
        reduction: str
            Options are 'mean', 'batch', ''sum'.
        """
        sec_latent_trunc, anchor_latent_trunc = truncate(
            sec_latent, anchor_latent, torch.inf
        )
        anchor_max_length = anchor_latent_trunc.size(1)
        sec_max_length = sec_latent.size(1)
        max_len_trunc = sec_latent_trunc.size(1)
        # Recalculate the lengths post-truncation
        length = torch.minimum(
            anchor_length * anchor_max_length / max_len_trunc,
            sec_length * sec_max_length / max_len_trunc,
        )
        sec_neg_len_match = tile_to_length(
            batch=sec_latent_neg, lengths=sec_length_neg, target_lengths=length
        )
        return self.latent_distance_loss_fn(
            anchor=anchor_latent_trunc,
            positive=sec_latent_trunc,
            negative=sec_neg_len_match,
            length=length / sec_max_length,
            reduction=reduction,
        )

    def compute_length_loss(self, length_preds, latents, length, reduction):
        """Computes the length loss
        
        Arguments
        ---------
        length_preds: dict
            length predictions for each modality
        latents: dict
            latent representations for each modality
        length: dict
            relative length for each modality
        reduction: str
            the loss reduction method

        Returns
        -------
        length_loss: torch.Tensor
            the total length loss
        modality_length_loss: dict
            the length loss, broken down by modality
        weighted_modality_length_loss: dict
            the weighted length loss, broken down by modality
        """
        modality_length_loss = {
            key: self.length_loss_fn(
                length_preds[key],
                latents[key],
                length[self.anchor],
                reduction=reduction,
            )
            for key in length_preds
        }
        (
            length_loss,
            weighted_modality_length_loss,
        ) = self._weighted_modality_loss(modality_length_loss)
        return length_loss, modality_length_loss, weighted_modality_length_loss

    def compute_eos_loss(self, latents, latent_lengths, reduction="mean"):
        modality_eos_loss = {
            key: self.eos_loss_fn(latents[key], latent_lengths[key], reduction)
            for key in latents
        }
        eos_loss, weighted_modality_eos_loss = self._weighted_modality_loss(
            modality_eos_loss
        )
        return eos_loss, modality_eos_loss, weighted_modality_eos_loss


class ContextAlignmentLoss(nn.Module):
    """A module that applies an alignment loss (e.g. guided
    attention) to the alignment matrices found in a context
    """

    def __init__(self, loss_fn, keys, anchor, weights=None):
        super().__init__()
        self.loss_fn = loss_fn
        self.keys = keys
        if weights is None:
            weights = {key: 1.0 for key in keys}
        self.weights = weights
        self.anchor = anchor

    def forward(
        self, out_context, latent_lengths, target_lengths, reduction="mean"
    ):
        """Computes the alignment loss
        
        Arguments
        ---------
        out_context: dict
            output context keys containing alignments
        latent_lengths: dict
            sequence lengths in the latent space
        target_lengtsh: dict
            target lengths
        reduction: str
            reduction type

        Returns
        -------
        context_loss: torch.Tensor
            the fully aggregated context loss
        component_content_loss: torch.Tensor
            the context loss, broken down by component
        weighted_conetxt_loss: torch.Tensor
            the context loss, weighted according to the 
            weights supplied in the constructor
        """
        component_context_loss = {
            context_key: self.get_component(
                context_key,
                out_context,
                latent_lengths[mod_key],
                target_lengths[mod_key],
                reduction,
            )
            for context_key, mod_key in self.keys.items()
            if context_key in out_context
        }
        context_loss, weighted_context_loss = weighted_loss(
            component_context_loss, self.weights
        )
        return context_loss, component_context_loss, weighted_context_loss

    def get_component(
        self, key, out_context, latent_lengths, target_lengths, reduction="mean"
    ):
        """Computes a single component of the loss

        Arguments
        ---------
        key: str
            the context key
        out_context: dict
            the output context
        lengths: dict
            sequence lengths
        reduction: str
            reduction type

        Returns
        -------
        result: torch.Tensor
            the loss value
        """
        alignments = out_context[key]
        _, max_len_out, max_len_in = alignments.shape
        target_lengths_abs = (target_lengths * max_len_out).round().int()
        latent_lengths_clip = latent_lengths.clip(0, max_len_in)
        return self.loss_fn(
            attention=alignments,
            input_lengths=latent_lengths_clip,
            target_lengths=target_lengths_abs,
            reduction=reduction,
        )


class DistanceDiffLengthLoss(nn.Module):
    """A length loss that uses `distance_diff_loss`, a loss that presumes
    that predictions are a probability distribution over timesteps, with 
    the penalty increasing exponentially up to a limit the further away
    from the ground truth a probability is output
    
    Arguments
    ---------
    beta: float
        a hyperparameter controlling the rate of penalty increase
    max_weight: float
        the maximum penalty weight
    """

    def __init__(self, beta=0.25, max_weight=100.0):
        super().__init__()
        self.beta = beta
        self.max_weight = max_weight

    def forward(self, length_pred, latent, length, reduction="mean"):
        length_abs = (latent.size(1) * length).int()
        return distance_diff_loss(
            length_pred,
            length_abs,
            beta=self.beta,
            length=length,
            max_weight=self.max_weight,
            reduction=reduction,
            use_masked_penalty=True,
        )


class GateLengthLoss(nn.Module):
    def forward(self, length_pred, latent, length, reduction="mean"):
        batch_size, max_len, _ = latent.shape
        ground_truth = torch.zeros(
            batch_size, max_len, device=length_pred.device
        )
        length_range = torch.arange(0, max_len)
        length_abs = length * max_len
        ground_truth[(length_range > length_abs[..., None] - 1)] = 1.0

        return F.cross_entropy(length_pred, ground_truth, reduction=reduction)


def weighted_loss(components, weights):
    """Multiplies loss components by their respected weights
    
    Arguments
    ---------
    components: dict
        a str -> tensor dictionary of loss components
        
    weights: dict
        a dictionary of weights. The keys need to match
        the keys in components
        
    Returns
    -------
    total_loss: torch.Tensor
        the total of all weighted losses
    weighted_loss: dict
        an str -> tensor dictionary of weighted loss
        components"""

    weighted_loss = {
        key: weights[key] * component for key, component in components.items()
    }
    total_loss = torch.stack(list(weighted_loss.values())).sum(dim=0)
    return total_loss, weighted_loss


class AlignmentLoss(nn.Module):
    """A modified guided attention loss forcing attention to the end-of-sequence
    marker
    
    Arguments
    ---------
    sigma: float
        the guided attention sigma parameter
    eos_weight: float
        the end-of-sequence loss weight"""

    def __init__(self, sigma=0.2, eos_weight=1.0):
        super().__init__()
        self.attn_loss = GuidedAttentionLoss(sigma)
        self.eos_weight = eos_weight

    def forward(
        self,
        attention,
        input_lengths,
        target_lengths,
        max_input_len=None,
        max_target_len=None,
        reduction="mean",
    ):
        """Computes the guided attention loss for a single batch

        Arguments
        ---------
        attention: torch.Tensor
            A padded attention/alignments matrix
            (batch, targets, inputs)
        input_lengths: torch.tensor
            A (batch, lengths) tensor of input lengths
        target_lengths: torch.tensor
            A (batch, lengths) tensor of target lengths
        max_input_len: int
            The maximum input length - optional,
            if not computed will be set to the maximum
            of target_lengths. Setting it explicitly
            might be necessary when using data parallelism
        max_target_len: int
            The maximum target length - optional,
            if not computed will be set to the maximum
            of target_lengths. Setting it explicitly
            might be necessary when using data parallelism
        reduction: str
            the reduction type (similar to other losses)


        Returns
        -------
        loss: torch.Tensor
            A single-element or multi-element tensor with the loss value
        """
        attn_loss = self.attn_loss(
            attention=attention,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            max_input_len=max_input_len,
            max_target_len=max_target_len,
            reduction=reduction,
        )

        eos_loss = self.eos_loss(
            attention=attention,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            max_input_len=max_input_len,
            max_target_len=max_target_len,
            reduction=reduction,
        )
        return attn_loss + self.eos_weight * eos_loss

    def eos_loss(
        self,
        attention,
        input_lengths,
        target_lengths,
        max_input_len,
        max_target_len,
        reduction,
    ):
        """Computes the EOS component of the loss

        Arguments
        ---------
        attention: torch.Tensor
            A padded attention/alignments matrix
            (batch, targets, inputs)
        input_lengths: torch.tensor
            A (batch, lengths) tensor of input lengths
        target_lengths: torch.tensor
            A (batch, lengths) tensor of target lengths
        max_input_len: int
            The maximum input length - optional,
            if not computed will be set to the maximum
            of target_lengths. Setting it explicitly
            might be necessary when using data parallelism
        max_target_len: int
            The maximum target length - optional,
            if not computed will be set to the maximum
            of target_lengths. Setting it explicitly
            might be necessary when using data parallelism
        reduction: str
            the reduction type (similar to other losses)


        Returns
        -------
        loss: torch.Tensor
            A single-element or multi-element tensor with the loss value
        """
        if max_input_len is None:
            max_input_len = input_lengths.max()
        if max_target_len is None:
            max_target_len = target_lengths.max()
        input_coords, target_coords = torch.meshgrid(
            torch.arange(max_input_len, device=attention.device),
            torch.arange(max_target_len, device=attention.device),
            indexing="xy",
        )
        batch_size = attention.size(0)
        input_coords = input_coords.unsqueeze(0).repeat(batch_size, 1, 1)
        target_coords = target_coords.unsqueeze(0).repeat(batch_size, 1, 1)
        input_match = input_coords == input_lengths[:, None, None] - 1
        target_match = target_coords == target_lengths[:, None, None] - 1
        input_exceeds = input_coords >= input_lengths[:, None, None]
        target_exceeds = target_coords >= target_lengths[:, None, None]
        mask = (input_match ^ target_match) & (
            ~(input_exceeds | target_exceeds)
        )
        from matplotlib import pyplot as plt

        return reduce_loss(attention * mask, mask, reduction)


class LatentEOSMarkerLoss(nn.Module):
    """A loss component that forces EOS markers
    in the latent space
    
    Arguments
    ---------
    eos_mark: torch.nn.Module
        an EOS mark module
        
    loss_fn: callable
        the function that will be used to compute the loss"""

    def __init__(self, eos_mark, loss_fn=None):
        super().__init__()
        self.eos_mark = eos_mark
        self.loss_fn = loss_fn

    def forward(self, latents, latent_lengths, reduction="mean"):
        """Computes the loss
        
        Arguments
        ---------
        latents: torch.Tensor
            the latent representation
        latent_lengths: torch.Tensor
            latent lengths (absolute)
        """
        batch_size, max_len, feature_size = latents.shape
        idx_range = torch.arange(max_len, device=latents.device)[
            None, :, None
        ].expand_as(latents)
        idx = (latent_lengths[:, None, None].expand_as(latents) - 1).clip(1)
        eos_markers = latents[idx_range == idx].reshape(
            batch_size, feature_size
        )
        desired_eos_markers = self.eos_mark.marker[None, :].expand_as(
            eos_markers
        )
        return self.loss_fn(
            eos_markers, desired_eos_markers, reduction=reduction
        )


def with_prefix(prefix, loss_dict):
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
    return {f"{prefix}_{key}_loss": value for key, value in loss_dict.items()}


def with_prefix_transfer(prefix, loss_dict):
    """A version of with_prefix for the transfer
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


def tile_to_length(batch, lengths, target_lengths):
    """Tiles a batch to match the specified length. This is used for negative examples,
    which could be shorter than the target, for calculating triplet losses
    
    Arguments
    ---------
    batch: torch.Tensor
        a (batch x length x feature) batch
    
    lengths: torch.Tensor
        the actual lengths of a batch
    
    target_lengths: torch.Tensor
        the lengths to which examples will be tiled


    Returns
    -------
    result: torch.Tensor
        a batch tiled to the specified length
    """
    max_ratio = (target_lengths / lengths).max().ceil().int().item()
    batch_tiled, _ = concat_padded_features(
        feats=[batch] * max_ratio,
        lengths=[lengths / batch.size(1)] * max_ratio,
        dim=1,
    )
    max_target_length = target_lengths.max().int().item()
    result = batch_tiled[:, :max_target_length, :]
    mask = length_to_mask(target_lengths, max_target_length).unsqueeze(-1)
    return result * mask
