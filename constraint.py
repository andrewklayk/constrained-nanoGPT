import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F



def softmax_constraint(y, token, from_logits=False):
    """
    Compute a regularization loss term that penalizes high probability of a specific token.

    Args:
        logits: The model's logits (before softmax), shape (batch_size, seq_len, vocab_size)
        token: The token ID to penalize

    Returns:
        A scalar loss term to be added to the total loss
    """
    if from_logits:
        # Compute softmax probabilities 
        probs = F.softmax(y, dim=-1)  # shape (batch_size, seq_len, vocab_size)
    else:
        probs = y
    # Extract the probability of the unwanted token
    token_probs = probs[..., token]  # shape (batch_size, seq_len)

    # Penalize high probability: use mean to aggregate over batch and sequence
    penalty = torch.mean(token_probs)
    return penalty

def fand(alpha, beta=None):
    if beta:
        return torch.min(alpha, beta)
    return torch.min(alpha, dim=-1)

def fneg(alpha):
    return 1 - alpha

def fimpl(alpha, beta):

    # return (alpha <= beta) + (alpha > beta) * beta
    # return beta
    return torch.min(torch.ones_like(beta), beta/alpha)
    # return torch.max(1-alpha, beta)
    # return 1 - alpha + alpha*beta

def impl_constraint(logits, tokens_pos, tokens_neg, token_target):
    """
    Compute a regularization loss term that penalizes fuzzy implication of  of a specific token.

    Args:
        logits: The model's logits (before softmax), shape (batch_size, seq_len, vocab_size)
        token: The token ID to penalize

    Returns:
        A scalar loss term to be added to the total loss
    """

    probs = torch.nn.functional.softmax(logits, dim=-1)

    if tokens_pos:
        lh_pos = probs[..., tokens_pos]
    
    if tokens_neg:
        lh_neg = probs[..., tokens_neg]
        nlh_neg = fneg(lh_neg)
    
    rh = probs[..., token_target]
    
    if tokens_pos and tokens_neg:
        lh = fand(fand(lh_pos), fand(nlh_neg))
    elif tokens_pos:
        lh = fand(lh_pos)
    elif tokens_neg:
        lh = fand(lh_neg)
    else:
        raise ValueError('empty lhs')
    
    imp = fimpl(lh, rh)
    return imp

def pos_impl_constraint(probs1, prob2):
    """
    Compute a regularization loss term that penalizes fuzzy implication of a specific token.

    Args:
        probs1: Probabilities of the left-side of the implication. Will be conjuncted.
        prob2: Probability of the right-side.

    Returns:
        A scalar loss term to be added to the total loss
    """
    if probs1.ndim > 1 and probs1.shape[-1] > 1:
        lh = fand(probs1)
    else:
        lh = probs1
    
    rh = prob2
    
    imp = fimpl(lh, rh)
    return imp
