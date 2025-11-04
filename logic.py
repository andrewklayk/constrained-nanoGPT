import torch

def fand(alpha, beta=None):
    """fuzzy conjunction"""
    if beta:
        return torch.min(alpha, beta)
    return torch.min(alpha)

def fneg(alpha):
    """fuzzy negation"""
    return 1 - alpha

def fimpl(alpha, beta):
    """fuzzy implication"""
    return 1 - alpha + alpha*beta

def impl_constraint(logits, tokens_pos, tokens_neg, token_target):
    """fuzzy implication constraint"""

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