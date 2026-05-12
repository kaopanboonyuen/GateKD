
import torch

def predictive_entropy(logits):
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
    confidence = torch.exp(-entropy)
    return confidence
