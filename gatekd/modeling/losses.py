
import torch
import torch.nn.functional as F

def gated_soft_loss(student_logits, teacher_logits, confidence):
    teacher_probs = torch.softmax(teacher_logits, dim=-1)
    student_log_probs = torch.log_softmax(student_logits, dim=-1)

    loss = -(teacher_probs * student_log_probs).sum(dim=-1)
    loss = confidence * loss
    return loss.mean()

def gated_hidden_loss(student_hidden, teacher_hidden, gate):
    return gate * F.mse_loss(student_hidden, teacher_hidden)

def gated_attention_loss(student_attn, teacher_attn, gate):
    return gate * F.mse_loss(student_attn, teacher_attn)
