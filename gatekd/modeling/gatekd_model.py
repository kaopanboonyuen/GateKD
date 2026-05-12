
import torch
import torch.nn as nn

from transformers import AutoModelForSeq2SeqLM

from gatekd.modeling.confidence import predictive_entropy
from gatekd.modeling.losses import (
    gated_soft_loss,
    gated_hidden_loss,
    gated_attention_loss
)

class GateKDModel(nn.Module):

    def __init__(
        self,
        teacher_name,
        student_name,
        lambda_soft=1.0,
        lambda_hidden=0.5,
        lambda_attention=0.1
    ):
        super().__init__()

        self.teacher = AutoModelForSeq2SeqLM.from_pretrained(
            teacher_name,
            output_hidden_states=True,
            output_attentions=True
        )

        self.student = AutoModelForSeq2SeqLM.from_pretrained(
            student_name,
            output_hidden_states=True,
            output_attentions=True
        )

        for p in self.teacher.parameters():
            p.requires_grad = False

        self.lambda_soft = lambda_soft
        self.lambda_hidden = lambda_hidden
        self.lambda_attention = lambda_attention

    def forward(self, batch):

        with torch.no_grad():
            teacher_outputs = self.teacher(**batch)

        student_outputs = self.student(**batch)

        confidence = predictive_entropy(teacher_outputs.logits)

        soft_loss = gated_soft_loss(
            student_outputs.logits,
            teacher_outputs.logits,
            confidence
        )

        hidden_loss = 0.0

        for s_h, t_h in zip(
            student_outputs.decoder_hidden_states,
            teacher_outputs.decoder_hidden_states
        ):
            gate = (confidence.mean() > confidence.mean()).float()
            hidden_loss += gated_hidden_loss(s_h, t_h, gate)

        attention_loss = 0.0

        for s_a, t_a in zip(
            student_outputs.decoder_attentions,
            teacher_outputs.decoder_attentions
        ):
            gate = (confidence.mean() > confidence.mean()).float()
            attention_loss += gated_attention_loss(s_a, t_a, gate)

        total_loss = (
            student_outputs.loss
            + self.lambda_soft * soft_loss
            + self.lambda_hidden * hidden_loss
            + self.lambda_attention * attention_loss
        )

        return {
            "loss": total_loss
        }
