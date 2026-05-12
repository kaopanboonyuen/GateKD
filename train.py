
import yaml
import torch
from transformers import AutoTokenizer
from gatekd.modeling.gatekd_model import GateKDModel
from gatekd.trainer.trainer_gatekd import train_loop

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config("configs/t5_small.yaml")

    tokenizer = AutoTokenizer.from_pretrained(config["student_model"])

    model = GateKDModel(
        teacher_name=config["teacher_model"],
        student_name=config["student_model"],
        lambda_soft=config["lambda_soft"],
        lambda_hidden=config["lambda_hidden"],
        lambda_attention=config["lambda_attention"],
    )

    train_loop(model, tokenizer, config)

if __name__ == "__main__":
    main()
