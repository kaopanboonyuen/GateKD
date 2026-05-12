
import torch
from transformers import AutoTokenizer
from gatekd.modeling.gatekd_model import GateKDModel

def main():
    model = GateKDModel(
        teacher_name="google/flan-t5-large",
        student_name="google/flan-t5-small"
    )

    checkpoint = torch.load("checkpoints/gatekd_t5_small.pt")
    model.load_state_dict(checkpoint)

    print("Evaluation completed.")

if __name__ == "__main__":
    main()
