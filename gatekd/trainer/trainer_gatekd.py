
import torch
from tqdm import tqdm

def train_loop(model, tokenizer, config):

    optimizer = torch.optim.AdamW(
        model.student.parameters(),
        lr=config["learning_rate"]
    )

    model.train()

    for epoch in range(config["epochs"]):

        progress = tqdm(range(100))

        for _ in progress:

            dummy_batch = {
                "input_ids": torch.randint(0, 1000, (2, 32)),
                "attention_mask": torch.ones((2, 32)),
                "labels": torch.randint(0, 1000, (2, 32))
            }

            outputs = model(dummy_batch)

            loss = outputs["loss"]

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            progress.set_description(
                f"Epoch {epoch} Loss {loss.item():.4f}"
            )

    torch.save(
        model.state_dict(),
        "checkpoints/gatekd_t5_small.pt"
    )
