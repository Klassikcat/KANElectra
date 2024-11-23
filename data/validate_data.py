import os
import sys
from transformers import AutoTokenizer
from pathlib import Path

sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))
from src.ElectraKAN import ElectraPretrainingDataset, ElectraKANDataModule

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google/electra-base-generator")
    with open("./data/dev-v2.0.txt", "r") as f:
        data = f.read()

    max_length: int = 512

    dataset = ElectraPretrainingDataset(
        texts=data.split('\n'),
        tokenizer=tokenizer,
        max_length=512
    )
    assert len(dataset[0]) == 4
    for i in range(len(dataset[0])):
        assert len(dataset[0][i]) == max_length

    datamodule = ElectraKANDataModule(
        train_dataset=dataset,
        val_dataset=dataset,
        test_dataset=dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=False
    )

    print("initialization test passed!")
