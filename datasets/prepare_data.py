
from datasets import load_dataset

def load_csqa():
    dataset = load_dataset("commonsense_qa")
    return dataset

if __name__ == "__main__":
    dataset = load_csqa()
    print(dataset)
