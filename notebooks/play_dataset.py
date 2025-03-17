import torch
from torch.utils.data import Dataset

class HumanDataset(Dataset):
    def __init__(self, size=5):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        label = torch.tensor(idx)  # simple label
        target_dict = {"age": torch.tensor(idx + 20), "gender": torch.tensor(idx % 2)}  # simple dict target
        return label, target_dict


class ObjectDataset(Dataset):
    def __init__(self, size=5):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        label = torch.tensor(idx + 100)  # simple label with offset
        target_tensor = torch.tensor([idx, idx + 1])  # simple tensor target
        
        return label, target_tensor
    

class CombinedDataset(Dataset):
    def __init__(self, human_dataset, object_dataset):
        self.human_dataset = human_dataset
        self.object_dataset = object_dataset
        self.human_size = len(human_dataset)
        self.object_size = len(object_dataset)
        self.total_size = self.human_size + self.object_size

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        if idx < self.human_size:
            label, target_dict = self.human_dataset[idx]
            return label, target_dict, "human"
        else:
            label, target_tensor = self.object_dataset[idx - self.human_size]
            return label, target_tensor, "object"
        
def custom_collate_fn(batch):
    labels = [item[0]for item in batch]
    dataset_types = [item[2] for item in batch]

    human_targets = []
    object_targets = []

    for item in batch:
        if item[2] == "human":
            human_targets.append(item[1])
        else:
            object_targets.append(item[1])

    return labels, human_targets, object_targets, dataset_types
    
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # Instantiate datasets
    human_dataset = HumanDataset(size=30)
    object_dataset = ObjectDataset(size=30)

    # Combine them
    combined_dataset = CombinedDataset(human_dataset, object_dataset)

    # Create DataLoader
    dataloader = DataLoader(combined_dataset, batch_size=12, collate_fn=custom_collate_fn, shuffle=True)

    # Iterate over DataLoader
    for batch_idx, (labels, targets,targets_object, dataset_types) in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        print("Labels:", labels)
        print("Dataset Types:", dataset_types)
        print("Human Targets:", targets)
        print("Object Targets:", targets_object)
        
