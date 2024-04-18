from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import ssl

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
ssl._create_default_https_context = ssl._create_unverified_context

def get_transform():
    """Define the image transformations."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

def load_dataset(data_dir, train=True, transform=None):
    """Load a dataset."""
    return datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)

def get_data_loader(data_dir='./data', test=False, max_learners=1):
    """Create and return data loaders for training and validation/test, equally divided among learners."""
    transform = get_transform()
    if test:
        dataset = load_dataset(data_dir, train=False, transform=transform)
        return DataLoader(dataset, batch_size=32, shuffle=False)

    # Training and validation setup
    dataset = load_dataset(data_dir, train=True, transform=transform)
    num_samples = len(dataset)
    indices = np.arange(num_samples)

    split = int(np.floor(0.1 * num_samples))
    train_idx, valid_idx = indices[split:], indices[:split]

    # REMOVE LATER
    num_samples_needed = 100 * 32  # 50 batches, 32 samples per batch
    train_idx = train_idx[:num_samples_needed]

    # Distribute indices evenly among learners
    per_learner = len(train_idx) // max_learners
    extra_samples = len(train_idx) % max_learners
    
    train_loader = []
    
    for i in range(max_learners):
        start_idx = i * per_learner + min(i, extra_samples)
        end_idx = start_idx + per_learner + (1 if i < extra_samples else 0)

        # Calculate the number of batches
        num_samples_for_learner = end_idx - start_idx
        num_batches = num_samples_for_learner // 32 + (num_samples_for_learner % 32 > 0)

        # Use Subset to directly slice the dataset without sampling
        train_subset = Subset(dataset, train_idx[start_idx:end_idx])
        train_loader.append(DataLoader(train_subset, batch_size=32))
        
    valid_subset = Subset(dataset, valid_idx)
    valid_loader = DataLoader(valid_subset, batch_size=32)

    return train_loader, valid_loader, num_batches
