import torch
from torchvision import transforms
from datasets import load_dataset
import matplotlib.pyplot as plt
import random

def loadPixelArtDataset():
    return load_dataset("jiovine/pixel-art-nouns")['train']

class PixelArtDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, image_size: int = 32):
        self.dataset = hf_dataset
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = sample['image'].convert('RGB')
        img = self.transform(img)
        text = sample['text']
        return {'image': img, 'text': text}
    
def displayRandomSample(dataset):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    indices = random.sample(range(len(dataset)), 10)

    for i, ax in enumerate(axes):
        sample = dataset[indices[i]]
        img = sample['image']
        
        ax.imshow(img, interpolation='nearest')
        ax.axis('off')

    plt.suptitle("PixelDiffusion - Random samples", fontsize=13)
    plt.tight_layout()
    plt.show()