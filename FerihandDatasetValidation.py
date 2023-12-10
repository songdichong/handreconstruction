import os
import json
import torch
from PIL import Image
from torchvision import transforms

class FerihandDatasetValidation(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotations_file):
        """
        Args:
            image_dir (string): Directory with all the images.
            annotations_file (string): Path to the JSON file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.annotations = json.load(open(annotations_file))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir, f'{idx:08d}.jpg') 
        image = Image.open(img_name).convert('RGB')
        
        annotations = self.annotations[idx]
        annotations = torch.tensor(annotations).view(-1)

        sample = {'image': image, 'annotations': torch.tensor(annotations)}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample