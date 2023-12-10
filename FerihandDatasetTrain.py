import os
import json
import torch
from PIL import Image
from torchvision import transforms

class FerihandDatasetTrain(torch.utils.data.Dataset):
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
        self.num_unique_images = 32560

    def __len__(self):
        return len(self.annotations*4)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        annotation_idx = idx % self.num_unique_images

        # Format the index as an 8-digit number for image file name
        img_name = os.path.join(self.image_dir, f'{idx:08d}.jpg')
        image = Image.open(img_name).convert('RGB')

        annotations = self.annotations[annotation_idx]
         # Flatten the annotation list to a 63-element vector
        annotations = torch.tensor(annotations).view(-1)
        if self.transform:
            image = self.transform(image)

        return {'image': image, 'annotations': torch.tensor(annotations)}
        