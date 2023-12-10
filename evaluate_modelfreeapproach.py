import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from torchvision import transforms
from FerihandDatasetValidation import FerihandDatasetValidation
import matplotlib.pyplot as plt
import pickle
model_joints = models.resnet50(pretrained=False)
model_name = "ResNet50"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define the transform
transform = transforms.ToTensor()
additional_layer_joints = nn.Sequential(
            nn.Linear(model_joints.fc.out_features, 1024),  # First dense layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),  # Second dense layer,
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 63) 
        )
# Apply the transform to the image
new_classifier = nn.Sequential(
    model_joints.fc,
    additional_layer_joints,
)
model_joints.fc = new_classifier
model_joints.load_state_dict(torch.load('./joints/resnet50model_state_dict6.pth'))
model_joints.eval()
model_joints = model_joints.to(device)

model_vertex = models.resnet50(pretrained=False)
additional_layer_vertex = nn.Sequential(
            nn.Linear(model_vertex.fc.out_features, 2048),  # First dense layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 778*3),  # Second dense layer,
        )
new_classifier = nn.Sequential(
            model_vertex.fc,
            additional_layer_vertex,
        )
model_vertex.fc = new_classifier
model_vertex.load_state_dict(torch.load('./vertex/resnet50model_state_dict4.pth'))
model_vertex.eval()
model_vertex = model_vertex.to(device)

validation_dataset_joints = FerihandDatasetValidation(image_dir='./evaluation/rgb', annotations_file='evaluation_xyz.json')
validation_dataset_vertex = FerihandDatasetValidation(image_dir='./evaluation/rgb', annotations_file='evaluation_verts.json')

validation_loader_joints = DataLoader(validation_dataset_joints, batch_size=32, shuffle=False)
validation_loader_vertex = DataLoader(validation_dataset_vertex, batch_size=32, shuffle=False)

absolute_differences_joints = []
absolute_differences_vertex = []

with torch.no_grad(): 
    for batch in validation_loader_joints:  
        inputs = batch['image'].to(device) 
        ground_truth = batch['annotations'].to(device) 
        outputs = model_joints(inputs)
        differences = torch.abs(outputs - ground_truth)
        absolute_differences_joints.extend(differences.cpu().numpy())

with torch.no_grad(): 
    for batch in validation_loader_vertex:  
        inputs = batch['image'].to(device) 
        ground_truth = batch['annotations'].to(device) 
        outputs = model_vertex(inputs)
        differences = torch.abs(outputs - ground_truth)
        absolute_differences_vertex.extend(differences.cpu().numpy())
absolute_differences_joints = np.array(absolute_differences_joints).flatten()
absolute_differences_vertex = np.array(absolute_differences_vertex).flatten()

def calculate_percentage(absolute_differences, threshold):
    within_threshold = absolute_differences <= threshold
    percentage = np.mean(within_threshold) 
    return percentage*100

thresholds = [0, 1/255, 5/255, 10/255, 15/255, 20/255, 25/255, 1] 
custom_positions = [0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 1]

percentages_joints = [calculate_percentage(absolute_differences_joints, t) for t in thresholds]
percentages_vertex = [calculate_percentage(absolute_differences_vertex, t) for t in thresholds]

# Plot the data
plt.figure(figsize=(8, 5))

# Plotting for joints and vertex models
plt.plot(custom_positions, percentages_joints, 'bo-', label='Joints Model')  # Blue circles and line for joints
plt.plot(custom_positions, percentages_vertex, 'ro-', label='Vertex Model')  # Red circles and line for vertex

# Custom x-ticks and labels
plt.xticks(custom_positions, ['0', '1', '5', '10', '15', '20', '25', '255'])

# Adding labels, title, and legend
plt.xlabel('Threshold (pixel)')
plt.ylabel('Percentage within Threshold')
plt.title('AUC of Joints and Vertex')
plt.legend()

# Show the plot
plt.savefig('auc-both.png')