import torch
import torch.nn as nn
import torchvision.models as models
from manopth.manolayer import ManoLayer
from manopth import demo
import numpy as np
from PIL import Image
from torchvision import transforms

ncomps = 6
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
model = models.mobilenet_v2(pretrained=False)
additional_layer = nn.Sequential(
            nn.Linear(model.classifier[1].out_features, 512),  # First dense layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 61)  # Second dense layer, outputting 61 features
        )
new_classifier = nn.Sequential(
    model.classifier,
    additional_layer,
)
model.classifier = new_classifier
model.load_state_dict(torch.load('./failure1/MobileNetV2model_state_dict4.pth'))
model.eval()
model = model_joints.to(device)
mano_layer = ManoLayer(
    mano_root='./manopth/mano/models', use_pca=True, ncomps=ncomps, flat_hand_mean=True)
for i in range(9):
    image_path = './evaluation/rgb/0000000'+str(i)+'.jpg'
    print("current image path: ", image_path)
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0) 

    image = image.to(device)
    with torch.no_grad():  # Disable gradient calculation for inference
        output_joints = model_joints(image)
        output_vertex = model_vertex(image)
    output_joints = output_joints.view(-1, 21, 3)
    output_vertex = output_vertex.view(-1, 778, 3)
    output_joints = output_joints.cpu()
    output_vertex = output_vertex.cpu()
    print(output_vertex.shape)
    print(output_joints.shape)
    # with torch.no_grad():
    #     output_mano = model(image)
    # output_mano = output_mano.cpu()

    # print(output_mano.shape)
    # pose = output_mano[:, :45]
    # shape = output_mano[:, 45:55]

    # output_vertex, output_joints = mano_layer(pose, shape)

    demo.display_hand({
        'verts': output_vertex,
        'joints': output_joints
    },mano_faces=mano_layer.th_faces)
