import torch
import torchvision.models as models
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split
from FerihandDatasetTrain import FerihandDatasetTrain
from FerihandDatasetValidation import FerihandDatasetValidation
from torch import nn
import pickle
import matplotlib.pyplot as plt
def evaluate_model(model, data_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad(): # Do not calculate gradients for this part
        for batch in data_loader:
            images, labels = batch['image'].to(device), batch['annotations'].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(data_loader)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
best_val_loss = float('inf')  # Initialize with a very high value
best_epoch = 0
model_name_list = ["resnet50", "GoogleNet", "MobileNetV2"]
model_list = [models.resnet50(pretrained=True),models.googlenet(pretrained=True),
              models.mobilenet_v2(pretrained=True)]


for i in range(3):
    # Load a pretrained ResNet model
    model = model_list[i]
    model_name = model_name_list[i]

    training_dataset = FerihandDatasetTrain(image_dir='./training/rgb', annotations_file='training_verts.json')
    training_data_loader = DataLoader(training_dataset, batch_size=32, shuffle=True)

    validation_dataset = FerihandDatasetValidation(image_dir='./evaluation/rgb', annotations_file='evaluation_verts.json')
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

    # Freeze all layers in the model
    for param in model.parameters():
        param.requires_grad = False

    if i > 1:
        additional_layer = nn.Sequential(
            nn.Linear(model.classifier[1].out_features, 2048),  # First dense layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 778*3),  # Second dense layer
        )
        new_classifier = nn.Sequential(
            model.classifier,
            additional_layer,
        )
        model.classifier = new_classifier
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)  # Only optimize the parameters of the added layers

    else:
        # Adding two dense layers
        additional_layer = nn.Sequential(
            nn.Linear(model.fc.out_features, 2048),  # First dense layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 778*3),  # Second dense layer,
        )
        new_fc = nn.Sequential(
            model.fc,
            additional_layer,
        )
        model.fc = new_fc
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.0001)  # Only optimize the parameters of the added layers

    model = model.to(device)

    criterion = nn.MSELoss() # Mean Squared Error loss
    num_epochs = 10
    train_loss_history = []
    val_loss_history = []
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        # for name, param in model.named_parameters():
        #     print(f"{name} is on {param.device}")
        for batch in training_data_loader:
            images, labels = batch['image'].to(device), batch['annotations'].to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss, etc.
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(training_data_loader)

        avg_val_loss = evaluate_model(model, validation_loader, criterion)
        # Save model if validation loss has improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_name+' best_model_state_dict1.pth')
            print(f"Epoch {epoch}: Validation loss improved to {avg_val_loss:.2f}, saving model...")

        # Save losses
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')

    # Save loss history
    with open('loss_history.pkl', 'wb') as f:
        pickle.dump({'train_loss': train_loss_history, 'val_loss': val_loss_history}, f)
    # Save model
    torch.save(model.state_dict(), model_name+'model_state_dict1.pth')

    epochs = range(0, len(train_loss_history) + 1)
    # Plot training and validation loss
    plt.figure()
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.xticks(epochs)
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(model_name+'loss_plot1.png')

    # Unfreeze the last 20 layers
    layers_to_unfreeze = 20
    for child in reversed(list(model.children())):
        if layers_to_unfreeze <= 0:
            break
        for param in child.parameters():
            param.requires_grad = True
        layers_to_unfreeze -= 1
    if i > 1:
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.0001)
    else:
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.0001)
     # Initialize lists to track loss

    for epoch in range(num_epochs, 3*num_epochs):
        model.train()
        total_train_loss = 0
        for batch in training_data_loader:
            images, labels = batch['image'].to(device), batch['annotations'].to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss, etc.
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(training_data_loader)

        avg_val_loss = evaluate_model(model, validation_loader, criterion)
        # Save model if validation loss has improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_name+' best_model_state_dict2.pth')
            print(f"Epoch {epoch}: Validation loss improved to {avg_val_loss:.2f}, saving model...")

        # Save losses
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')

    # Save loss history
    with open('loss_history.pkl', 'wb') as f:
        pickle.dump({'train_loss': train_loss_history, 'val_loss': val_loss_history}, f)
    # Save model
    torch.save(model.state_dict(), model_name+'model_state_dict2.pth')
    epochs = range(0, len(train_loss_history) + 1)
    # Plot training and validation loss
    plt.figure()
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.xticks(epochs)
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(model_name+'loss_plot2.png')