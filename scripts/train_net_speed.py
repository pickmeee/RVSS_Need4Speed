import torch
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
import numpy as np

from steerDS import SteerDataSet

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from structure_net import Net

from torch.utils.tensorboard import SummaryWriter



# transform = transforms.Compose(
# [transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



transform = transforms.Compose([
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ColorJitter(contrast=0.15, saturation=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


script_path = os.path.dirname(os.path.realpath(__file__))


ds = SteerDataSet(os.path.join(script_path, '..', 'data', 'train_v'), '.jpg', transform)

print("The dataset contains %d images " % len(ds))

batch_size = 4

ds_dataloader = DataLoader(ds,batch_size,shuffle=True)
all_y = []
for S in ds_dataloader:
    im, y = S    
    # print(f'shape: {im.shape}')
    # print(f'label: {y}')
    all_y += y.tolist()

print(f'Input shape: {im.shape}')
print('Outputs and their counts:')
print(np.unique(all_y, return_counts = True))



trainset = ds
trainloader = ds_dataloader
classes = ('right', 'straight', 'left')

# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 16, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(16, 32, 5)
#         self.conv3 = nn.Conv2d(32, 64, 3)
#         self.fc1 = nn.Linear(64 * 12 * 37, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 3)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

net = Net()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)


# visualization


# Initialize the SummaryWriter
writer = SummaryWriter('./RVSS_Need4Speed/runs/steering_model_experiment')

# Assuming `inputs` is a batch of input data that has already been loaded
sample_inputs, _ = next(iter(ds_dataloader))
sample_inputs = sample_inputs.to(device)
writer.add_graph(net, sample_inputs)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

stop_training = False


for epoch in range(60):  # loop over the dataset multiple times
    class_correct = {classname: 0 for classname in classes}
    class_total = {classname: 0 for classname in classes}


    running_loss = 0.0
    for i, data in enumerate(ds_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, y = data[0].to(device), data[1].to(device)

        labels = torch.zeros(y.size(0), dtype=torch.long, device=device)  # Ensure labels tensor matches batch size
        labels[y < 0] = 0  # left
        labels[y == 0] = 1  # straight
        labels[y > 0] = 2  # right

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update class_correct and class_total
        for label, prediction in zip(labels, predicted):
            if label == prediction:
                class_correct[classes[label.item()]] += 1
            class_total[classes[label.item()]] += 1


        avg_loss = 1
        # print statistics
        running_loss += loss.item()
        if i % 50 == 49:    # print every 50 mini-batches
            avg_loss = running_loss / 50
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.5f}')
            running_loss = 0.0
            # Log training loss using SummaryWriter
            writer.add_scalar('training_loss', loss.item(), epoch * len(ds_dataloader) + i)

        # Check if average loss in the last 50 mini-batches is below the threshold
        if avg_loss < 0.001:
            print(f"Stopping training as loss reached {avg_loss:.4f} which is below the threshold.")
            stop_training = True
            break
    
    for classname in classes:
        accuracy = 100 * class_correct[classname] / class_total[classname]
        writer.add_scalar(f'Training_Accuracy/{classname}', accuracy, epoch)
        print(f'Epoch {epoch+1} - Class {classname} Accuracy: {accuracy:.2f}%')
        
        
    scheduler.step()

    if stop_training:
        break

print('Finished Training')
writer.close()


PATH = '/home/yanzhang/rvss_ws/RVSS_Need4Speed/models/train_steer_class_net_modified.pth'
torch.save(net.state_dict(), PATH)

# test on tds_dataloader

tds = SteerDataSet(os.path.join(script_path, '..', 'data', 'test'), '.jpg', transform)

print("The test dataset contains %d images " % len(tds))

tds_dataloader = DataLoader(tds,batch_size,shuffle=True)
all_ty = []
for S in tds_dataloader:
    im, ty = S    
    # print(f'shape: {im.shape}')
    # print(f'label: {y}')
    all_ty += ty.tolist()

print(f'Input shape: {im.shape}')
print('Outputs and their counts:')
print(np.unique(all_ty, return_counts = True))

# Set model to evaluation mode
net.eval()

# Initialize variables to track accuracy
correct = 0
total = 0

# Initialize counters for correct predictions and total instances per class
correct_per_class = {classname: 0 for classname in classes}
total_per_class = {classname: 0 for classname in classes}


# Disable gradient computation
with torch.no_grad():
    for data in tds_dataloader:
        # images, ty = data
        images, ty = data[0].to(device), data[1].to(device)
        labels = torch.zeros(ty.size(0), dtype=torch.long, device=device)
        labels[ty < 0] = 0  # left
        labels[ty == 0] = 1  # straight
        labels[ty > 0] = 2  # right

        # Forward pass
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)

         # Update total and correct counts per class
        for label, prediction in zip(labels, predicted):
            if label == prediction:
                correct_per_class[classes[label.item()]] += 1
            total_per_class[classes[label.item()]] += 1

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        # Log accuracy to TensorBoard
        writer.add_scalar('Accuracy/Test', accuracy, epoch)

 # Update total and correct counts per class
for label, prediction in zip(labels, predicted):
    if label == prediction:
        correct_per_class[classes[label.item()]] += 1
        total_per_class[classes[label.item()]] += 1

# Print accuracy
print(f'Accuracy of the network on the {len(tds)} images: {accuracy} %')

for classname in classes:
    accuracy = 100 * float(correct_per_class[classname]) / total_per_class[classname]
    writer.add_scalar(f'Accuracy/{classname}', accuracy, epoch)
    print(classname, accuracy)


