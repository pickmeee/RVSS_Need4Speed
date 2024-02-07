import torch
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
import numpy as np

from steerDS import SteerDataSet

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter



transform = transforms.Compose(
[transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

script_path = os.path.dirname(os.path.realpath(__file__))


ds = SteerDataSet(os.path.join(script_path, '..', 'data', 'train'), '.jpg', transform)

print("The dataset contains %d images " % len(ds))

batch_size = 10

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

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 27 * 77, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


# visualization


# Initialize the SummaryWriter
writer = SummaryWriter('./RVSS_Need4Speed/runs/steering_model_experiment')

# Assuming `inputs` is a batch of input data that has already been loaded
sample_inputs, _ = next(iter(ds_dataloader))
writer.add_graph(net, sample_inputs)

for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(ds_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, y = data

        labels = torch.zeros(y.size(0), dtype=torch.long)  # Ensure labels tensor matches batch size
        labels[y < 0] = 0  # left
        labels[y == 0] = 1  # straight
        labels[y > 0] = 2  # right

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 50 == 49:    # print every 500 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}')
            running_loss = 0.0
        
        # Log training loss using SummaryWriter
        writer.add_scalar('training_loss', loss.item(), epoch * len(ds_dataloader) + i)

print('Finished Training')
writer.close()


PATH = './RVSS_Need4Speed/models/train_steer_class_net.pth'
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
        images, ty = data
        labels = torch.zeros(ty.size(0), dtype=torch.long)
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


