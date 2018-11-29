import torch
import torchvision
import torchvision.transforms as transforms
import wandb
import os
import json


wandb.init(project="sm-pytorch-cifar")

config = wandb.config
# Set defaults if we dont have values from SageMaker
if config.get('batch_size') is None:
    config.batch_size = 20
    config.lr = 0.001
    config.momentum = 0.9
    config.epochs = 10
    config.hidden_nodes = 120
    config.conv1_channels = 5
    config.conv2_channels = 16


########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(
    os.getenv('SM_CHANNEL_TRAINING', 'data'), train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    os.getenv('SM_CHANNEL_TRAINING', 'data'), train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(
            3, config.conv1_channels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(config.conv1_channels,
                               config.conv2_channels, 5)
        self.fc1 = nn.Linear(config.conv2_channels *
                             5 * 5, config.hidden_nodes)
        self.fc2 = nn.Linear(config.hidden_nodes, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, config.conv2_channels * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)

wandb.watch(net)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=float(
    config.lr), momentum=config.momentum)

# loop over the dataset multiple times
for epoch in range(config.epochs):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    dataiter = iter(testloader)
    images, labels = dataiter.next()
    images, labels = images.to(device), labels.to(device)

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    example_images = [wandb.Image(image, caption=classes[predicted])
                      for image, predicted, label in zip(images, predicted, labels)]

    # Let us look at how the network performs on the whole dataset.

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print("Test Accuracy: %.4f" % test_acc)
    class_acc = {}
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
        class_acc["Accuracy of %5s" %
                  (classes[i])] = 100 * class_correct[i] / class_total[i]

    wandb.log(class_acc, commit=False)
    wandb.log({"Examples": example_images,
               "Test Acc": test_acc,
               "Loss": running_loss})
