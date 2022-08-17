import wandb
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST

class FashionCNN(nn.Module):
  """Simple CNN for Fashion MNIST."""

  def __init__(self):
    super().__init__()
    self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.layer2 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )    
    self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
    self.drop = nn.Dropout2d(0.25)
    self.fc2 = nn.Linear(in_features=600, out_features=120)
    self.fc3 = nn.Linear(in_features=120, out_features=10) 
  
  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = out.view(out.size(0), -1)
    out = self.fc1(out)
    out = self.drop(out)
    out = self.fc2(out)
    out = self.fc3(out)
    return out


# Training hyperparamters
config = {
  "learning_rate": 0.0001,
  "batch_size": 32,
  "epochs": 5,
}

# Pass config into wandb.init
with wandb.init(config=config) as run:
  
  # Training setup
  config = run.config
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = FashionCNN()
  model.to(device)
  train_dataset = FashionMNIST("./data/", download=True, train=True, transform=transforms.ToTensor())
  train_loader = DataLoader(train_dataset, batch_size=config.batch_size, pin_memory=True)
  error = nn.CrossEntropyLoss()
  learning_rate = config.learning_rate
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  # Epoch loop
  iter = 0
  losses = []
  for epoch in range(config.epochs):

    # Iterate over batches of the data
    for idx, (images, labels) in enumerate(train_loader):

      iter += 1
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      loss = error(outputs, labels)
      losses.append(loss.item())

      if iter % 25 == 1:
        run.log(
          {
            "train/loss": sum(losses)/len(losses),  # Log average loss
            "train/losses": wandb.Histogram(losses)  # Log all losses
          }
        )
        losses = []

      if idx == 0:
        table = wandb.Table(columns=["image", "label", "prediction"])
        for im, lab, pred in zip(images, labels, outputs):
          pred = torch.argmax(pred)
          table.add_data(wandb.Image(im.cpu()), lab.item(), pred.item())
        run.log({"train/predictions": table})

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()