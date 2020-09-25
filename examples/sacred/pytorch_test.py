import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sacred import Experiment
from wandb.sacred import WandbObserver
from torchvision.utils import make_grid
import numpy as np
EXPERIMENT_NAME = 'my_experiment-on-slurm'
YOUR_CPU = None  # None is the default setting and will result in using localhost, change if you want something else

ex = Experiment(EXPERIMENT_NAME)

ex.observers.append(WandbObserver(config={"Job_type":"torch_test","gpu_type":"t4"}, reinit=False ))


class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:

    def __init__(self):
        # SACRED: we don't need any parameters here, they're in the config and the functions get a @ex.capture handle
        # later
        self.model = self.make_model()
        self.optimizer = self.make_optimizer()
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_dataset, self.test_dataset = self.get_datasets()
        self.train_loader, self.test_loader = self.get_dataloaders()

    @ex.capture
    def make_model(self, input_size, hidden_size, num_classes):
        model = NeuralNet(input_size, hidden_size, num_classes).to(device)
        return model

    @ex.capture
    def make_optimizer(self, learning_rate):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        return optimizer

    def get_datasets(self):
        train_dataset = torchvision.datasets.MNIST(root='../../data', train=True, transform=transforms.ToTensor(),
                                                   download=True)

        test_dataset = torchvision.datasets.MNIST(root='../../data', train=False, transform=transforms.ToTensor())

        return train_dataset, test_dataset

    @ex.capture
    def get_dataloaders(self, batch_size):
        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)

        return train_loader, test_loader

    @ex.capture
    def train(self, num_epochs, _run):
        total_step = len(self.train_loader)
        for epoch in range(num_epochs):
            total_loss = 0
            for i, (images, labels) in enumerate(self.train_loader):
                # Move tensors to the configured device
                tensor_imgs = images
                images = images.reshape(-1, 28 * 28).to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = self.model(images)

                loss = self.loss_fn(outputs, labels)
                ex.log_scalar('Batch_loss', float(loss.data))  # SACRED: Keep track of the loss

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss = total_loss + loss
                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                    npgrid = make_grid(tensor_imgs, nrow=10)
                    npgrid = np.transpose(npgrid.cpu().detach().numpy(), (1, 2, 0))
                    ex.log_scalar("Images",npgrid)
                    
            ex.log_scalar("accuracy",self.test())            


    @ex.capture
    def test(self):
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images = images.reshape(-1, 28 * 28).to(device)
                labels = labels.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            acc = 100 * correct / total
            print('Accuracy of the network on the 10000 test images: {} %'.format(acc))
            return acc  # SACRED: We return this so that we can add it to our MongoDB

    
    @ex.capture
    def run(self, model_file):
        self.train()


        torch.save(self.model.state_dict(), model_file)
        print('Model saved in {}'.format(model_file))
        ex.add_artifact(model_file)



@ex.config
def get_config():
    input_size = 784
    hidden_size = 500
    num_classes = 10
    num_epochs = 5  # SACRED: Have a look at train_nn.job for an example of how we can change parameter settings
    batch_size = 100
    learning_rate = 0.001
    
    model_file = 'model.ckpt'


@ex.main
def main(_run):

    trainer = Trainer()
    trainer.run()


if __name__ == '__main__':
    ex.run_commandline() 
