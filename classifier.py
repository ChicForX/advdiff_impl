import torch
import torch.nn as nn
import torch.optim as optim
import os


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_net = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_net(x)
        return x

    def train_model(self, device, train_loader, val_loader=None, epochs=10, lr=0.001):
        checkpoint_path = 'results/classifier.pt'
        if os.path.exists(checkpoint_path):
            print("Loading classifier from checkpoint...")
            state = torch.load(checkpoint_path, map_location=device)
            self.load_state_dict(state['model_state_dict'])
            print("Loading classifier completed!")
            return

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        self.to(device)

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if batch_idx % 100 == 99:  # Print every 100 mini-batches
                    print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {running_loss / 100:.4f}')
                    running_loss = 0.0

            if val_loader:
                self.evaluate_model(device, val_loader)

        if not os.path.isdir('results'):
            os.makedirs('results')
        torch.save({
            'model_state_dict': self.state_dict(),
        }, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

    def evaluate_model(self, device, val_loader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = self(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        print(f'Accuracy on the test set: {100 * correct / total:.2f}%')
