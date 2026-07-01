# Importation des modules
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Définition de l'architecture du CNN pour MNIST
class MNISTCNN(nn.Module):
  def __init__(self):
    super(MNISTCNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

    self.pool = nn.MaxPool2d(2, 2)

    self.fc1 = nn.Linear(64 * 7 * 7, 128)
    self.fc2 = nn.Linear(128, 10)

    self.dropout = nn.Dropout(0.25)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))

    x = x.view(-1, 64*7*7)

    x = self.dropout(F.relu(self.fc1(x)))
    x = self.fc2(x)

    return x
  

# Fonction d'entrainement
def train(model, device, train_loader, optimizer, epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)

    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()

    if batch_idx % 200 == 0:
      print(f"époch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\nLoss: {loss.item():.4f}")


# Fonction de test
def test(model, device, test_loader):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += F.cross_entropy(output, target, reduction='sum').item()
      pred = output.argmax(dim=1, keepdim=True)
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)
  accuracy = 100. * correct / len(test_loader.dataset)
  print(f"\nTarget de Test: Moyenne Loss: {test_loss:.4f},\nPrécision (Accuracy): {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")
  return accuracy

# Fonction principale
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entrainement sur : {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("Telechargement du dataset...")
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

model = MNISTCNN().to(device)
optimizer =optim.Adam(model.parameters(), lr=0.001)

best_accuracy = 0.0
for epoch in range(1, 4):
    train(model, device, train_loader, optimizer, epoch)
    accuracy = test(model, device, test_loader)

os.makedirs("modele", exist_ok=True)

if accuracy > best_accuracy:
    best_accuracy = accuracy
    torch.save(model.state_dict(), "modele/mnist_cnn.pth")
    print("Le modele s'est ameliore, sauvegarde effectuee !")