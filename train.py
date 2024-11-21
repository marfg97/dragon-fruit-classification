

# import argparse
# import os
# import torch
# import torchvision.transforms as transforms
# from torchvision import datasets
# from torch import nn, optim
# from torch.utils.data import DataLoader

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
#     parser.add_argument('--batch_size', type=int, default=32)
#     parser.add_argument('--epochs', type=int, default=10)
#     parser.add_argument('--learning_rate', type=float, default=0.001)
#     args = parser.parse_args()

#     # Definir transformaciones
#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#     ])

#     train_dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)

#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

#     # Definir un modelo simple
#     model = nn.Sequential(
        
#         nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
#         nn.ReLU(),
#         nn.MaxPool2d(kernel_size=2, stride=2),
#         nn.Flatten(),
#         nn.Linear(16 * 64 * 64, 2),  # Ajusta el tamaño de entrada según tu arquitectura
#     )

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

#     # Entrenar el modelo
#     model.train()
#     for epoch in range(args.epochs):
#         for images, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {loss.item():.4f}')

# if __name__ == '__main__':
#     main()


import os
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torch

# Definir una red simple para clasificación
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 32 * 32, num_classes)  # Ajusta según el tamaño de entrada

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 32 * 32)  # Ajusta según el tamaño de entrada
        x = self.fc1(x)
        return x


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Agregar argumentos para parámetros del modelo
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    args = parser.parse_args()

    # Validar que la ruta de datos exista
    if not args.data_dir or not os.path.exists(args.data_dir):
        raise ValueError(f"No se encontraron datos en la ruta: {args.data_dir}")

    print(f"Datos encontrados en: {args.data_dir}")

    # Transformaciones para los datos
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Cargar dataset desde las carpetas anomaly y normal
    dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Configurar dispositivo (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Inicializar el modelo
    num_classes = len(dataset.classes)
    model = SimpleCNN(num_classes=num_classes).to(device)

    # Configurar loss y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Entrenamiento
    for epoch in range(args.epochs):
        train_loss = train(model, dataloader, criterion, optimizer, device)

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Entrenamiento: Loss = {train_loss:.4f}")

    # Guardar el modelo
    model_path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado en: {model_path}")
