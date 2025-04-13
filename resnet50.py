import os
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
import torch

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
    parser.add_argument("--data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    args = parser.parse_args()

    # Transformaciones para los datos (ajustadas para ResNet50)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Cargar dataset
    dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Configurar dispositivo (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar ResNet50 pre-entrenada
    model = models.resnet50(pretrained=True)
    
    # Congelar todos los parámetros excepto la capa final
    for param in model.parameters():
        param.requires_grad = False
    
    # Reemplazar la capa final para que coincida con nuestro número de clases
    num_classes = len(dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Mover modelo al dispositivo
    model = model.to(device)

    # Configurar loss y optimizador (solo optimizamos la capa final)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)

    # Entrenamiento
    for epoch in range(args.epochs):
        train_loss = train(model, dataloader, criterion, optimizer, device)

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Entrenamiento: Loss = {train_loss:.4f}")

    # Guardar el modelo (guardamos todo el modelo, no solo state_dict)
    model_path = os.path.join(args.model_dir, "model.pth")
    torch.save(model, model_path)
    print(f"Modelo guardado en: {model_path}")