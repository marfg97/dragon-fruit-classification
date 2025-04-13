import os
import argparse
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np 
import datetime


# Definir una red simple para clasificación
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 32 * 32, num_classes)  

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 32 * 32)  
        x = self.fc1(x)
        return x



def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Métricas para clases desbalanceadas
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1])
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'loss': running_loss / len(dataloader),
        'accuracy': (np.array(all_preds) == np.array(all_labels)).mean(),
        'precision_anomaly': precision[0],
        'recall_anomaly': recall[0],
        'f1_anomaly': f1[0],
        'precision_normal': precision[1],
        'recall_normal': recall[1],
        'f1_normal': f1[1],
        'confusion_matrix': cm.tolist()
    }
    return metrics


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

def save_model(model, path, class_to_idx, input_size, metrics=None):
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'input_size': input_size,
        'metrics': metrics or {},
        'classes': list(class_to_idx.keys()),
        'timestamp': datetime.now().isoformat()
    }, path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Agregar argumentos para parámetros del modelo
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    args = parser.parse_args()

    # Validar la ruta de datos 
    if not args.data_dir or not os.path.exists(args.data_dir):
        raise ValueError(f"Not Found Data in: {args.data_dir}")

    print(f"Data Found in: {args.data_dir}")

    # Transformaciones para los datos
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Cargar dataset desde las carpetas a
    dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Inicializar el modelo
    num_classes = len(dataset.classes)
    model = SimpleCNN(num_classes=num_classes).to(device)

    # Configurar loss y optimizador
    class_counts = [0] * len(dataset.classes)
    for _, label in dataset.samples:
        class_counts[label] += 1

    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    # Entrenamiento
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Ajustar learning rate basado en F1-score
        scheduler.step(val_metrics['f1_anomaly'])
        
        # Guardar el mejor modelo basado en F1 para anomalías
        if val_metrics['f1_anomaly'] > best_f1:
            best_f1 = val_metrics['f1_anomaly']
            save_model(
                model, 
                os.path.join(args.model_dir, "best_model.pth"),
                dataset.class_to_idx,
                (3, 64, 64),
                val_metrics
            )

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"TrainLoss: {train_loss:.4f}")
        print(f"ValLoss: {val_metrics['loss']:.4f}")
        print(f"ValAcc: {val_metrics['accuracy']:.4f}")
        print(f"ValF1Anomaly: {val_metrics['f1_anomaly']:.4f}")
        print(f"ValRecallAnomaly: {val_metrics['recall_anomaly']:.4f}")

    # Guardar el modelo
    model_path = os.path.join(args.model_dir, "model.pth")
    
    print(f"Modelo guardado en: {model_path}")

    test_metrics = evaluate(model, test_loader, criterion, device)
    save_model(
        model, 
        model_path,
        dataset.class_to_idx,
        (3, 64, 64),
        test_metrics
    )