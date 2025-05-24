import os
import json
import torch
import boto3
from io import BytesIO
from PIL import Image
import tarfile
import numpy as np
from torchvision import transforms

# Configuración global
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_fn(model_dir):
    """Carga todos los modelos desde S3 con sus pesos"""
    try:
        # Configuración desde variables de entorno
        model_weights = list(map(float, os.environ.get('MODEL_WEIGHTS', '0.5,0.3,0.2').split(',')))
        model_paths = os.environ.get('MODEL_PATHS', '').split(',')
        
        models = []
        s3 = boto3.client('s3')
        
        for path, weight in zip(model_paths, model_weights):
            # Descargar modelo desde S3
            bucket, key = path.replace("s3://", "").split("/", 1)
            local_path = f"/tmp/{os.path.basename(key)}"
            s3.download_file(bucket, key, local_path)
            
            # Extraer y cargar modelo
            with tarfile.open(local_path) as tar:
                tar.extractall(path="/tmp/model")
            
            # Cargar modelo (ajusta según tus arquitecturas)
            model = torch.load("/tmp/model/model.pth", map_location=device)
            model.eval()
            models.append((model.to(device), weight))
        
        return models
    
    except Exception as e:
        raise RuntimeError(f"Error loading models: {str(e)}")

def input_fn(request_body, request_content_type):
    """Procesa la imagen de entrada"""
    if request_content_type == 'image/jpeg':
        return Image.open(BytesIO(request_body)).convert('RGB')
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, models):
    """Ejecuta la inferencia con los modelos ensamblados"""
    try:
        # Preprocesamiento
        input_tensor = preprocess(input_data).to(device)
        
        # Predicción ponderada
        with torch.no_grad():
            weighted_output = None
            for model, weight in models:
                output = model(input_tensor)
                if weighted_output is None:
                    weighted_output = weight * output
                else:
                    weighted_output += weight * output
        
        return weighted_output
    
    except Exception as e:
        raise RuntimeError(f"Prediction error: {str(e)}")

def output_fn(prediction, accept_type):
    """Formatea la salida"""
    if accept_type != 'application/json':
        raise ValueError(f"Unsupported accept type: {accept_type}")
    
    # Convertir a probabilidades
    probabilities = torch.nn.functional.softmax(prediction, dim=1)
    return json.dumps({
        'predictions': probabilities.cpu().numpy().tolist()
    })

def preprocess(image):
    """Transformaciones consistentes con el entrenamiento"""
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

if __name__ == '__main__':
    # Para pruebas locales
    class MockModel:
        def __init__(self, output_value):
            self.output_value = output_value
        def __call__(self, x):
            return torch.tensor([[self.output_value, -self.output_value]])
    
    # Configurar entorno de prueba
    os.environ['MODEL_WEIGHTS'] = '0.5,0.3,0.2'
    os.environ['MODEL_PATHS'] = 's3://dummy/model1.tar.gz,s3://dummy/model2.tar.gz,s3://dummy/model3.tar.gz'
    
    # Simular modelos
    test_models = [
        (MockModel(1.0), 0.5),
        (MockModel(2.0), 0.3),
        (MockModel(3.0), 0.2)
    ]
    
    # Ejecutar pipeline completo
    img = Image.new('RGB', (256, 256))
    input_data = input_fn(img.tobytes(), 'image/jpeg')
    prediction = predict_fn(input_data, test_models)
    output = output_fn(prediction, 'application/json')
    print("Test output:", output)