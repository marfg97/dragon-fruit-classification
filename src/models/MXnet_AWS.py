import boto3
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.estimator import Estimator

def train_MXNet():
    # Configuración básica
    role = get_execution_role()
    sess = boto3.Session()
    region = sess.region_name
    bucket = sess.resource('s3').Bucket('your-bucket-name')
    
    # Hiperparámetros para el algoritmo incorporado
    hyperparameters = {
        "num_layers": 18,
        "image_shape": "3,224,224",
        "num_classes": 2,
        "num_training_samples": 2300,
        "mini_batch_size": 48,
        "epochs": 40,
        "learning_rate": 0.0003,
        "use_pretrained_model": 1
    }
    
    # Obtener imagen del algoritmo
    training_image = get_image_uri(region, 'image-classification')
    
    # Crear estimador
    estimator = Estimator(
        training_image,
        role,
        instance_count=1,
        instance_type='ml.p2.xlarge',
        hyperparameters=hyperparameters
    )
    
    # Iniciar entrenamiento
    estimator.fit({'train': 's3://your-data/train.rec',
                   'validation': 's3://your-data/val.rec'})