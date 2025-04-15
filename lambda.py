
import boto3
import json
from decimal import Decimal

def lambda_handler(event, context):
    runtime_client = boto3.client('runtime.sagemaker')
    endpoint_name = 'Dragon-fruit-endpoint'
    
    try:
        if 'body' not in event:
            raise ValueError("No input data provided in request body")
            
        request_body = json.loads(event['body'])
        
        input_data = request_body.get('input', '')
        
        if not input_data:
            raise ValueError("Input data cannot be empty")
        
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',  
            Body=json.dumps({
                'instances': [input_data]
            })
        )
        
        result = json.loads(response['Body'].read().decode('utf-8'))
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'prediction': result,
                'message': 'Success'
            })
        }
        
    except ValueError as e:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': str(e)})
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Internal server error',
                'details': str(e)
            })
        }