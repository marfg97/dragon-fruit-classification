
import boto3
import json
fro

def lambda_handler(event, context):
    runtime_client = boto3.client('runtime.sagemaker')

    endpoint_name = 'Dragon-fruit-endpoint'

    sample = '5.1 , 3.5 , 1.4 , 0.2'

    reponse = runtime_client.invoke_endpoint(EndpointName=endpoint_name,
                                             ContentType='text/csv',
                                             Body=sample)
    
    result = response['Body'].read().decode('ascii')

    print(result)
    return {
        'statusCode' : 200,
        'body' :json.dumps('Hello from Lambda')
    } 