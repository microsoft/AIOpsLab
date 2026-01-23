import os
from openai import AzureOpenAI
import traceback

# Load environment variables
env_vars = {}
with open('.env', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            env_vars[key] = value.strip('"')

for key, value in env_vars.items():
    os.environ[key] = value

try:
    print('Testing Azure OpenAI configuration...')
    print(f'API Key set: {bool(os.getenv("OPENAI_API_KEY"))}')
    print(f'Endpoint: {os.getenv("AZURE_OPENAI_ENDPOINT")}')
    print(f'API Version: {os.getenv("OPENAI_API_VERSION")}')
    print(f'Deployment: {os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")}')
    
    client = AzureOpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        api_version=os.getenv('OPENAI_API_VERSION', '2023-12-01-preview'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
    )
    
    print('Client created successfully. Testing API call...')
    
    response = client.chat.completions.create(
        model=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4'),
        messages=[{'role': 'user', 'content': 'Hello, just testing the connection.'}],
        max_tokens=10
    )
    
    print('SUCCESS! Response:', response.choices[0].message.content)
    
except Exception as e:
    print(f'ERROR: {str(e)}')
    print('Full traceback:')
    traceback.print_exc()
