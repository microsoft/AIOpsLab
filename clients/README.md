# AIOpsLab Clients

This directory contains implementation of clients that can interact with AIOPsLab.
These clients are some baselines that we have implemented and evaluated to help you get started.

## Clients

- [GPT](/clients/gpt.py): A naive GPT series LLM agent with only shell access.
- [DeepSeek](/clients/deepseek.py): A naive DeepSeek series LLM agent with only shell access.
- [Qwen](/clients/qwen.py): A naive Qwen series LLM agent with only shell access.
- [vLLM](/clients/vllm.py): A naive vLLM agent with any open source LLM deployed locally and only shell access.
- [GPT with Azure OpenAI](/clients/gpt_azure_identity.py): A naive GPT4-based LLM agent for Azure OpenAI, using identity-based authentication.
- [ReAct](/clients/react.py): A naive LLM agent that uses the ReAct framework.
- [FLASH](/clients/flash.py): A naive LLM agent that uses status supervision and hindsight integration components to ensure the high reliability of workflow execution.
- [OpenRouter](/clients/openrouter.py): A naive OpenRouter LLM agent with only shell access.

### Using the vLLM Client

The vLLM client allows you to run local open-source models as an agent for AIOpsLab tasks. This approach is particularly useful when you want to:
- Use your own hardware for inference
- Experiment with different open-source models
- Work in environments without internet access to cloud LLM providers

#### Quick Setup Guide

1. **Launch the vLLM server**:
    ```bash
    # Make the script executable
    chmod +x ./clients/launch_vllm.sh

    # Run the script
    ./clients/launch_vllm.sh
    ```
    This will launch vLLM in the background using the default model (Qwen/Qwen2.5-3B-Instruct).

2. **Check server status**:
    ```bash
    # View the log file to confirm the server is running
    cat vllm_Qwen_Qwen2.5-3B-Instruct.log
    ```

3. **Customize the model** (optional):
    Edit `launch_vllm.sh` to change the model:
    ```bash
    # Open the file
    nano ./clients/launch_vllm.sh

    # Change the MODEL variable to your preferred model
    # Example: MODEL="mistralai/Mistral-7B-Instruct-v0.1"
    ```

4. **Run the vLLM agent**:
    ```
    python clients/vllm.py
    ```

#### Requirements

- Poetry for dependency management
- Sufficient GPU resources for your chosen model
- The model must support the OpenAI chat completion API format

#### Advanced Configuration

The vLLM client connects to `http://localhost:8000/v1` by default. If you've configured vLLM to use a different port or host, update the base_url in `clients/utils/llm.py` in the vLLMClient class.

## Environment Variables

The following environment variables are used by various clients. Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
# Edit .env with your actual API keys
```

### API Keys (Required for respective clients)
- `OPENAI_API_KEY`: OpenAI API key for GPT clients
- `DEEPSEEK_API_KEY`: DeepSeek API key for DeepSeek client
- `DASHSCOPE_API_KEY`: Alibaba DashScope API key for Qwen client
- `OPENROUTER_API_KEY`: OpenRouter API key for OpenRouter client
- `GROQ_API_KEY`: Groq API key for Groq-based clients

### Optional Configuration
- `OPENROUTER_MODEL`: OpenRouter model to use (default: `openai/gpt-4o-mini`)
- `USE_WANDB`: Enable Weights & Biases logging (default: `false`)

### Keyless Authentication

The script [`gpt_azure_identity.py`](/clients/gpt_azure_identity.py) supports keyless authentication for **securely** accessing Azure OpenAI endpoints. It supports two authentication methods:

- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/?view=azure-cli-latest)
- [User-assigned managed identity](https://learn.microsoft.com/en-us/entra/identity/managed-identities-azure-resources/how-manage-user-assigned-managed-identities?pivots=identity-mi-methods-azp)

#### 1. Azure CLI Authentication

**Steps**
- The user must have the appropriate role assigned (e.g., `Cognitive Services OpenAI User`) on the Azure OpenAI resource.
- Run the following command to authenticate ([How to install the Azure CLI?](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest)):

```bash
az login --scope https://cognitiveservices.azure.com/.default
```

#### 2. Managed Identity Authentication

- Follow the official documentation to assign a user-assigned managed identity to the VM where the client script would be run:  
[Add a user-assigned managed identity to a VM](https://learn.microsoft.com/en-us/entra/identity/managed-identities-azure-resources/how-to-configure-managed-identities?pivots=qs-configure-portal-windows-vm#user-assigned-managed-identity)
- The managed identity must have the appropriate role assigned (e.g., `Cognitive Services OpenAI User`) on the Azure OpenAI resource.
- Specify the managed identity to use by setting the following environment variable before running the script:

```bash
export AZURE_CLIENT_ID=<client-id>
```

Please ensure the required Azure configuration is provided using the /configs/example_azure_config.yml file, or use it as a template to create a new configuration file.

### Useful Links
1. [How to configure Azure OpenAI Service with Microsoft Entra ID authentication](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/managed-identity)  
2. [Azure Identity client library for Python](https://learn.microsoft.com/en-us/python/api/overview/azure/identity-readme?view=azure-python#defaultazurecredential)
