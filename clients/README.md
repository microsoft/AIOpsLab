# AIOpsLab Clients

This directory contains implementation of clients that can interact with AIOPsLab.
These clients are some baselines that we have implemented and evaluated to help you get started.

## Clients

- [GPT](/clients/gpt.py): A naive GPT4-based LLM agent with only shell access.
- [GPT with Azure OpenAI](/clients/gpt_azure_identity.py): A naive GPT4-based LLM agent for Azure OpenAI, using identity-based authentication.
- [ReAct](/clients/react.py): A naive LLM agent that uses the ReAct framework.
- [FLASH](/clients/flash.py): A naive LLM agent that uses status supervision and hindsight integration components to ensure the high reliability of workflow execution.

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
