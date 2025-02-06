# AIOpsLab Clients

This directory contains implementation of clients that can interact with AIOPsLab.
These clients are some baselines that we have implemented and evaluated to help you get started.

## Clients

- [GPT](/clients/gpt.py): A naive GPT4-based LLM agent with only shell access.
- [ReAct](/clients/react.py): A naive LLM agent that uses the ReAct framework.
- [FLASH](/clients/flash.py): A naive LLM agent that uses status supervision and hindsight integration components to ensure the high reliability of workflow execution.

<!-- 
Note: The script [GPT-managed-identity](/clients/gpt_managed_identity.py) uses the `DefaultAzureCredential` method from the `azure-identity` package to authenticate. This method simplifies authentication by supporting various credential types, including managed identities.

We recommend using a [user-assigned managed identity](https://learn.microsoft.com/en-us/entra/identity/managed-identities-azure-resources/how-manage-user-assigned-managed-identities?pivots=identity-mi-methods-azp) for this setup. Ensure the following steps are completed:

1. **Role Assignment**: Assign the managed identity appropriate roles:
   - A role that provides read access to the VM, such as the built-in **Reader** role.
   - A role that grants read/write access to the Azure OpenAI Service, such as the **Azure AI Developer** role.

2. **Attach the Managed Identity to the Controller VM**:  
   Follow the steps in the official documentation to add the managed identity to the VM:  
   [Add a user-assigned managed identity to a VM](https://learn.microsoft.com/en-us/entra/identity/managed-identities-azure-resources/how-to-configure-managed-identities?pivots=qs-configure-portal-windows-vm#user-assigned-managed-identity).

Please ensure the required Azure configuration is provided using the /configs/example_azure_config.yml file, or use it as a template to create a new configuration file

### Useful Links
1. [How to configure Azure OpenAI Service with Microsoft Entra ID authentication](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/managed-identity)  
2. [Azure Identity client library for Python](https://learn.microsoft.com/en-us/python/api/overview/azure/identity-readme?view=azure-python#defaultazurecredential) -->
