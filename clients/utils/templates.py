# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""prompt templates to share API documentation and instructions with clients"""

# standard documentation and apis template

DOCS = """{prob_desc}
You are provided with the following APIs to interact with the service:


{telemetry_apis}


You are also provided an API to a secure terminal to the service where you can run commands:


{shell_api}


Finally, you will submit your solution for this task using the following API:


{submit_api}


At each turn think step-by-step and respond with:
Thought: <your thought>
Action: <your action>
"""


### custom templates

DOCS_SHELL_ONLY = """{prob_desc}
You are provided with a direct API to a secure terminal to the service where you can run commands:

{shell_api}

Finally, you will submit your solution for this task using the following API:

{submit_api}

IMPORTANT:
1. The submit() call must strictly follow its defined parameter signature for this task.
2. Provide the call in a markdown code block.

At each turn respond with:
Action: <your action>
"""

AUTOGEN_DOCS = """{prob_desc}
You are provided with the following APIs to interact with the service:

{telemetry_apis}

You are also provided an API to a secure terminal to the service where you can run commands:

{shell_api}

Finally, you will submit your solution for this task using the following API:

{submit_api}

Collaborate with your team to analyze the problem and suggest appropriate API calls.
Do not execute commands. Suggest API calls in the specified format within markdown code blocks.
"""
