<div align="center">

<h1>AIOpsLab</h1>

[🤖Overview](#🤖overview) | 
[🚀Quick Start](#🚀quickstart) | 
[📦Installation](#📦installation) | 
[⚙️Usage](#⚙️usage) | 
[📂Project Structure](#📂project-structure) |
[📄How to Cite](#📄how-to-cite)

[![ArXiv Link](https://img.shields.io/badge/arXiv-2501.06706-red?logo=arxiv)](https://arxiv.org/pdf/2501.06706)
[![ArXiv Link](https://img.shields.io/badge/arXiv-2407.12165-red?logo=arxiv)](https://arxiv.org/pdf/2407.12165)
</div>



<h2 id="🤖overview">🤖 Overview</h2>

![alt text](./assets/images/aiopslab-arch-open-source.png)


AIOpsLab is a holistic framework to enable the design, development, and evaluation of autonomous AIOps agents that, additionally, serve the purpose of building reproducible, standardized, interoperable and scalable benchmarks. AIOpsLab can deploy microservice cloud environments, inject faults, generate workloads, and export telemetry data, while orchestrating these components and providing interfaces for interacting with and evaluating agents. 

Moreover, AIOpsLab provides a built-in benchmark suite with a set of problems to evaluate AIOps agents in an interactive environment. This suite can be easily extended to meet user-specific needs. See the problem list [here](/aiopslab/orchestrator/problems/registry.py#L15).

<h2 id="📦installation">📦 Installation</h2>

### Requirements
For specific platform and troubleshooting instructions, please see [SETUP.md](./SETUP.md).

Recommended installation:
```bash
sudo apt install python3.11 python3.11-venv python3.11-dev python3-pip # poetry requires python >= 3.11
```

We recommend [Poetry](https://python-poetry.org/docs/) for managing dependencies. You can also use a standard `pip install -e .` to install the dependencies.

```bash
git clone --recurse-submodules <CLONE_PATH_TO_THE_REPO>
cd AIOpsLab
poetry env use python3.11
export PATH="$HOME/.local/bin:$PATH" # export poetry to PATH if needed
poetry install # -vvv for verbose output
poetry self add poetry-plugin-shell # installs poetry shell plugin
poetry shell
```

<h2 id="🚀quickstart">🚀 Quick Start </h2>

<!-- TODO: Add instructions for both local cluster and remote cluster -->
Choose either a) or b) to set up your cluster and then proceed to the next steps.

### a) Local simulated cluster
AIOpsLab can be run on a local simulated cluster using [kind](https://kind.sigs.k8s.io/) on your local machine. Please look at this [README](kind/README.md#prerequisites) for a list of prerequisites.

```bash
# For x86 machines
kind create cluster --config kind/kind-config-x86.yaml

# For ARM machines
kind create cluster --config kind/kind-config-arm.yaml
```

If you're running into issues, consider building a Docker image for your machine by following this [README](kind/README.md#deployment-steps). Please also open an issue.

After finishing cluster creation, proceed to the next "Update `config.yml`" step.

### b) Remote cluster
AIOpsLab supports any remote kubernetes cluster that your `kubectl` context is set to, whether it's a cluster from a cloud provider or one you build yourself. We have some Ansible playbooks we have to setup clusters on providers like [CloudLab](https://www.cloudlab.us/) and our own machines. Follow this [README](./scripts/ansible/README.md) to set up your own cluster, and then proceed to the next "Update `config.yml`" step.

### Update `config.yml`
```bash
cd aiopslab
cp config.yml.example config.yml
```
Update your `config.yml` so that `k8s_host` is the host name of the control plane node of your cluster. Update `k8s_user` to be your username on the control plane node. If you are using a kind cluster, your `k8s_host` should be `kind`. If you're running AIOpsLab on cluster, your `k8s_host` should be `localhost`.

### Running agents
Human as the agent:

```bash
python3 cli.py
(aiopslab) $ start misconfig_app_hotel_res-detection-1 # or choose any problem you want to solve
# ... wait for the setup ...
(aiopslab) $ submit("Yes") # submit solution
```

Run GPT-4 baseline agent:

```bash
export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
python3 clients/gpt.py # you can also change the problem to solve in the main() function
```

You can check the running status of the cluster using [k9s](https://k9scli.io/) or other cluster monitoring tools conveniently.

## 🔧 Azure OpenAI Integration Setup

This section documents the modifications needed to run AIOpsLab with Azure OpenAI instead of standard OpenAI.

### Prerequisites for Azure OpenAI

1. **Azure OpenAI Resource**: You need an Azure OpenAI resource with a deployed model (e.g., GPT-4)
2. **Environment Configuration**: Set up proper environment variables for Azure OpenAI authentication

### Environment Configuration

Create a `.env` file in the project root with your Azure OpenAI credentials:

```bash
# Environment variables for AIOpsLab - Azure OpenAI Configuration
OPENAI_API_KEY="your-azure-openai-api-key"
OPENAI_API_BASE="https://your-resource-name.openai.azure.com/"
OPENAI_API_TYPE="azure"
OPENAI_API_VERSION="2025-01-01-preview"
AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4"
```

### Required Code Modifications

The following files have been modified to support Azure OpenAI:

#### 1. **LLM Client Updates** (`clients/utils/llm.py`)
- Added Azure OpenAI client support alongside standard OpenAI
- Automatic detection of Azure vs standard OpenAI based on environment variables
- Proper model name mapping for Azure deployments

#### 2. **Evaluation System Updates** (`aiopslab/orchestrator/evaluators/qualitative.py`)
- Fixed authentication to use `OPENAI_API_KEY` instead of `OPENAI_KEY`
- Added Azure OpenAI support for qualitative evaluation
- Proper error handling for Azure endpoints

#### 3. **Storage Class Fixes** (`fix-storage-classes.yaml`)
Added missing storage classes required for MongoDB pods:
```yaml
# Created storage classes: geo-storage, profile-storage, rate-storage, 
# recommendation-storage, reservation-storage, user-storage
# All pointing to openebs.io/local provisioner
```

#### 4. **Environment Loading**
Load environment variables in PowerShell:
```powershell
Get-Content .env | ForEach-Object { 
    if($_ -match '^([^#][^=]+)=(.+)$') { 
        $name = $matches[1]; $value = $matches[2].Trim('"'); 
        [Environment]::SetEnvironmentVariable($name, $value, 'Process') 
    } 
}
```

### Running with Azure OpenAI

1. **Set up environment variables** (load `.env` file as shown above)
2. **Enable qualitative evaluation** in `config.yml`:
   ```yaml
   qualitative_eval: true
   ```
3. **Run agents normally**:
   ```bash
   # GPT Agent
   poetry run python clients/gpt.py
   
   # Flash Agent (single scenario)
   poetry run python test_flash_single.py
   
   # Flash Agent (all scenarios)
   poetry run python clients/flash.py
   ```

### Troubleshooting Common Issues

#### Storage Issues
If pods get stuck in `Pending` state due to PVC issues:
```bash
kubectl apply -f AIOpsLab/troubleshooting/fix-storage-classes.yaml
```

#### Azure OpenAI Authentication Errors
- Verify API key is correct and not expired
- Ensure deployment name matches your Azure OpenAI Studio deployment
- Check that API version is supported

#### Environment Variable Loading
Make sure to reload environment variables in each new PowerShell session:
```powershell
# Load from .env file
Get-Content .env | ForEach-Object { if($_ -match '^([^#][^=]+)=(.+)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2].Trim('"'), 'Process') } }
```

### Logging and Debugging

Save agent execution logs for analysis:
```powershell
# Save output to file
poetry run python clients/gpt.py > gpt_output.log 2>&1

# Save with display (using Tee-Object)
poetry run python test_flash_single.py | Tee-Object -FilePath flash_output.log
```

**Security Note**: The `.gitignore` file has been updated to exclude all log files and environment files to prevent accidental commit of sensitive information.

## 🔧 Complete Setup and Troubleshooting Guide

This section documents all the changes and fixes needed to run AIOpsLab successfully on Windows with kind cluster and Azure OpenAI.

### 📋 Prerequisites Checklist

Before starting, ensure you have:
- ✅ Python 3.11+ installed
- ✅ Poetry installed and configured
- ✅ Docker Desktop running
- ✅ kind installed and available in PATH
- ✅ kubectl installed and available in PATH
- ✅ Helm installed and available in PATH
- ✅ Azure OpenAI resource with deployed model (e.g., GPT-4)

### 🚀 Complete Setup Steps

#### 1. **Initial Repository Setup**
```bash
# Clone with submodules
git clone --recurse-submodules <repository-url>
cd AIOpsLab

# Install dependencies
poetry install
poetry shell
```

#### 2. **Kubernetes Cluster Setup**
```bash
# Create kind cluster (choose based on your architecture)
kind create cluster --config kind/kind-config-x86.yaml  # for x86
# OR
kind create cluster --config kind/kind-config-arm.yaml  # for ARM

# Verify cluster is running
kubectl cluster-info
```

#### 3. **Initialize Git Submodules** (if not done during clone)
```bash
git submodule init
git submodule update
```

#### 4. **Deploy Core Infrastructure**

**Install Helm and add repositories:**
```bash
# Add required Helm repositories
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add openebs https://openebs.github.io/charts
helm repo update

# Install OpenEBS (required for storage)
helm install openebs openebs/openebs --namespace openebs --create-namespace
```

**Install Prometheus monitoring stack:**
```bash
helm install prometheus prometheus-community/kube-prometheus-stack --namespace monitoring --create-namespace
```

#### 5. **Fix Storage Class Issues**

Create and apply storage class fix:
```bash
# Apply the storage class fix
kubectl apply -f AIOpsLab/troubleshooting/fix-storage-classes.yaml
```

The `fix-storage-classes.yaml` file creates the following storage classes:
- geo-storage
- profile-storage  
- rate-storage
- recommendation-storage
- reservation-storage
- user-storage

All pointing to `openebs.io/local` provisioner.

#### 6. **Deploy Hotel Reservation Application**
```bash
# Deploy the hotel reservation microservices
kubectl apply -f aiopslab-applications/hotelReservation/kubernetes/
```

#### 7. **Environment Configuration**

Create `.env` file with Azure OpenAI credentials:
```bash
# .env file content
OPENAI_API_KEY="your-azure-openai-api-key"
OPENAI_API_BASE="https://your-resource-name.openai.azure.com/"
OPENAI_API_TYPE="azure"
OPENAI_API_VERSION="2025-01-01-preview"
AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4"
```

**Load environment variables in PowerShell:**
```powershell
Get-Content .env | ForEach-Object { 
    if($_ -match '^([^#][^=]+)=(.+)$') { 
        $name = $matches[1]; $value = $matches[2].Trim('"'); 
        [Environment]::SetEnvironmentVariable($name, $value, 'Process') 
    } 
}
```

#### 8. **Configure AIOpsLab**
```bash
cd aiopslab
cp config.yml.example config.yml
```

Update `config.yml`:
- Set `k8s_host: kind` (for kind cluster)
- Set `k8s_user: <your-username>`
- Set `qualitative_eval: true` (to enable Azure OpenAI evaluation)

### 🔧 Key Code Modifications Made

#### 1. **Azure OpenAI Support** (`clients/utils/llm.py`)
- Added detection for Azure vs standard OpenAI configuration
- Implemented proper Azure OpenAI client initialization
- Added support for deployment name mapping

#### 2. **Evaluation System Fix** (`aiopslab/orchestrator/evaluators/qualitative.py`)
- Fixed authentication to use correct environment variable (`OPENAI_API_KEY`)
- Added Azure OpenAI client support for evaluation system
- Enhanced error handling for Azure endpoints

#### 3. **Storage Infrastructure** (`fix-storage-classes.yaml`)
Created missing storage classes required for MongoDB persistence:
```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: geo-storage
provisioner: openebs.io/local
# ... (similar for other storage classes)
```

#### 4. **Security Enhancements** (`.gitignore`)
Added patterns to exclude sensitive files:
```gitignore
# Environment files
.env*
!.env.example

# Log files
*.log
*_output.log

# Common secret patterns
*secret*
*key*
*token*
```

### 🧪 Testing Agent Execution

#### Test GPT Agent:
```bash
# Load environment first
Get-Content .env | ForEach-Object { if($_ -match '^([^#][^=]+)=(.+)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2].Trim('"'), 'Process') } }

# Run GPT agent
poetry run python clients/gpt.py
```

#### Test Flash Agent (single scenario):
```bash
# Run single Flash scenario
poetry run python test_flash_single.py
```

#### Test Flash Agent (all scenarios):
```bash
# Run all Flash scenarios
poetry run python clients/flash.py
```

### 🐛 Common Issues and Solutions

#### **Pod Stuck in Pending State**
**Problem**: MongoDB pods stuck in Pending due to missing storage classes
**Solution**: Apply storage class fix
```bash
kubectl apply -f fix-storage-classes.yaml
```

#### **Azure OpenAI Authentication Errors**
**Problem**: "Invalid API key" or "Resource not found"
**Solution**: 
- Verify API key is correct and active
- Ensure deployment name matches Azure OpenAI Studio
- Check API version compatibility

#### **Environment Variables Not Loading**
**Problem**: Azure OpenAI credentials not recognized
**Solution**: Reload environment in each new PowerShell session:
```powershell
Get-Content .env | ForEach-Object { if($_ -match '^([^#][^=]+)=(.+)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2].Trim('"'), 'Process') } }
```

#### **Git Submodules Missing**
**Problem**: Application deployments fail due to missing charts
**Solution**: Initialize and update submodules:
```bash
git submodule init
git submodule update
```

#### **Helm Chart Dependencies**
**Problem**: Charts fail to install due to missing repositories
**Solution**: Add and update Helm repositories:
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add openebs https://openebs.github.io/charts
helm repo update
```

### 📊 Verification Commands

Check cluster status:
```bash
# Check all pods are running
kubectl get pods --all-namespaces

# Check storage classes
kubectl get storageclass

# Check persistent volume claims
kubectl get pvc --all-namespaces

# Monitor pod status with watch
kubectl get pods -w
```

Check Helm deployments:
```bash
# List all Helm releases
helm list --all-namespaces

# Check Prometheus status
helm status prometheus -n monitoring

# Check OpenEBS status  
helm status openebs -n openebs
```

### 📝 Logging and Output Capture

Save agent execution logs:
```powershell
# Save with output display
poetry run python clients/gpt.py | Tee-Object -FilePath gpt_output.log

# Save without display  
poetry run python test_flash_single.py > flash_output.log 2>&1
```

### 🔒 Security Best Practices

1. **Never commit sensitive files**: `.env`, log files, and API keys are excluded via `.gitignore`
2. **Check git status before commits**: Verify no secrets are staged
3. **Use environment variables**: Store all credentials in `.env` file
4. **Review logs before sharing**: Use search tools to check for sensitive data

<h2 id="⚙️usage">⚙️ Usage</h2>

AIOpsLab can be used in the following ways:
For detailed instructions on developing and testing agents, see [clients/README_AGENTS.md](./clients/README_AGENTS.md).


### How to onboard your agent to AIOpsLab?

AIOpsLab makes it extremely easy to develop and evaluate your agents. You can onboard your agent to AIOpsLab in 3 simple steps:

1. **Create your agent**: You are free to develop agents using any framework of your choice. The only requirements are:
    - Wrap your agent in a Python class, say `Agent`
    - Add an async method `get_action` to the class:

        ```python
        # given current state and returns the agent's action
        async def get_action(self, state: str) -> str:
            # <your agent's logic here>
        ```

2. **Register your agent with AIOpsLab**: You can now register the agent with AIOpsLab's orchestrator. The orchestrator will manage the interaction between your agent and the environment:

    ```python
    from aiopslab.orchestrator import Orchestrator

    agent = Agent()             # create an instance of your agent
    orch = Orchestrator()       # get AIOpsLab's orchestrator
    orch.register_agent(agent)  # register your agent with AIOpsLab
    ```

3. **Evaluate your agent on a problem**:

    1. **Initialize a problem**: AIOpsLab provides a list of problems that you can evaluate your agent on. Find the list of available problems [here](/aiopslab/orchestrator/problems/registry.py) or using `orch.probs.get_problem_ids()`. Now initialize a problem by its ID: 

        ```python
        problem_desc, instructs, apis = orch.init_problem("k8s_target_port-misconfig-mitigation-1")
        ```
    
    2. **Set agent context**: Use the problem description, instructions, and APIs available to set context for your agent. (*This step depends on your agent's design and is left to the user*)


    3. **Start the problem**: Start the problem by calling the `start_problem` method. You can specify the maximum number of steps too:

        ```python
        import asyncio
        asyncio.run(orch.start_problem(max_steps=30))
        ```

This process will create a [`Session`](/aiopslab/session.py) with the orchestrator, where the agent will solve the problem. The orchestrator will evaluate your agent's solution and provide results (stored under `data/results/`). You can use these to improve your agent.


### How to add new applications to AIOpsLab?

AIOpsLab provides a default [list of applications](/aiopslab/service/apps/) to evaluate agents for operations tasks. However, as a developer you can add new applications to AIOpsLab and design problems around them.

> *Note*: for auto-deployment of some apps with K8S, we integrate Helm charts (you can also use `kubectl` to install as [HotelRes application](/aiopslab/service/apps/hotelres.py)). More on Helm [here](https://helm.sh).

To add a new application to AIOpsLab with Helm, you need to:

1. **Add application metadata**
    - Application metadata is a JSON object that describes the application.
    - Include *any* field such as the app's name, desc, namespace, etc.
    - We recommend also including a special `Helm Config` field, as follows:

        ```json
        "Helm Config": {
            "release_name": "<name for the Helm release to deploy>",
            "chart_path": "<path to the Helm chart of the app>",
            "namespace": "<K8S namespace where app should be deployed>"
        }
        ```
        > *Note*: The `Helm Config` is used by the orchestrator to auto-deploy your app when a problem associated with it is started.

        > *Note*: The orchestrator will auto-provide *all other* fields as context to the agent for any problem associated with this app.

    Create a JSON file with this metadata and save it in the [`metadata`](/aiopslab/service/metadata) directory. For example the `social-network` app: [social-network.json](/aiopslab/service/metadata/social-network.json)

2. **Add application class**

    Extend the base class in a new Python file in the [`apps`](/aiopslab/service/apps) directory:

    ```python
    from aiopslab.service.apps.base import Application

    class MyApp(Application):
        def __init__(self):
            super().__init__("<path to app metadata JSON>")
    ```

    The `Application` class provides a base implementation for the application. You can override methods as needed and add new ones to suit your application's requirements, but the base class should suffice for most applications.



### How to add new problems to AIOpsLab?

Similar to applications, AIOpsLab provides a default [list of problems](/aiopslab/orchestrator/problems/registry.py) to evaluate agents. However, as a developer you can add new problems to AIOpsLab and design them around your applications.

Each problem in AIOpsLab has 5 components:
1. *Application*: The application on which the problem is based.
2. *Task*: The AIOps task that the agent needs to perform.
 Currently we support: [Detection](/aiopslab/orchestrator/tasks/detection.py), [Localization](/aiopslab/orchestrator/tasks/localization.py), [Analysis](/aiopslab/orchestrator/tasks/analysis.py), and [Mitigation](/aiopslab/orchestrator/tasks/mitigation.py).
3. *Fault*: The fault being introduced in the application.
4. *Workload*: The workload that is generated for the application.
5. *Evaluator*: The evaluator that checks the agent's performance.

To add a new problem to AIOpsLab, create a new Python file 
in the [`problems`](/aiopslab/orchestrator/problems) directory, as follows:

1. **Setup**. Import your chosen application (say `MyApp`) and task (say `LocalizationTask`):

    ```python
    from aiopslab.service.apps.myapp import MyApp
    from aiopslab.orchestrator.tasks.localization import LocalizationTask
    ```

2. **Define**. To define a problem, create a class that inherits from your chosen `Task`, and defines 3 methods: `start_workload`, `inject_fault`, and `eval`:

    ```python
    class MyProblem(LocalizationTask):
        def __init__(self):
            self.app = MyApp()
        
        def start_workload(self):
            # <your workload logic here>
        
        def inject_fault(self)
            # <your fault injection logic here>
        
        def eval(self, soln, trace, duration):
            # <your evaluation logic here>
    ```

3. **Register**. Finally, add your problem to the orchestrator's registry [here](/aiopslab/orchestrator/problems/registry.py).


See a full example of a problem [here](/aiopslab/orchestrator/problems/k8s_target_port_misconfig/target_port.py). 
<details>
  <summary>Click to show the description of the problem in detail</summary>

- **`start_workload`**: Initiates the application's workload. Use your own generator or AIOpsLab's default, which is based on [wrk2](https://github.com/giltene/wrk2):

    ```python
    from aiopslab.generator.workload.wrk import Wrk

    wrk = Wrk(rate=100, duration=10)
    wrk.start_workload(payload="<wrk payload script>", url="<app URL>")
    ```
    > Relevant Code: [aiopslab/generators/workload/wrk.py](/aiopslab/generators/workload/wrk.py)

- **`inject_fault`**: Introduces a fault into the application. Use your own injector or AIOpsLab's built-in one which you can also extend. E.g., a misconfig in the K8S layer:

    ```python
    from aiopslab.generators.fault.inject_virtual import *

    inj = VirtualizationFaultInjector(testbed="<namespace>")
    inj.inject_fault(microservices=["<service-name>"], fault_type="misconfig")
    ```

    > Relevant Code: [aiopslab/generators/fault](/aiopslab/generators/fault)


- **`eval`**: Evaluates the agent's solution using 3 params: (1) *soln*: agent's submitted solution if any, (2) *trace*: agent's action trace, and (3) *duration*: time taken by the agent.

    Here, you can use built-in default evaluators for each task and/or add custom evaluations. The results are stored in `self.results`:
    ```python
    def eval(self, soln, trace, duration) -> dict:
        super().eval(soln, trace, duration)     # default evaluation
        self.add_result("myMetric", my_metric(...))     # add custom metric
        return self.results
    ```

    > *Note*: When an agent starts a problem, the orchestrator creates a [`Session`](/aiopslab/session.py) object that stores the agent's interaction. The `trace` parameter is this session's recorded trace.

    > Relevant Code: [aiopslab/orchestrator/evaluators/](/aiopslab/orchestrator/evaluators/)

</details>




<h2 id="📂project-structure">📂 Project Structure</h2>

<summary><code>aiopslab</code></summary>
<details>
  <summary>Generators</summary>
  <pre>
  generators - the problem generators for aiopslab
  ├── fault - the fault generator organized by fault injection level
  │   ├── base.py
  │   ├── inject_app.py
  │  ...
  │   └── inject_virtual.py
  └── workload - the workload generator organized by workload type
      └── wrk.py - wrk tool interface
  </pre>
</details>

<details>
  <summary>Orchestrator</summary>
  <pre>
  orchestrator
  ├── orchestrator.py - the main orchestration engine
  ├── parser.py - parser for agent responses
  ├── evaluators - eval metrics in the system
  │   ├── prompts.py - prompts for LLM-as-a-Judge
  │   ├── qualitative.py - qualitative metrics
  │   └── quantitative.py - quantitative metrics
  ├── problems - problem definitions in aiopslab
  │   ├── k8s_target_port_misconfig - e.g., A K8S TargetPort misconfig problem
  │  ...
  │   └── registry.py
  ├── actions - actions that agents can perform organized by AIOps task type
  │   ├── base.py
  │   ├── detection.py
  │   ├── localization.py
  │   ├── analysis.py
  │   └── mitigation.py
  └── tasks - individual AIOps task definition that agents need to solve
      ├── base.py
      ├── detection.py
      ├── localization.py
      ├── analysis.py
      └── mitigation.py
  </pre>
</details>

<details>
  <summary>Service</summary>
  <pre>
  service
  ├── apps - interfaces/impl. of each app
  ├── helm.py - helm interface to interact with the cluster
  ├── kubectl.py - kubectl interface to interact with the cluster
  ├── shell.py - shell interface to interact with the cluster
  ├── metadata - metadata and configs for each apps
  └── telemetry - observability tools besides observer, e.g., in-memory log telemetry for the agent
  </pre>
</details>

<details>
  <summary>Observer</summary>
  <pre>
  observer
  ├── filebeat - Filebeat installation
  ├── logstash - Logstash installation
  ├── prometheus - Prometheus installation
  ├── log_api.py - API to store the log data on disk
  ├── metric_api.py - API to store the metrics data on disk
  └── trace_api.py - API to store the traces data on disk
  </pre>
</details>

<details>
  <summary>Utils</summary>
  <pre>
  ├── config.yml - aiopslab configs
  ├── config.py - config parser
  ├── paths.py - paths and constants
  ├── session.py - aiopslab session manager
  └── utils
      ├── actions.py - helpers for actions that agents can perform
      ├── cache.py - cache manager
      └── status.py - aiopslab status, error, and warnings
  </pre>
</details>

<summary><code>cli.py</code>: A command line interface to interact with AIOpsLab, e.g., used by human operators.</summary>


<h2 id="📄how-to-cite">📄 How to Cite</h2>

```bibtex
@inproceedings{
chen2025aiopslab,
title={{AIO}psLab: A Holistic Framework to Evaluate {AI} Agents for Enabling Autonomous Clouds},
author={Yinfang Chen and Manish Shetty and Gagan Somashekar and Minghua Ma and Yogesh Simmhan and Jonathan Mace and Chetan Bansal and Rujia Wang and Saravan Rajmohan},
booktitle={Eighth Conference on Machine Learning and Systems},
year={2025},
url={https://openreview.net/forum?id=3EXBLwGxtq}
}
@inproceedings{shetty2024building,
  title = {Building AI Agents for Autonomous Clouds: Challenges and Design Principles},
  author = {Shetty, Manish and Chen, Yinfang and Somashekar, Gagan and Ma, Minghua and Simmhan, Yogesh and Zhang, Xuchao and Mace, Jonathan and Vandevoorde, Dax and Las-Casas, Pedro and Gupta, Shachee Mishra and Nath, Suman and Bansal, Chetan and Rajmohan, Saravan},
  year = {2024},
  booktitle = {Proceedings of 15th ACM Symposium on Cloud Computing},
}
```

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.


## License

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT](LICENSE.txt) license.


### Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos is subject to those third-party’s policies.
