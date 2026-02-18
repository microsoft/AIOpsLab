# Setting up AIOpsLab locally using kind
## Prerequisites
- Python >= 3.11
- [Docker](https://docs.docker.com/get-started/get-docker/)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- [Helm](https://helm.sh/)
- [kind](https://kind.sigs.k8s.io/docs/user/quick-start/)
- [Poetry](https://python-poetry.org/docs/)
- [k9s](https://k9scli.io/) - Not necessary but recommended for monitoring the cluster

## Setup
Clone the repository:
```bash
git clone --recurse-submodules https://github.com/microsoft/AIOpsLab
cd AIOpsLab
```
Install dependencies:
```bash
cd AIOpsLab
poetry env use python3.11
export PATH="$HOME/.local/bin:$PATH" # export poetry to PATH if needed
poetry install # -vvv for verbose output
eval $(poetry env activate)
```

Create `config.yml`
```bash
cd aiopslab
cp config.yml.example config.yml
```
Update `config.yml` so that `k8s_host` is `kind`. It should look like this:
```yaml
# Kubernetes control node
k8s_host: kind
k8s_user: your_username # Username doesn't matter when using kind

# ssh key path
ssh_key_path: ~/.ssh/id_rsa # Replace with your ssh key path

# Directory where data files are stored
data_dir: data

# Flag to enable/disable qualitative evaluation (makes LLM calls)
qualitative_eval: false

# Flag to enable/disable printing the session
print_session: false
```
Go back to the root directory:
```bash
cd ..
```

Setup kind cluster:
```bash
# For x86 machines
kind create cluster --config kind/kind-config-x86.yaml

# For ARM machines
kind create cluster --config kind/kind-config-arm.yaml
```

## Running AIOpsLab
Run the "human agent" interface:
```bash
python cli.py
```
You should see output like this:
```bash
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                                  AIOpsLab                                                                                   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                                                                                                                                                               
 • Type your commands or actions below.                                                                                                                                        
 • Use exit to quit the application.                                                                                                                                           
 • Use start <problem_id> to begin a new problem.                                                                                                                              

aiopslab> 
```

Start a problem in the CLI:
```bash
start misconfig_app_hotel_res-detection-1
```

You should see the following prompt:
```bash
You are an expert DevOps engineer who has been tasked with detecting anomalies in a deployed service.                                                                          

The service you are working with today is described below: Service Name: Hotel Reservation Namespace: test-hotel-reservation Description: A hotel reservation application built
with Go and gRPC, providing backend in-memory and persistent databases, a recommender system for hotel recommendations, and a functionality to place reservations. Supported   
Operations:                                                                                                                                                                    

 • Get profile and rates of nearby hotels available during given time periods                                                                                                  
 • Recommend hotels based on user provided metrics                                                                                                                             
 • Place reservations                                                                                                                                                          

You will begin by analyzing the service's state and telemetry, and then submit your solution:                                                                                  

 1 str: Yes or No to indicate whether there are anomalies detected                                                                                                             

You are provided with the following APIs to interact with the service:                                                                                                         

get_logs Collects relevant log data from a pod using Kubectl.                                                                                                                  

                                                                                                                                                                               
     Args:                                                                                                                                                                     
         namespace (str): The namespace in which the service is running.                                                                                                       
         service (str): The name of the service.                                                                                                                               
                                                                                                                                                                               
     Returns:                                                                                                                                                                  
         str | dict | list[dicts]: Log data as a structured object or a string.                                                                                                
                                                                                                                                                                               

get_metrics Collects metrics data from the service using Prometheus.                                                                                                           

                                                                                                                                                                               
     Args:                                                                                                                                                                     
         namespace (str): The namespace in which the service is running.                                                                                                       
         duration (int): The number of minutes from now to start collecting metrics until now.                                                                                 
                                                                                                                                                                               
     Returns:                                                                                                                                                                  
         str: Path to the directory where metrics are saved.                                                                                                                   
                                                                                                                                                                               

get_traces Collects trace data from the service using Jaeger.                                                                                                                  

                                                                                                                                                                               
     Args:                                                                                                                                                                     
         namespace (str): The namespace in which the service is running.                                                                                                       
         duration (int): The number of minutes from now to start collecting traces until now.                                                                                  
                                                                                                                                                                               
     Returns:                                                                                                                                                                  
         str: Path to the directory where traces are saved.                                                                                                                    
                                                                                                                                                                               

read_metrics Reads and returns metrics from a specified CSV file.                                                                                                              

                                                                                                                                                                               
     Args:                                                                                                                                                                     
         file_path (str): Path to the metrics file (CSV format).                                                                                                               
                                                                                                                                                                               
     Returns:                                                                                                                                                                  
         str: The requested metrics or an error message.                                                                                                                       
                                                                                                                                                                               

read_traces Reads and returns traces from a specified CSV file.                                                                                                                

                                                                                                                                                                               
     Args:                                                                                                                                                                     
         file_path (str): Path to the traces file (CSV format).                                                                                                                
                                                                                                                                                                               
     Returns:                                                                                                                                                                  
         str: The requested traces or an error message.                                                                                                                        
                                                                                                                                                                               

You are also provided an API to a secure terminal to the service where you can run commands:                                                                                   

exec_shell Execute any shell command in a predefined debugging environment. Note: this is NOT A STATEFUL OR INTERACTIVE shell session. So you cannot execute commands like     
"kubectl edit".                                                                                                                                                                

                                                                                                                                                                               
     Args:                                                                                                                                                                     
         command (str): The command to execute.                                                                                                                                
                                                                                                                                                                               
     Returns:                                                                                                                                                                  
         str: The output of the command.                                                                                                                                       
                                                                                                                                                                               

Finally, you will submit your solution for this task using the following API:                                                                                                  

submit Submit if anomalies are detected to the orchestrator for evaluation.                                                                                                    

                                                                                                                                                                               
     Args:                                                                                                                                                                     
         has_anomaly (str): Yes if anomalies are detected, No otherwise.                                                                                                       
                                                                                                                                                                               
     Returns:                                                                                                                                                                  
         SubmissionStatus: The status of the submission.                                                                                                                       
                                                                                                                                                                               

At each turn think step-by-step and respond with your action.                                                                                                                  
╭──────────────────────────────────────────────────────────────────────────────── Environment ────────────────────────────────────────────────────────────────────────────────╮
│ Please take the next action                                                                                                                                                 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

aiopslab> 
```

### Cleaning up
You can destroy the cluster with the following command:
```bash
kind delete cluster
```
