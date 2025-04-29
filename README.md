# GNC-to-the-Moon-RL-Training



Project for Cloud Computing Class Spring 2025, GSU.




The Soft Actor–Critic (SAC) training loop in Stable‑Baselines3 begins by instantiating a stochastic actor πₜheta and two Q‑value critics Q_{φ₁}, Q_{φ₂}—each paired with a Polyak‑averaged target network φ_{targ,1}, φ_{targ,2} via a soft update coefficient τ—before any interactions with the environment :contentReference[oaicite:0]{index=0}.  
The agent then collects transitions (sₜ, aₜ, rₜ, sₜ₊₁, dₜ) by sampling actions aₜ ∼ πₜheta(·|sₜ) and storing them in a replay buffer for off‑policy learning :contentReference[oaicite:1]{index=1}.  
During each gradient step, the critics are updated by minimizing the soft Bellman residual  
$$
L(φᵢ)=\mathbb{E}_{(s,a,r,s',d)\sim D}\bigl[\,Q_{φᵢ}(s,a)-y(r,s',d)\bigr]^2,
\quad
y=r+\gamma(1-d)\Bigl(\min_{j=1,2}Q_{φ_{targ,j}}(s',\tilde a')-\alpha\logπ_{θ}(\tilde a'|s')\Bigr),
\;\tilde a'∼π_{θ}(\cdot|s'),
$$  
which injects an entropy regularizer α to stabilize learning :contentReference[oaicite:2]{index=2}.  
Next, the policy is updated using the reparameterization trick to minimize  
$$
L(θ)=\mathbb{E}_{s,ξ}\bigl[\,α\logπ_{θ}(\tilde a_{θ}(s,ξ)|s)-\min_{j}Q_{φ_{j}}(s,\tilde a_{θ}(s,ξ))\bigr],
$$  
thereby maximizing expected return plus entropy :contentReference[oaicite:3]{index=3}.  
Optionally, α itself is auto‑tuned by minimizing  
$$
L(α)=\mathbb{E}_{a∼π_{θ}}\bigl[-α\bigl(\logπ_{θ}(a|s)+\bar H\bigr)\bigr],
$$  
to match a target entropy :contentReference[oaicite:4]{index=4}. Finally, after each update the target networks undergo a Polyak update  
$$
φ_{targ,i}\leftarrow τ\,φ_{i} + (1-τ)\,φ_{targ,i},
$$  
before the loop repeats until convergence :contentReference[oaicite:5]{index=5}.


# Azure RL Training Deployment Strategy:

This project provides a strategy and code for deploying distributed Reinforcement Learning (RL) training jobs on multiple Azure Virtual Machines (VMs) using Azure Blob Storage for code distribution and result collection.

The example demonstrates training the Soft Actor-Critic (SAC) agent from the `stable-baselines3` library on the LunarLander environment (`gymnasium`).

## Features

* **Multi-VM Deployment:** Deploy a cluster of VMs using an Azure Resource Manager (ARM) template.
* **Centralized Code Distribution:** Upload project code to Azure Blob Storage. VMs download the code automatically during setup.
* **Automated VM Setup:** A custom shell script configures the environment, installs dependencies, and initiates training on each VM.
* **Parameterization:** Training parameters (timesteps, learning rate, buffer size, seed) are configured via a JSON file and read by the setup script on each VM.
* **Centralized Results Collection:** Training logs, models, and evaluation results are uploaded back to Azure Blob Storage.
* **Managed Identity Authentication:** Uses Azure Managed Identities for secure access to Blob Storage from VMs without managing secrets.

## Prerequisites

* An Azure account and subscription.
* [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli) installed and logged in (`az login`).
* Python 3.x installed.
* Python `venv` module (usually included with Python 3.3+).
* Permissions to create resources in your Azure subscription (Resource Group, VMs, Storage Account, RBAC Assignments).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-project-directory>
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Python dependencies:**
    Ensure you have a `requirements.txt` file with the necessary packages (e.g., `gymnasium`, `stable-baselines3`, `azure-storage-blob`, `azure-identity`, `numpy`, `jq`).
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Azure Storage Account environment variable:**
    Replace `<your-storage-account-name>` with the name of the Azure Storage Account you will use (this name is also a parameter in the ARM template).
    ```bash
    export AZURE_STORAGE_ACCOUNT=<your-storage-account-name>
    ```

5.  **Configure VM training parameters:**
    Generate the `azure_vm_config.json` file. This file defines the number of VMs and specific training hyperparameters (including a unique seed) for each.
    ```bash
    python prepare_deployment.py --create-config --num-vms <number-of-vms> --vm-size <azure-vm-size>
    # Example: python prepare_deployment.py --create-config --num-vms 5 --vm-size Standard_DS3_v2
    ```
    Review and edit `azure_vm_config.json` if needed.

6.  **Upload code and configuration to Azure Blob Storage:**
    This uploads all necessary files (`*.py`, `requirements.txt`, `setup_vm.sh`, `azure_vm_config.json`, etc. - excluding specified directories) to the designated storage container (`rl-training-files` by default).
    ```bash
    python prepare_deployment.py --upload --container rl-training-files
    ```

## Deployment

1.  **Deploy the Azure infrastructure:**
    Use the Azure CLI to deploy the ARM template (`azure_deployment.json`). You will be prompted for parameters like admin username, password/key, VM count, and the storage account name.
    ```bash
    az group create --name <your-resource-group-name> --location <azure-region>
    az deployment group create \
      --resource-group <your-resource-group-name> \
      --template-file azure_deployment.json \
      --parameters \
        vmName="rl-training-vm" \
        vmCount=<number-of-vms-matching-config> \
        adminUsername="<admin-username>" \
        adminPasswordOrKey="<admin-password-or-ssh-public-key>" \
        authenticationType="<password-or-sshPublicKey>" \
        vmSize="<azure-vm-size-matching-config>" \
        storageAccountName="<your-storage-account-name-matching-env-var>" \
        containerName="rl-training-files" \
        resultContainerName="rl-training-results"
    ```

2.  **Assign RBAC Role (CRITICAL):**
    The ARM template enables a System-Assigned Managed Identity on each VM, but you **must** grant this identity permissions to access your storage account. This step must be done *after* the VMs are created. Grant the **"Storage Blob Data Contributor"** role to each VM's Managed Identity on the storage account specified by `storageAccountName`.

    You can do this via the Azure portal, or using Azure CLI (requires finding the Principal ID of each VM's identity):

    ```bash
    # Example for a single VM (repeat for each VM or script this)
    VM_NAME="rl-training-vm-1" # Replace with actual VM name
    RESOURCE_GROUP="<your-resource-group-name>"
    STORAGE_ACCOUNT_ID=$(az storage account show -n <your-storage-account-name> -g $RESOURCE_GROUP --query id -o tsv)
    VM_IDENTITY_PRINCIPAL_ID=$(az vm show -g $RESOURCE_GROUP -n $VM_NAME --query identity.principalId -o tsv)

    az role assignment create \
      --assignee $VM_IDENTITY_PRINCIPAL_ID \
      --role "Storage Blob Data Contributor" \
      --scope $STORAGE_ACCOUNT_ID
    ```
    Repeat or script this for all VMs.

## Execution and Monitoring

* Upon successful deployment and RBAC assignment, the Custom Script Extension on each VM will automatically execute `setup_vm.sh`.
* The shell script downloads the code, installs dependencies, reads its specific training parameters from `azure_vm_config.json`, and starts the `cloud_train.py` script.
* Training output (verbose=1) will be visible via the VM's Boot Diagnostics or by SSHing into the VM and checking the console output or redirected logs.
* Training artifacts (checkpoints, final model, evaluation results) are saved locally in `$HOME/rl-project/models/` and `$HOME/rl-project/logs/` on each VM.
* The `cloud_train.py` script uploads the `evaluation_results.json` and the final/best models to the `rl-training-results` blob container, organized by a unique run ID (e.g., `vm1_seed1000_<timestamp>/...`).

## Project Structure

* `prepare_deployment.py`: Script for local setup (creating config, uploading code).
* `cloud_train.py`: The main Python script executed on Azure VMs for training.
* `setup_vm.sh`: Shell script executed by the ARM template's Custom Script Extension to set up the VM and run training.
* `azure_deployment.json`: Azure Resource Manager (ARM) template for infrastructure deployment.
* `requirements.txt`: Lists Python dependencies required for the training script.
* `azure_vm_config.json`: JSON file generated by `prepare_deployment.py` defining VM-specific training parameters (downloaded and used by `setup_vm.sh`).
* `local_train.py` (Optional): Script for local testing/training.
* `infer_model.py` (Optional): Script for loading a trained model and running inference.

## Sensitivity to Changes

The deployment is sensitive to the following changes in your training code (`cloud_train.py`) and setup scripts:

* **CLI Argument Names:** If you change the names of the `--vm-id`, `--seed`, `--timesteps`, `--lr`, `--buffer-size`, or `--container` arguments in `cloud_train.py`, you **must** update the `python cloud_train.py` command in `setup_vm.sh` to match.
* **Required Python Packages:** If `cloud_train.py` requires new libraries, you **must** update `requirements.txt` locally and re-upload your code using `prepare_deployment.py --upload`.
* **Local Artifact Paths:** If `cloud_train.py` saves logs, models, or results to different local directories or filenames, you may need to update:
    * The directory creation commands (`mkdir`) in `setup_vm.sh`.
    * The directory exclusion logic in `prepare_deployment.py`.
    * The local file paths referenced in the `upload_to_blob` calls within `cloud_train.py`.
* **Blob Paths/Container Names for Results:** If you change the target container or the blob path structure for uploaded results in `cloud_train.py`, ensure the `resultContainerName` parameter in the ARM template and the `$RESULT_CONTAINER_NAME` variable in `setup_vm.sh` (and the `--container` argument passed to `cloud_train.py`) are consistent.
* **Azure Authentication Logic:** Any changes to how `cloud_train.py` or `setup_vm.sh` authenticate with Azure Blob Storage (e.g., moving away from `DefaultAzureCredential` and Managed Identity) will require corresponding changes in the ARM template and potentially shell script login commands.

Changes to the core RL algorithm or environment interaction logic within `cloud_train.py` (provided dependencies and file interactions remain consistent) do not typically require changes to the deployment scripts themselves.

## Future Improvements

* Automate the RBAC role assignment within the ARM template or via a post-deployment script/Azure Policy.
* Implement automatic VM shutdown or deallocation after training completes to save costs.
* Centralize log aggregation (e.g., using Azure Log Analytics or transferring logs separately).
* Explore Azure Machine Learning or Azure Batch for more managed and scalable training orchestration.

## License

[Specify your project license here, e.g., MIT, Apache 2.0]
