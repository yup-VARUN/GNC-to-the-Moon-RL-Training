import os
import argparse
import json
import subprocess # Added for running Azure CLI commands
import time       # Added for delays/retries
import sys        # Added for system exit
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

# Helper function to run shell commands
def run_command(command, shell=True, check=True, **kwargs):
    """Runs a shell command and checks for errors."""
    print(f"\n--- Running command: {' '.join(command) if isinstance(command, list) else command}")
    try:
        # Setting env=os.environ ensures the AZURE_STORAGE_ACCOUNT env var is passed
        result = subprocess.run(command, shell=shell, check=check, capture_output=True, text=True, env=os.environ, **kwargs)
        print("Stdout:")
        print(result.stdout)
        if result.stderr:
            print("Stderr:")
            print(result.stderr)
        return result
    except subprocess.CalledProcessError as e:
        print(f"\n--- Command failed with error code {e.returncode} ---")
        print("Stderr:")
        print(e.stderr)
        print("Stdout:")
        print(e.stdout)
        raise

# Function to upload code to Azure Blob Storage (kept from your original script)
def upload_code_to_blob(container_name, local_directory="."):
    """Upload the entire local directory to Azure Blob Storage"""
    print(f"\n--- Uploading code from '{local_directory}' to blob container '{container_name}' ---")
    try:
        # Connect to Azure using default credentials (requires local auth setup)
        credential = DefaultAzureCredential()

        # Get the storage account name from environment variables
        account_name = os.environ.get("AZURE_STORAGE_ACCOUNT")
        if not account_name:
            raise ValueError("AZURE_STORAGE_ACCOUNT environment variable not set")

        # Create the BlobServiceClient
        blob_service_client = BlobServiceClient(
            account_url=f"https://{account_name}.blob.core.windows.net",
            credential=credential
        )

        # Create the container if it doesn't exist
        container_client = blob_service_client.get_container_client(container_name)
        try:
            container_client.create_container()
            print(f"Container '{container_name}' created.")
        except Exception: # Bare except is generally bad, but keeping original logic
            print(f"Container '{container_name}' already exists.")

        # Upload all files recursively
        print("Starting file upload...")
        for root, dirs, files in os.walk(local_directory):
            # Modify dirs in place to skip them in os.walk recursion
            if 'venv' in dirs:
                dirs.remove('venv')
            if '__pycache__' in dirs:
                 dirs.remove('__pycache__')
            if 'logs' in dirs:
                 dirs.remove('logs')
            if 'models' in dirs:
                 dirs.remove('models')


            for file in files:
                # Skip hidden files based on name
                if file.startswith('.'):
                    continue

                local_file_path = os.path.join(root, file)
                # Create blob path removing the initial './' if present
                relative_path = os.path.relpath(local_file_path, local_directory)
                blob_path = relative_path.replace('\\', '/') # Use forward slashes for blob paths

                # Upload file
                blob_client = container_client.get_blob_client(blob_path)
                try:
                    with open(local_file_path, "rb") as data:
                        blob_client.upload_blob(data, overwrite=True)
                    print(f"Uploaded {local_file_path} to {blob_path}")
                except Exception as upload_e:
                    print(f"Error uploading {local_file_path}: {upload_e}")
                    # Decide if you want to fail hard or continue on file upload errors

        print("All files processed for upload.")

    except Exception as e:
        print(f"Error during file upload process: {str(e)}")
        raise # Re-raise the exception to indicate failure

# Function to create VM configuration (kept from your original script)
def create_vm_deployment_config(config_file, num_vms, vm_size="Standard_DS3_v2"):
    """Create a configuration file for VM deployment"""
    print(f"\n--- Creating VM deployment configuration file '{config_file}' ---")
    config = {
        "vms": [],
        "storage": {
            "account_name": os.environ.get("AZURE_STORAGE_ACCOUNT"),
            "container_name": "rl-training-files" # Matches default upload container
        }
    }

    # Create VM configurations
    # Aligning seed calculation with 0-indexed array in config file
    for i in range(num_vms):
        config["vms"].append({
            "name": f"rl-training-vm-{i+1}", # Name format matches ARM template
            "size": vm_size,
            "training": {
                "total_timesteps": 500000, # Default training parameter
                "learning_rate": 0.0003,   # Default training parameter
                "buffer_size": 1000000,    # Default training parameter
                "seed": 1000 + i,          # Different seed for each VM (0-indexed)
            }
        })

    # Write the configuration to a file
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created VM deployment configuration at {config_file}.")
    except Exception as e:
        print(f"Error writing config file: {e}")
        raise

# --- New functionality: Orchestrate Deployment and RBAC ---

def deploy_infrastructure(resource_group, location, storage_account_name, code_container, results_container, arm_params):
    """Deploys Azure infrastructure using the ARM template."""
    print(f"\n--- Deploying Azure infrastructure to resource group '{resource_group}' ---")

    # Create resource group if it doesn't exist
    print(f"Ensuring resource group '{resource_group}' exists in '{location}'...")
    run_command(f"az group create --name {resource_group} --location {location}")

    # Start ARM deployment
    print("Starting ARM deployment...")
    # Ensure necessary parameters are in arm_params for the template
    full_arm_params = {
        "storageAccountName": storage_account_name,
        "containerName": code_container,
        "resultContainerName": results_container,
        **arm_params # Include other VM-specific params from CLI args
    }

    # Convert dict to JSON string for az cli
    arm_params_json = json.dumps(full_arm_params)

    try:
        # Use --wait to ensure deployment finishes before proceeding to RBAC
        deployment_command = [
            "az", "deployment", "group", "create",
            "--resource-group", resource_group,
            "--template-file", "azure_deployment.json", # Ensure this file is in the project directory
            "--parameters", arm_params_json,
            "--query", "properties.outputs" # Capture outputs like VM hostnames
        ]
        run_command(deployment_command, timeout=3600) # Add timeout for deployment

        print("\n--- ARM deployment completed successfully ---")

    except subprocess.TimeoutExpired:
        print("\n--- ARM deployment timed out ---")
        print("Deployment may still be running in Azure. Cannot proceed with automated RBAC.")
        print("You must manually assign the 'Storage Blob Data Contributor' role to the VMs' Managed Identities.")
        sys.exit(1)
    except Exception as e:
        print(f"\n--- Error during ARM deployment: {e} ---")
        print("Please check your ARM template, parameters, and Azure subscription permissions.")
        sys.exit(1)

def assign_rbac_to_vms(resource_group, storage_account_name, vm_base_name, vm_count):
    """Assigns the Storage Blob Data Contributor role to each VM's Managed Identity."""
    print("\n--- Assigning 'Storage Blob Data Contributor' role to VM Managed Identities ---")

    try:
        # Get Storage Account ID
        print(f"Getting ID for storage account '{storage_account_name}'...")
        storage_account_id_cmd = ["az", "storage", "account", "show", "--name", storage_account_name, "--resource-group", resource_group, "--query", "id", "-o", "tsv"]
        storage_account_id = run_command(storage_account_id_cmd).stdout.strip()
        if not storage_account_id:
             raise Exception(f"Could not retrieve Storage Account ID for '{storage_account_name}'.")
        print(f"Storage Account ID: {storage_account_id}")


        # Assign role to each VM
        for i in range(vm_count):
            vm_name = f"{vm_base_name}-{i+1}" # VM names are 1-indexed in ARM template and setup script

            # Get VM's Managed Identity Principal ID
            print(f"\nGetting Principal ID for VM '{vm_name}' in '{resource_group}'...")
            principal_id = None
            retries = 15 # Retry getting Principal ID as it might take a moment after VM creation
            delay = 20 # Seconds delay between retries
            for attempt in range(retries):
                try:
                    principal_id_cmd = ["az", "vm", "show", "--resource-group", resource_group, "--name", vm_name, "--query", "identity.principalId", "-o", "tsv"]
                    principal_id_result = run_command(principal_id_cmd, check=False) # Don't hard fail on initial attempts
                    principal_id = principal_id_result.stdout.strip()

                    if principal_id and principal_id != 'null':
                         print(f"Principal ID found: {principal_id}")
                         break # Exit retry loop if successful
                    else:
                         print(f"Principal ID not yet available for '{vm_name}'. Attempt {attempt + 1}/{retries}. Retrying in {delay} seconds...")
                         time.sleep(delay)
                except Exception as e:
                     print(f"Error during attempt {attempt + 1}/{retries} to get principal ID for '{vm_name}': {e}")
                     time.sleep(delay)

            if not principal_id or principal_id == 'null':
                 print(f"\n--- Failed to get Principal ID for VM '{vm_name}' after multiple retries. RBAC assignment skipped for this VM. ---")
                 print("Manual RBAC assignment is required for this VM for training to succeed.")
                 continue # Skip RBAC assignment for this VM, continue with others

            # Assign the role
            print(f"Assigning 'Storage Blob Data Contributor' role to '{vm_name}' ({principal_id}) on scope '{storage_account_id}'...")
            assign_role_cmd = [
                "az", "role", "assignment", "create",
                "--assignee", principal_id,
                "--role", "Storage Blob Data Contributor",
                "--scope", storage_account_id
            ]
            run_command(assign_role_cmd) # This should succeed if principal_id was found
            print(f"Role assigned successfully to '{vm_name}'.")

    except Exception as e:
        print(f"\n--- An error occurred during the RBAC assignment phase: {e} ---")
        print("Training on VMs might fail to access storage. Consider manual RBAC assignment.")
        # Do not exit here, let the script finish with a warning

# --- Main execution block ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy RL training setup to Azure")
    parser.add_argument("--resource-group", type=str, required=True, help="Azure Resource Group name")
    parser.add_argument("--location", type=str, default="eastus", help="Azure region for deployment")
    parser.add_argument("--storage-account", type=str, required=True, help="Azure Storage Account name (must be unique)")
    parser.add_argument("--vm-count", type=int, default=3, help="Number of VMs to deploy")
    parser.add_argument("--vm-size", type=str, default="Standard_DS3_v2", help="Azure VM size")
    parser.add_argument("--admin-username", type=str, required=True, help="Admin username for VMs")
    parser.add_argument("--admin-password-or-key", type=str, required=True, help="Admin password or SSH public key for VMs")
    parser.add_argument("--auth-type", type=str, choices=["password", "sshPublicKey"], default="sshPublicKey", help="Authentication type for VMs")
    parser.add_argument("--code-container", type=str, default="rl-training-files", help="Blob container for code upload")
    parser.add_argument("--results-container", type=str, default="rl-training-results", help="Blob container for results upload")

    # Optional flags to run only specific parts
    parser.add_argument("--skip-config", action="store_true", help="Skip generating the VM configuration file")
    parser.add_argument("--skip-upload", action="store_true", help="Skip uploading code and configuration to blob storage")
    parser.add_argument("--skip-deploy", action="store_true", help="Skip ARM deployment")
    parser.add_argument("--skip-rbac", action="store_true", help="Skip RBAC role assignment")


    args = parser.parse_args()

    # Set the storage account env var early
    os.environ["AZURE_STORAGE_ACCOUNT"] = args.storage_account
    storage_account_name = args.storage_account
    vm_base_name = "rl-training-vm" # Must match the vmName parameter in the ARM template default

    print("\n--- Starting RL Training Deployment Orchestration ---")

    # Step 1: Generate VM configuration
    config_file = "azure_vm_config.json"
    if not args.skip_config:
        try:
            create_vm_deployment_config(config_file, args.vm_count, args.vm_size)
        except Exception:
            print("Deployment aborted due to config generation failure.")
            sys.exit(1)
    else:
        print("\n--- Skipping VM configuration generation ---")
        if not os.path.exists(config_file):
             print(f"Warning: Skipping config generation, but '{config_file}' does not exist. This will cause issues later if needed.")

    # Step 2: Upload code and configuration
    # The config file generated in step 1 is included here
    if not args.skip_upload:
        try:
            upload_code_to_blob(args.code_container)
        except Exception:
            print("Deployment aborted due to code upload failure.")
            sys.exit(1)
    else:
        print("\n--- Skipping code and configuration upload ---")
        print("Ensure code and config are already in the blob container for deployment to work.")


    # Step 3: Deploy Azure Infrastructure
    if not args.skip_deploy:
        # Parameters specifically for the ARM template that come from CLI args
        arm_specific_params = {
            "vmName": vm_base_name,
            "vmCount": args.vm_count,
            "adminUsername": args.admin_username,
            "adminPasswordOrKey": args.admin_password_or_key,
            "authenticationType": args.auth_type,
            "vmSize": args.vm_size,
            # storage account and container names are passed separately to deploy_infrastructure
        }
        try:
            deploy_infrastructure(
                args.resource_group,
                args.location,
                storage_account_name,
                args.code_container,
                args.results_container,
                arm_specific_params # Pass the params specific to ARM
            )
        except Exception:
             print("Deployment aborted due to infrastructure deployment failure.")
             sys.exit(1)
    else:
         print("\n--- Skipping ARM infrastructure deployment ---")
         print("Ensure VMs, network, storage, and containers are already deployed correctly.")


    # Step 4: Assign RBAC role
    if not args.skip_rbac:
         # This step implicitly depends on Step 3 completing successfully (VMs exist)
        if args.skip_deploy:
            print("\n--- Attempting RBAC assignment on pre-existing VMs ---")
            print("Note: This assumes the VMs exist and have System-Assigned Managed Identities enabled.")

        try:
            assign_rbac_to_vms(args.resource_group, storage_account_name, vm_base_name, args.vm_count)
            print("\n--- RBAC assignment phase completed ---")
        except Exception:
             # assign_rbac_to_vms already prints errors, no need to re-print generic failure
             print("\n--- RBAC assignment encountered an error ---")
             # Do not exit, continue to final message
    else:
        print("\n--- Skipping RBAC role assignment ---")
        print("Manual RBAC assignment is required for training to access storage.")


    print("\n--- RL Training Deployment Orchestration Complete ---")
    print(f"Training should now be starting (or will start shortly) on the VMs in resource group '{args.resource_group}'.")
    print(f"Results will be uploaded to the '{args.results_container}' container in storage account '{storage_account_name}'.")
    print("You can monitor progress by checking VM Boot Diagnostics in the Azure portal or SSHing into the VMs.")
