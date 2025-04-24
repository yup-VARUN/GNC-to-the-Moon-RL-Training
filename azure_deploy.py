import os
import argparse
import json
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

# deploying training to Azure VMs
def upload_code_to_blob(container_name, local_directory="."):
    """Upload the entire local directory to Azure Blob Storage"""
    try:
        # Connect to Azure using default credentials
        credential = DefaultAzureCredential()
        
        # Get the connection string from environment variables
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
            print(f"Container '{container_name}' created")
        except:
            print(f"Container '{container_name}' already exists")
        
        # Upload all files recursively
        for root, dirs, files in os.walk(local_directory):
            for file in files:
                # Skip hidden files, virtual environment, and logs/models directories
                if (file.startswith('.') or 
                   'venv' in root or 
                   '__pycache__' in root or
                   'logs' in root or
                   'models' in root):
                    continue
                    
                local_file_path = os.path.join(root, file)
                # Create blob path removing the initial './'
                relative_path = os.path.relpath(local_file_path, local_directory)
                blob_path = relative_path.replace('\\', '/')
                
                # Upload file
                blob_client = container_client.get_blob_client(blob_path)
                with open(local_file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                print(f"Uploaded {local_file_path} to {blob_path}")
                
        print("All files uploaded successfully")
        
    except Exception as e:
        print(f"Error uploading files: {str(e)}")

def create_vm_deployment_config(config_file, num_vms, vm_size="Standard_DS3_v2"):
    """Create a configuration file for VM deployment"""
    config = {
        "vms": [],
        "storage": {
            "account_name": os.environ.get("AZURE_STORAGE_ACCOUNT"),
            "container_name": "rl-training-files"
        }
    }
    
    # Create VM configurations
    for i in range(num_vms):
        config["vms"].append({
            "name": f"rl-training-vm-{i+1}",
            "size": vm_size,
            "training": {
                "total_timesteps": 500000,
                "learning_rate": 0.0003,
                "buffer_size": 1000000,
                "seed": 1000 + i,  # Different seed for each VM
            }
        })

    # Write the configuration to a file
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created VM deployment configuration at {config_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and deploy RL training to Azure")
    parser.add_argument("--container", type=str, default="rl-training-files",
                        help="Blob storage container name")
    parser.add_argument("--num-vms", type=int, default=3,
                        help="Number of VMs to prepare configuration for")
    parser.add_argument("--vm-size", type=str, default="Standard_DS3_v2",
                        help="Azure VM size")
    parser.add_argument("--create-config", action="store_true",
                        help="Create VM configuration file")
    parser.add_argument("--upload", action="store_true",
                        help="Upload files to Azure Blob Storage")
    
    args = parser.parse_args()
    
    if args.create_config:
        create_vm_deployment_config("azure_vm_config.json", args.num_vms, args.vm_size)
    
    if args.upload:
        upload_code_to_blob(args.container)