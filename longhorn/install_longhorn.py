import subprocess
import platform
import os
import yaml

def install_helm():
    helm_version = "v3.11.2"
    system = platform.system().lower()
    helm_url = f"https://get.helm.sh/helm-{helm_version}-linux-amd64.tar.gz"
    print(f"Downloading Helm from {helm_url}...")
    subprocess.run(["curl", "-LO", helm_url], check=True)
    print("Installing Helm...")
    try:
        subprocess.run(["tar", "-zxvf", f"helm-{helm_version}-{system}-amd64.tar.gz"], check=True)
        helm_binary = f"{system}-amd64/helm"
        subprocess.run(["sudo", "mv", helm_binary, "/usr/local/bin/helm"], check=True)
        subprocess.run(["sudo", "chmod", "+x", "/usr/local/bin/helm"], check=True)
        os.remove(f"helm-{helm_version}-{system}-amd64.tar.gz")
        os.rmdir(f"{system}-amd64")
    except Exception as e:
        print(f"Error when installing helm {e}")

    return 0

def configure_longhorn_repo():
    print("Adding Longhorn Helm repository...")
    subprocess.run(["helm", "repo", "add", "longhorn", "https://charts.longhorn.io"], check=True)

    print("Updating Helm repositories...")
    subprocess.run(["helm", "repo", "update"], check=True)

    print("Searching for Longhorn chart in Helm repositories...")
    subprocess.run(["helm", "search", "repo", "longhorn"], check=True)

    print("Saving Longhorn default values to /tmp/longhorn-values.yaml...")
    subprocess.run(["helm show values longhorn/longhorn > /tmp/longhorn-values.yaml"], shell=True)

    return 0

def modify_longhorn_values():
    file_path = "/tmp/longhorn-values.yaml"

    # Load the current YAML data
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    # Modify the YAML data
    if 'service' in data and 'ui' in data['service']:
        data['service']['ui']['type'] = 'NodePort'
        data['service']['ui']['nodePort'] = 30001
    else:
        print("The structure of the YAML file is not as expected.")
        return

    # Write the modified YAML back to the file
    with open(file_path, 'w') as file:
        yaml.dump(data, file)

    print(f"Modified {file_path} with the specified values.")

def set_permissions():
    print("Setting permissions for /etc/rancher/k3s/k3s.yaml to 644...")
    try:
        subprocess.run(["mkdir", "-p", "/root/.kube"], check=True)
        subprocess.run(["cp", "/etc/rancher/k3s/k3s.yaml", "/root/.kube/config"], check=True)
        subprocess.run(["sudo", "chmod", "644", "/etc/rancher/k3s/k3s.yaml"], check=True)
        print("Permissions set successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error setting permissions: {e}")

def execute_additional_commands():
    print("Executing additional Helm command to install/upgrade Longhorn...")
    try:
        subprocess.run([
            "helm", "install", "longhorn", "longhorn/longhorn",
            "--namespace", "longhorn-system", "--create-namespace",
            "-f", "/tmp/longhorn-values.yaml"
        ], check=True)
        print("Longhorn installed/upgraded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")


if __name__ == "__main__":
    install_helm()
    configure_longhorn_repo()
    modify_longhorn_values()
    set_permissions()
    execute_additional_commands()
