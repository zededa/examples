import subprocess
import platform
import os
import yaml
import re
import argparse

def get_kubernetes_master_ip():
    # Run `kubectl get nodes -o wide` to get node details
    result = subprocess.run(["kubectl", "get", "nodes", "-o", "wide"], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error executing kubectl command: {result.stderr}")
        return None

    # Extract node IPs from the output
    lines = result.stdout.strip().split('\n')
    header = lines[0].split()
    ip_index = header.index("INTERNAL-IP")
    role_index = header.index("ROLES")

    for line in lines[1:]:
        columns = line.split()
        node_ip = columns[ip_index]
        node_role = columns[role_index]

        # Check if the role includes "master" or "control-plane"
        if "master" in node_role or "control-plane" in node_role:
            return node_ip

    print("No master node found.")
    return None

def get_local_ips():
    # Run `ip addr` to get local network interface details
    result = subprocess.run(["ip", "addr"], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error executing ip addr command: {result.stderr}")
        return []

    # Extract IP addresses using regex
    ip_pattern = re.compile(r'inet (\d+\.\d+\.\d+\.\d+)/')
    local_ips = ip_pattern.findall(result.stdout)

    return local_ips

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

def apply_rancher_yaml(url):
    yaml_file_path = "/tmp/rancher.yaml"

    # Download Rancher YAML using curl and save to /tmp/rancher.yaml
    print(f"Downloading Rancher YAML from {url}...")
    try:
        subprocess.run(
            ["curl", "--insecure", "-sfL", url, "-o", yaml_file_path],
            check=True
        )
        print(f"Rancher YAML downloaded to {yaml_file_path}.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading Rancher YAML: {e}")
        return

    # Apply the downloaded YAML file using kubectl
    print("Applying Rancher YAML using kubectl...")
    try:
        subprocess.run(
            ["kubectl", "apply", "-f", yaml_file_path],
            check=True
        )
        print("Rancher YAML applied successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error applying Rancher YAML: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Apply Rancher YAML to Kubernetes.")
    parser.add_argument("url", help="URL of the Rancher YAML file")
    args = parser.parse_args()

    master_ip = get_kubernetes_master_ip()
    local_ips = get_local_ips()
    # Check if the master IP is among the local interfaces
    if master_ip in local_ips:
        install_helm()
        configure_longhorn_repo()
        modify_longhorn_values()
        set_permissions()
        execute_additional_commands()
        apply_rancher_yaml(args.url)
