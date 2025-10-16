#!/usr/bin/env python3
"""
MinIO Model Sync Sidecar for OpenVINO Model Server

This sidecar continuously monitors a MinIO bucket for new models,
downloads them, and updates the OVMS config.json file to automatically
load new models without manual intervention.
"""

import os
import sys
import time
import json
import yaml
import logging
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

class ModelSyncError(Exception):
    """Custom exception for model sync operations."""
    pass

class MinIOClient:
    """MinIO client wrapper for model operations."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize MinIO S3 client."""
        try:
            self.client = boto3.client(
                's3',
                endpoint_url=self.config['minio']['endpoint'],
                aws_access_key_id=self.config['minio']['access_key'],
                aws_secret_access_key=self.config['minio']['secret_key'],
                verify=self.config['minio'].get('secure', False)
            )
            self.logger.info(f"Connected to MinIO at {self.config['minio']['endpoint']}")
        except Exception as e:
            raise ModelSyncError(f"Failed to connect to MinIO: {str(e)}")
    
    def list_objects(self, bucket: str, prefix: str = "") -> List[Dict[str, Any]]:
        """List objects in MinIO bucket."""
        try:
            response = self.client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix
            )
            return response.get('Contents', [])
        except ClientError as e:
            self.logger.error(f"Failed to list objects in bucket {bucket}: {str(e)}")
            return []
    
    def download_object(self, bucket: str, key: str, local_path: str) -> bool:
        """Download object from MinIO to local path."""
        try:
            # Ensure local directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            self.client.download_file(bucket, key, local_path)
            self.logger.info(f"Downloaded {key} to {local_path}")
            return True
        except ClientError as e:
            self.logger.error(f"Failed to download {key}: {str(e)}")
            return False
    
    def get_object_hash(self, bucket: str, key: str) -> Optional[str]:
        """Get ETag (hash) of object in MinIO."""
        try:
            response = self.client.head_object(Bucket=bucket, Key=key)
            return response.get('ETag', '').strip('"')
        except ClientError as e:
            self.logger.error(f"Failed to get hash for {key}: {str(e)}")
            return None

class OVMSConfigManager:
    """Manages OpenVINO Model Server configuration."""
    
    def __init__(self, config_path: str, logger: logging.Logger):
        self.config_path = config_path
        self.logger = logger
        self.models = {}
        self._load_config()
    
    def _load_config(self):
        """Load existing OVMS configuration."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Normalize models to the correct format (remove model_version_policy if present)
                    self.models = {}
                    for model in config.get('model_config_list', []):
                        model_name = model['config']['name']
                        normalized_model = {
                            "config": {
                                "name": model_name,
                                "base_path": model['config']['base_path']
                            }
                        }
                        self.models[model_name] = normalized_model
                self.logger.info(f"Loaded OVMS config with {len(self.models)} models")
            else:
                self.logger.info("No existing OVMS config found, starting fresh")
        except Exception as e:
            self.logger.error(f"Failed to load OVMS config: {str(e)}")
            self.models = {}
    
    def add_model(self, name: str, base_path: str, version_policy: str = "latest") -> bool:
        """Add or update a model in the configuration."""
        try:
            model_config = {
                "config": {
                    "name": name,
                    "base_path": base_path
                }
            }
            
            self.models[name] = model_config
            self.logger.info(f"Added/updated model {name} with base_path {base_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add model {name}: {str(e)}")
            return False
    
    def remove_model(self, name: str) -> bool:
        """Remove a model from the configuration."""
        try:
            if name in self.models:
                del self.models[name]
                self.logger.info(f"Removed model {name}")
                return True
            else:
                self.logger.warning(f"Model {name} not found in config")
                return False
        except Exception as e:
            self.logger.error(f"Failed to remove model {name}: {str(e)}")
            return False
    
    def save_config(self) -> bool:
        """Save the configuration to file."""
        try:
            config = {
                "model_config_list": list(self.models.values())
            }
            
            # Write to temporary file first, then move (atomic operation)
            temp_path = f"{self.config_path}.tmp"
            with open(temp_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            os.rename(temp_path, self.config_path)
            self.logger.info(f"Saved OVMS config with {len(self.models)} models")
            
            # Also save a models list file for the client to read
            self.save_models_list()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to save OVMS config: {str(e)}")
            return False
    
    def save_models_list(self) -> bool:
        """Save list of available models for client discovery."""
        try:
            models_list = {
                "models": list(self.models.keys()),
                "updated_at": datetime.now().isoformat()
            }
            
            models_path = os.path.join(os.path.dirname(self.config_path), "models.json")
            temp_path = f"{models_path}.tmp"
            
            with open(temp_path, 'w') as f:
                json.dump(models_list, f, indent=2)
            
            os.rename(temp_path, models_path)
            self.logger.info(f"Saved models list with {len(self.models)} models")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save models list: {str(e)}")
            return False

class ModelTracker:
    """Tracks model states and changes."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.model_hashes = {}  # model_path -> hash
        self.model_timestamps = {}  # model_path -> timestamp
    
    def is_model_changed(self, model_path: str, current_hash: str) -> bool:
        """Check if model has changed based on hash."""
        previous_hash = self.model_hashes.get(model_path)
        if previous_hash != current_hash:
            self.model_hashes[model_path] = current_hash
            self.model_timestamps[model_path] = datetime.now()
            return True
        return False
    
    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """Get model tracking information."""
        return {
            'hash': self.model_hashes.get(model_path),
            'timestamp': self.model_timestamps.get(model_path),
            'last_updated': self.model_timestamps.get(model_path, datetime.now()).isoformat()
        }

class ModelSyncSidecar:
    """Main sidecar application for syncing models from MinIO to OVMS."""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = setup_logging(self.config['sync']['log_level'])
        
        self.minio_client = MinIOClient(self.config, self.logger)
        self.ovms_config = OVMSConfigManager(
            self.config['ovms']['config_path'], 
            self.logger
        )
        self.model_tracker = ModelTracker(self.logger)
        
        self.bucket_name = self.config['minio']['bucket_name']
        self.models_path = self.config['sync']['models_path']
        self.sync_interval = self.config['sync']['interval']
        
        # Ensure models directory exists
        os.makedirs(self.models_path, exist_ok=True)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise ModelSyncError(f"Failed to load config from {config_path}: {str(e)}")
    
    def discover_models_in_minio(self) -> List[Dict[str, Any]]:
        """Discover available models in MinIO bucket."""
        discovered_models = []
        
        try:
            objects = self.minio_client.list_objects(self.bucket_name)
            
            # Group objects by model (assuming structure: model_name/version/files)
            model_groups = {}
            for obj in objects:
                key = obj['Key']
                if not key.endswith('/'):  # Skip directories
                    parts = key.split('/')
                    if len(parts) >= 2:
                        model_name = parts[0]
                        version = parts[1] if len(parts) > 2 else "1"
                        
                        if model_name not in model_groups:
                            model_groups[model_name] = {}
                        if version not in model_groups[model_name]:
                            model_groups[model_name][version] = []
                        
                        model_groups[model_name][version].append({
                            'key': key,
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'],
                            'etag': obj['ETag'].strip('"')
                        })
            
            # Convert to model list
            for model_name, versions in model_groups.items():
                for version, files in versions.items():
                    discovered_models.append({
                        'name': model_name,
                        'version': version,
                        'files': files,
                        'base_path': os.path.join(self.models_path, model_name)
                    })
            
            self.logger.info(f"Discovered {len(discovered_models)} models in MinIO")
            return discovered_models
            
        except Exception as e:
            self.logger.error(f"Failed to discover models: {str(e)}")
            return []
    
    def sync_model(self, model_info: Dict[str, Any]) -> bool:
        """Sync a specific model from MinIO to local storage."""
        model_name = model_info['name']
        version = model_info['version']
        files = model_info['files']
        base_path = model_info['base_path']
        
        try:
            # Create model directory structure
            model_version_path = os.path.join(base_path, version)
            os.makedirs(model_version_path, exist_ok=True)
            
            # Track if any files were updated
            files_updated = False
            
            for file_info in files:
                key = file_info['key']
                filename = os.path.basename(key)
                local_file_path = os.path.join(model_version_path, filename)
                
                # Check if file needs updating
                current_hash = file_info['etag']
                if self.model_tracker.is_model_changed(local_file_path, current_hash):
                    if self.minio_client.download_object(self.bucket_name, key, local_file_path):
                        files_updated = True
                        self.logger.info(f"Updated {filename} for model {model_name}:{version}")
                    else:
                        self.logger.error(f"Failed to download {filename} for model {model_name}:{version}")
                        return False
            
            if files_updated:
                # Update OVMS configuration
                if self.ovms_config.add_model(model_name, base_path):
                    if self.ovms_config.save_config():
                        self.logger.info(f"Successfully synced model {model_name}:{version}")
                        return True
                    else:
                        self.logger.error(f"Failed to save OVMS config after syncing {model_name}:{version}")
                        return False
                else:
                    self.logger.error(f"Failed to update OVMS config for {model_name}:{version}")
                    return False
            else:
                self.logger.debug(f"No updates needed for model {model_name}:{version}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to sync model {model_name}:{version}: {str(e)}")
            return False
    
    def sync_all_models(self) -> bool:
        """Sync all models from MinIO."""
        try:
            models = self.discover_models_in_minio()
            if not models:
                self.logger.warning("No models found in MinIO bucket")
                return True
            
            success_count = 0
            for model_info in models:
                if self.sync_model(model_info):
                    success_count += 1
            
            self.logger.info(f"Successfully synced {success_count}/{len(models)} models")
            return success_count == len(models)
            
        except Exception as e:
            self.logger.error(f"Failed to sync models: {str(e)}")
            return False
    
    def run_continuous_sync(self):
        """Run continuous model synchronization."""
        self.logger.info(f"Starting continuous model sync with {self.sync_interval}s interval")
        
        while True:
            try:
                self.logger.debug("Starting model sync cycle")
                self.sync_all_models()
                self.logger.debug(f"Sync cycle completed, sleeping for {self.sync_interval}s")
                time.sleep(self.sync_interval)
                
            except KeyboardInterrupt:
                self.logger.info("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in sync loop: {str(e)}")
                time.sleep(self.sync_interval)
    
    def run_health_check(self) -> bool:
        """Run health check to verify sidecar is working correctly."""
        try:
            # Check MinIO connectivity
            objects = self.minio_client.list_objects(self.bucket_name)
            self.logger.info(f"Health check: MinIO connection OK ({len(objects)} objects)")
            
            # Check models directory
            if os.path.exists(self.models_path):
                self.logger.info(f"Health check: Models directory accessible at {self.models_path}")
            else:
                self.logger.warning(f"Health check: Models directory not found at {self.models_path}")
                return False
            
            # Check OVMS config
            if os.path.exists(self.ovms_config.config_path):
                self.logger.info(f"Health check: OVMS config accessible at {self.ovms_config.config_path}")
            else:
                self.logger.warning(f"Health check: OVMS config not found at {self.ovms_config.config_path}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False

def main():
    """Main entry point."""
    # Get config path from environment variable
    config_path = os.environ.get('CONFIG_PATH', '/config/sync-config.yaml')
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    
    try:
        sidecar = ModelSyncSidecar(config_path)
        
        # Run initial health check
        if not sidecar.run_health_check():
            print("Health check failed, exiting...")
            sys.exit(1)
        
        # Perform initial sync
        sidecar.logger.info("Performing initial model sync...")
        sidecar.sync_all_models()
        
        # Start continuous sync
        sidecar.run_continuous_sync()
        
    except ModelSyncError as e:
        print(f"Model sync error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
