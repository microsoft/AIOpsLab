# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Abstracts the configuration file for AIOpsLab."""

import yaml


class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        with open(self.config_path, "r") as file:
            return yaml.safe_load(file)

    def get(self, key, default=None):
        return self.config.get(key, default)


def get_kube_context():
    """Get the kubernetes context from config.yml with consistent priority logic
    
    Priority (highest to lowest):
    1. Explicit kube_context setting
    2. If k8s_host is 'kind', construct from kind_cluster_name
    3. No context (return None to use system default)
    
    Returns:
        str or None: Context name if should be specified, None if should use default
    """
    try:
        # Import BASE_DIR inside function to avoid circular import
        from aiopslab.paths import BASE_DIR
        config_yaml = Config(BASE_DIR / "config.yml")
        
        # Priority 1: Explicit kube_context setting
        kube_context = config_yaml.get("kube_context")
        if kube_context:
            return kube_context
        
        # Priority 2: If k8s_host is kind, construct from kind_cluster_name
        k8s_host = config_yaml.get("k8s_host")
        if k8s_host == "kind":
            cluster_name = config_yaml.get("kind_cluster_name", "kind")
            return f"kind-{cluster_name}"
        
        # Priority 3: No context specified, use system default
        return None
        
    except Exception:
        # If config reading fails, use system default
        return None


# Usage example
# config = Config(Path("config.yml"))
# data_dir = config.get("data_dir")
# qualitative_eval = config.get("qualitative_eval")
