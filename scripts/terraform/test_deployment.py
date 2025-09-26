#!/usr/bin/env python3
"""
Basic tests for the AIOpsLab deployment automation.
These tests validate configuration, syntax, and basic functionality.
"""

import unittest
import sys
import os
import yaml
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# Add the terraform directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from deploy_unified import AIOpsLabDeployment


class TestAIOpsLabDeployment(unittest.TestCase):
    """Test cases for the AIOpsLabDeployment class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.deployment = AIOpsLabDeployment()
        
    def test_default_config_loading(self):
        """Test that default configuration is loaded correctly."""
        self.assertIsInstance(self.deployment.config, dict)
        self.assertIn('resource_location', self.deployment.config)
        self.assertEqual(self.deployment.config['resource_location'], 'westus2')
        self.assertEqual(self.deployment.config['username'], 'azureuser')
        
    def test_config_file_loading(self):
        """Test loading configuration from file."""
        test_config = {
            'resource_group_name': 'test-rg',
            'resource_name_prefix': 'test-prefix',
            'custom_setting': 'test-value'
        }
        
        with patch('builtins.open', mock_open(read_data=yaml.dump(test_config))):
            with patch('os.path.exists', return_value=True):
                deployment = AIOpsLabDeployment('test_config.yml')
                
        self.assertEqual(deployment.config['resource_group_name'], 'test-rg')
        self.assertEqual(deployment.config['custom_setting'], 'test-value')
        # Default values should still be present
        self.assertEqual(deployment.config['resource_location'], 'westus2')
        
    def test_generate_ansible_inventory(self):
        """Test Ansible inventory generation from Terraform outputs."""
        test_outputs = {
            'public_ip_address_1': '1.2.3.4',
            'public_ip_address_2': '5.6.7.8',
            'username': 'testuser'
        }
        
        with patch('builtins.open', mock_open()) as mock_file:
            inventory_file = self.deployment.generate_ansible_inventory(test_outputs)
            
        # Check that the file was written
        mock_file.assert_called_once()
        handle = mock_file()
        written_content = ''.join(call.args[0] for call in handle.write.call_args_list)
        
        # Parse the written YAML content
        inventory = yaml.safe_load(written_content)
        
        # Validate inventory structure
        self.assertIn('all', inventory)
        self.assertIn('children', inventory['all'])
        self.assertIn('control_nodes', inventory['all']['children'])
        self.assertIn('worker_nodes', inventory['all']['children'])
        
        # Validate controller configuration
        controller = inventory['all']['children']['control_nodes']['hosts']['control_node']
        self.assertEqual(controller['ansible_host'], '1.2.3.4')
        self.assertEqual(controller['ansible_user'], 'testuser')
        
        # Validate worker configuration
        worker = inventory['all']['children']['worker_nodes']['hosts']['worker_node_1']
        self.assertEqual(worker['ansible_host'], '5.6.7.8')
        self.assertEqual(worker['ansible_user'], 'testuser')
        
    @patch('subprocess.run')
    def test_run_command_success(self, mock_run):
        """Test successful command execution."""
        mock_result = MagicMock()
        mock_result.stdout = 'test output\n'
        mock_run.return_value = mock_result
        
        result = self.deployment.run_command(['echo', 'test'], capture_output=True)
        
        self.assertEqual(result, 'test output')
        mock_run.assert_called_once()
        
    @patch('subprocess.run')
    def test_run_command_failure(self, mock_run):
        """Test command execution failure handling."""
        from subprocess import CalledProcessError
        
        mock_run.side_effect = CalledProcessError(1, ['false'])
        
        with self.assertRaises(CalledProcessError):
            self.deployment.run_command(['false'])
            
    def test_save_private_keys(self):
        """Test SSH private key saving functionality."""
        test_outputs = {
            'key_data_1': 'test-private-key-1',
            'key_data_2': 'test-private-key-2'
        }
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('os.chmod') as mock_chmod:
                key_files = self.deployment.save_private_keys(test_outputs)
                
        # Check that both key files were created
        self.assertEqual(len(key_files), 2)
        self.assertIn('vm_1', key_files)
        self.assertIn('vm_2', key_files)
        
        # Check that files were written with correct permissions
        self.assertEqual(mock_chmod.call_count, 2)
        for call in mock_chmod.call_args_list:
            self.assertEqual(call[0][1], 0o600)  # Proper SSH key permissions
            
    @patch('subprocess.run')
    def test_get_terraform_outputs(self, mock_run):
        """Test Terraform output retrieval."""
        test_outputs = {
            'public_ip_address_1': {'value': '1.2.3.4'},
            'username': {'value': 'testuser'},
            'key_data_1': {'value': 'test-key', 'sensitive': True}
        }
        
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(test_outputs)
        mock_run.return_value = mock_result
        
        outputs = self.deployment.get_terraform_outputs()
        
        # Check that values were extracted correctly
        self.assertEqual(outputs['public_ip_address_1'], '1.2.3.4')
        self.assertEqual(outputs['username'], 'testuser')
        self.assertEqual(outputs['key_data_1'], 'test-key')
        
    def test_validate_prerequisites_config_missing(self):
        """Test prerequisite validation with missing configuration."""
        # Clear required config
        self.deployment.config['resource_group_name'] = ''
        
        # Mock successful tool checks
        with patch.object(self.deployment, 'run_command') as mock_run:
            mock_run.return_value = '/usr/bin/terraform'
            result = self.deployment.validate_prerequisites()
            
        self.assertFalse(result)


class TestConfigurationFiles(unittest.TestCase):
    """Test configuration file templates and examples."""
    
    def test_config_example_is_valid_yaml(self):
        """Test that the configuration example file is valid YAML."""
        config_file = Path(__file__).parent / 'config.yml.example'
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            self.assertIsInstance(config, dict)
            # Check for required fields in example
            expected_fields = ['resource_group_name', 'resource_name_prefix', 'resource_location']
            for field in expected_fields:
                self.assertIn(field, config)


class TestAnsibleIntegration(unittest.TestCase):
    """Test Ansible playbook integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.ansible_dir = Path(__file__).parent.parent / 'ansible'
        
    def test_ansible_playbooks_exist(self):
        """Test that required Ansible playbooks exist."""
        required_playbooks = ['setup_common.yml', 'remote_setup_controller_worker.yml']
        
        for playbook in required_playbooks:
            playbook_path = self.ansible_dir / playbook
            self.assertTrue(playbook_path.exists(), f"Playbook {playbook} not found at {playbook_path}")
            
    def test_ansible_playbooks_valid_yaml(self):
        """Test that Ansible playbooks are valid YAML."""
        playbooks = ['setup_common.yml', 'remote_setup_controller_worker.yml']
        
        for playbook in playbooks:
            playbook_path = self.ansible_dir / playbook
            if playbook_path.exists():
                with open(playbook_path, 'r') as f:
                    try:
                        yaml.safe_load(f)
                    except yaml.YAMLError as e:
                        self.fail(f"Playbook {playbook} contains invalid YAML: {e}")


class TestTerraformConfiguration(unittest.TestCase):
    """Test Terraform configuration files."""
    
    def setUp(self):
        """Set up test environment."""
        self.terraform_dir = Path(__file__).parent
        
    def test_terraform_files_exist(self):
        """Test that required Terraform files exist."""
        required_files = [
            'main.tf', 'variables.tf', 'outputs.tf', 
            'ssh.tf', 'data.tf', 'providers.tf'
        ]
        
        for tf_file in required_files:
            file_path = self.terraform_dir / tf_file
            self.assertTrue(file_path.exists(), f"Terraform file {tf_file} not found")
            
    def test_variables_have_descriptions(self):
        """Test that Terraform variables have descriptions."""
        variables_file = self.terraform_dir / 'variables.tf'
        
        if variables_file.exists():
            with open(variables_file, 'r') as f:
                content = f.read()
                
            # Check that variable blocks contain descriptions
            self.assertIn('description', content)
            # Check for specific required variables
            self.assertIn('resource_group_name', content)
            self.assertIn('resource_name_prefix', content)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)