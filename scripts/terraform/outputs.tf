output "controller" {
  description = "Controller node details"
  value = {
    name       = azurerm_linux_virtual_machine.controller.name
    public_ip  = azurerm_public_ip.controller.ip_address
    private_ip = azurerm_network_interface.controller.ip_configuration[0].private_ip_address
    username   = var.admin_username
  }
}

output "workers" {
  description = "Worker nodes details"
  value = [
    for key, vm in azurerm_linux_virtual_machine.workers : {
      name       = vm.name
      public_ip  = azurerm_public_ip.workers[key].ip_address
      private_ip = azurerm_network_interface.workers[key].ip_configuration[0].private_ip_address
      username   = var.admin_username
    }
  ]
}

output "cluster_info" {
  description = "Complete cluster information"
  value = {
    resource_group = data.azurerm_resource_group.rg.name
    location       = data.azurerm_resource_group.rg.location
    prefix         = var.prefix
    worker_count   = var.worker_vm_count
    vm_size        = var.vm_size
  }
}

output "ssh_config" {
  description = "SSH configuration"
  value = {
    public_key_path  = var.ssh_public_key_path
    private_key_path = replace(var.ssh_public_key_path, ".pub", "")
    username         = var.admin_username
  }
  sensitive = true
}