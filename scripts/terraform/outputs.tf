output "controller_public_ip" {
  value = azurerm_public_ip.controller.ip_address
}

output "worker_public_ips" {
  value = [for vm in azurerm_public_ip.workers : vm.ip_address]
}