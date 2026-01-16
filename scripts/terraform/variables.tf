variable "prefix" {
  type        = string
  description = "A unique prefix for the resource names."
  default     = "aiopslab"
}

variable "resource_group_name" {
  type        = string
  description = "The name of the existing Azure Resource Group to deploy resources into."
  default     = "aiopslab-rg"
}

variable "location" {
  type        = string
  description = "The Azure region to deploy the resources in."
  default     = "eastus"
}

variable "admin_username" {
  type        = string
  description = "The username for the VMs."
  default     = "azureuser"
}

variable "ssh_public_key_path" {
  type        = string
  description = "The path to the SSH public key file."
  default     = "~/.ssh/id_rsa.pub"
}

variable "vm_size" {
  type        = string
  description = "The size of the virtual machines."
  default     = "Standard_B2s"
}

variable "os_disk_type" {
  type        = string
  description = "The type of the OS disk. Allowed values: Standard_LRS, Premium_LRS, StandardSSD_LRS."
  default     = "Standard_LRS"
}

variable "os_publisher" {
  type        = string
  description = "The publisher of the OS image."
  default     = "Canonical"
}

variable "os_offer" {
  type        = string
  description = "The offer of the OS image."
  default     = "0001-com-ubuntu-server-jammy"
}

variable "os_sku" {
  type        = string
  description = "The SKU of the OS image. The default is Ubuntu 22.04 LTS."
  default     = "22_04-lts"
}

variable "worker_vm_count" {
  type        = number
  description = "The number of worker nodes to create."
  default     = 2
}