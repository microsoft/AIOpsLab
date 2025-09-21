variable "resource_location" {
  type        = string
  default     = "westus2"
  description = "Location of the resource."
}

variable "username" {
  type        = string
  description = "The username for the local account that will be created on the new VM."
  default     = "azureuser"
}

variable "resource_name_prefix" {
  type        = string
  description = "Prefix for all the resource names."
}

# TODO: Generate random text instead of taking prefix from user? Below will keep it unique for each resource group.
# resource "random_id" "resource_name_prefix" {
#  keepers = {
#    resource_group = var.resource_group_name
#  }
#
#  byte_length = 8
#}


variable "resource_group_name" {
  type        = string
  description = "The name of the resource group where the all the resources should be created."
}

variable "create_resource_group" {
  type        = bool
  default     = false
  description = "Whether to create a new resource group or use an existing one."
}

variable "subscription_id" {
  type        = string
  description = "Azure subscription ID"
  default     = ""
}