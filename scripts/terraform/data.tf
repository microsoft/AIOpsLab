# Resource group management
resource "azurerm_resource_group" "rg" {
  count    = var.create_resource_group ? 1 : 0
  name     = var.resource_group_name
  location = var.resource_location
}

data "azurerm_resource_group" "rg" {
  count = var.create_resource_group ? 0 : 1
  name  = var.resource_group_name
}

locals {
  resource_group_id   = var.create_resource_group ? azurerm_resource_group.rg[0].id : data.azurerm_resource_group.rg[0].id
  resource_group_name = var.resource_group_name
  resource_location   = var.resource_location
}