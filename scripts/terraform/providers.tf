terraform {
  required_version = "~>1.6"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~>3.0"
    }
  }
}

provider "azurerm" {
  features {}
  # SCENARIO GUIDE:
  # - If all required providers are already registered, the setting below is fine.
  # - If providers are not registered and you have permissions, remove the below line
  # - If you lack permissions, leave the below line as it is and have your
  #   Azure admin manually register the necessary providers before running `terraform apply`.
  # https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/resources/resource_provider_registration
  skip_provider_registration = true 
}