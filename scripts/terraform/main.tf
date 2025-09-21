# Create virtual network
resource "azurerm_virtual_network" "aiopslab_network" {
  name                = "${var.resource_name_prefix}_aiopslabVnet"
  address_space       = ["10.0.0.0/16"]
  location            = local.resource_location
  resource_group_name = local.resource_group_name
}

# Create subnet
resource "azurerm_subnet" "aiopslab_subnet" {
  name                 = "${var.resource_name_prefix}_aiopslabSubnet"
  resource_group_name  = local.resource_group_name
  virtual_network_name = azurerm_virtual_network.aiopslab_network.name
  address_prefixes     = ["10.0.1.0/24"]
}

# Create public IPs
resource "azurerm_public_ip" "aiopslab_public_ip_1" {
  name                = "${var.resource_name_prefix}_aiopslabPublicIP_1"
  location            = var.resource_location
  resource_group_name = local.resource_group_name
  allocation_method   = "Dynamic"
}

resource "azurerm_public_ip" "aiopslab_public_ip_2" {
  name                = "${var.resource_name_prefix}_aiopslabPublicIP_2"
  location            = var.resource_location
  resource_group_name = local.resource_group_name
  allocation_method   = "Dynamic"
}

# Create Network Security Group and rule with only CorpNet access
resource "azurerm_network_security_group" "aiopslab_nsg_1" {
  name                = "${var.resource_name_prefix}_aiopslabNSG_1"
  location            = var.resource_location
  resource_group_name = local.resource_group_name

  security_rule {
    name                       = "SSH"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
}

resource "azurerm_network_security_group" "aiopslab_nsg_2" {
  name                = "${var.resource_name_prefix}_aiopslabNSG_2"
  location            = var.resource_location
  resource_group_name = local.resource_group_name

  security_rule {
    name                       = "SSH"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
}

# Create network interfaces
resource "azurerm_network_interface" "aiopslab_nic_1" {
  name                = "${var.resource_name_prefix}_aiopslabNIC_1"
  location            = var.resource_location
  resource_group_name = local.resource_group_name

  ip_configuration {
    name                          = "${var.resource_name_prefix}_aioplabNICConfiguration_1"
    subnet_id                     = azurerm_subnet.aiopslab_subnet.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.aiopslab_public_ip_1.id
  }
}

resource "azurerm_network_interface" "aiopslab_nic_2" {
  name                = "${var.resource_name_prefix}_aiopslabNIC_2"
  location            = var.resource_location
  resource_group_name = local.resource_group_name

  ip_configuration {
    name                          = "${var.resource_name_prefix}_aioplabNICConfiguration_2"
    subnet_id                     = azurerm_subnet.aiopslab_subnet.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.aiopslab_public_ip_2.id
  }
}

# Connect the security groups to the network interfaces
resource "azurerm_network_interface_security_group_association" "aiopslab_nsg_association_1" {
  network_interface_id      = azurerm_network_interface.aiopslab_nic_1.id
  network_security_group_id = azurerm_network_security_group.aiopslab_nsg_1.id
}

resource "azurerm_network_interface_security_group_association" "aiopslab_nsg_association_2" {
  network_interface_id      = azurerm_network_interface.aiopslab_nic_2.id
  network_security_group_id = azurerm_network_security_group.aiopslab_nsg_2.id
}

resource "random_id" "random_id" {
  byte_length = 8
}


# Create storage accounts for boot diagnostics
resource "azurerm_storage_account" "aiopslab_storage_account_1" {
  # storage account names can only consist of lowercase letters and numbers
  name                     = "diag${random_id.random_id.hex}1"
  location                 = var.resource_location
  resource_group_name      = var.resource_group_name
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

resource "azurerm_storage_account" "aiopslab_storage_account_2" {
  name                     = "diag${random_id.random_id.hex}2"
  location                 = var.resource_location
  resource_group_name      = var.resource_group_name
  account_tier             = "Standard"
  account_replication_type = "LRS"
}



# Create virtual machines
resource "azurerm_linux_virtual_machine" "aiopslab_vm_1" {
  name                  = "${var.resource_name_prefix}_aiopslabVM_1"
  location              = var.resource_location
  resource_group_name   = var.resource_group_name
  network_interface_ids = [azurerm_network_interface.aiopslab_nic_1.id]
  size                  = "Standard_D4s_v3"

  os_disk {
    name                 = "${var.resource_name_prefix}_OsDisk_1"
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts-gen2"
    version   = "latest"
  }

  computer_name  = "kubeController"
  admin_username = var.username

  admin_ssh_key {
    username   = var.username
    public_key = azapi_resource_action.aiopslab_ssh_public_key_gen_1.output.publicKey
  }

  boot_diagnostics {
    storage_account_uri = azurerm_storage_account.aiopslab_storage_account_1.primary_blob_endpoint
  }
}

resource "azurerm_linux_virtual_machine" "aiopslab_vm_2" {
  name                  = "${var.resource_name_prefix}_aiopslabVM_2"
  location              = var.resource_location
  resource_group_name   = var.resource_group_name
  network_interface_ids = [azurerm_network_interface.aiopslab_nic_2.id]
  size                  = "Standard_F16s_v2"

  os_disk {
    name                 = "${var.resource_name_prefix}_OsDisk_2"
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts-gen2"
    version   = "latest"
  }

  computer_name  = "kubeWorker1"
  admin_username = var.username

  admin_ssh_key {
    username   = var.username
    public_key = azapi_resource_action.aiopslab_ssh_public_key_gen_2.output.publicKey
  }

  boot_diagnostics {
    storage_account_uri = azurerm_storage_account.aiopslab_storage_account_2.primary_blob_endpoint
  }
}
