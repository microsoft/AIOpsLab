
resource "azapi_resource" "aiopslab_ssh_public_key_1" {
  type      = "Microsoft.Compute/sshPublicKeys@2022-11-01"
  name      = "${var.resource_name_prefix}_ssh_public_key_1"
  location  = var.resource_location
  parent_id = local.resource_group_id
}

resource "azapi_resource_action" "aiopslab_ssh_public_key_gen_1" {
  type        = "Microsoft.Compute/sshPublicKeys@2022-11-01"
  resource_id = azapi_resource.aiopslab_ssh_public_key_1.id
  action      = "generateKeyPair"
  method      = "POST"

  response_export_values = ["publicKey", "privateKey"]
}


resource "azapi_resource" "aiopslab_ssh_public_key_2" {
  type      = "Microsoft.Compute/sshPublicKeys@2022-11-01"
  name      = "${var.resource_name_prefix}_ssh_public_key_2"
  location  = var.resource_location
  parent_id = local.resource_group_id
}

resource "azapi_resource_action" "aiopslab_ssh_public_key_gen_2" {
  type        = "Microsoft.Compute/sshPublicKeys@2022-11-01"
  resource_id = azapi_resource.aiopslab_ssh_public_key_2.id
  action      = "generateKeyPair"
  method      = "POST"

  response_export_values = ["publicKey", "privateKey"]
}


