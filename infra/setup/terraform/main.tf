# -------------------------------
# Resource Group
# -------------------------------
resource "azurerm_resource_group" "ml" {
  name     = var.resource_group_name
  location = var.location
}

# -------------------------------
# User Assigned Managed Identity
# -------------------------------
resource "azurerm_user_assigned_identity" "aml_identity" {
  name                = "aml-managed-identity"
  resource_group_name = azurerm_resource_group.ml.name
  location            = azurerm_resource_group.ml.location
}

# -------------------------------
# Azure Container Registry
# -------------------------------
resource "azurerm_container_registry" "acr" {
  name                = var.acr_name
  resource_group_name = azurerm_resource_group.ml.name
  location            = azurerm_resource_group.ml.location
  sku                 = "Standard"
  admin_enabled       = false
}

# -------------------------------
# Storage Account
# -------------------------------
resource "azurerm_storage_account" "blob" {
  name                     = var.storage_account_name
  resource_group_name      = azurerm_resource_group.ml.name
  location                 = azurerm_resource_group.ml.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

# -------------------------------
# Blob Containers
# -------------------------------
resource "azurerm_storage_container" "datasets" {
  name                  = "datasets"
  storage_account_name  = azurerm_storage_account.blob.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "mlflow" {
  name                  = "azureml-mlflow"
  storage_account_name  = azurerm_storage_account.blob.name
  container_access_type = "private"
}

# -------------------------------
# Key Vault
# -------------------------------
resource "azurerm_key_vault" "kv" {
  name                        = "${var.workspace_name}-kv"
  location                    = azurerm_resource_group.ml.location
  resource_group_name         = azurerm_resource_group.ml.name
  tenant_id                   = data.azurerm_client_config.current.tenant_id
  sku_name                    = "standard"
  purge_protection_enabled    = false
}

# -------------------------------
# Application Insights
# -------------------------------
resource "azurerm_application_insights" "ai" {
  name                = "${var.workspace_name}-ai"
  location            = azurerm_resource_group.ml.location
  resource_group_name = azurerm_resource_group.ml.name
  application_type    = "web"
}

# -------------------------------
# Azure ML Workspace
# -------------------------------
resource "azurerm_machine_learning_workspace" "aml" {
  name                = var.workspace_name
  location            = azurerm_resource_group.ml.location
  resource_group_name = azurerm_resource_group.ml.name

  identity {
    type = "SystemAssigned"
  }

  container_registry_id   = azurerm_container_registry.acr.id
  storage_account_id      = azurerm_storage_account.blob.id
  key_vault_id            = azurerm_key_vault.kv.id
  application_insights_id = azurerm_application_insights.ai.id
}
# -------------------------------
# GPU Compute Cluster
# -------------------------------
resource "azurerm_machine_learning_compute_cluster" "cpu" {
  name                          = "cpu-cluster"
  location                      = azurerm_resource_group.ml.location
  machine_learning_workspace_id = azurerm_machine_learning_workspace.aml.id
  vm_size                       = "Standard_DS3_v2"

  vm_priority = "Dedicated"

  scale_settings {
    min_node_count                    = 0
    max_node_count                    = 2
    scale_down_nodes_after_idle_duration = "PT15M"
  }

  identity {
    type = "SystemAssigned"
  }
}
# -------------------------------
# Role Assignments
# -------------------------------
resource "azurerm_role_assignment" "storage_access" {
  principal_id         = azurerm_user_assigned_identity.aml_identity.principal_id
  role_definition_name = "Storage Blob Data Contributor"
  scope                = azurerm_storage_account.blob.id
}

resource "azurerm_role_assignment" "acr_pull" {
  principal_id         = azurerm_user_assigned_identity.aml_identity.principal_id
  role_definition_name = "AcrPull"
  scope                = azurerm_container_registry.acr.id
}
