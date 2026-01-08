variable "location" {
  description = "Azure region"
  type        = string
  default     = "canadaeast"
}

variable "resource_group_name" {
  description = "Resource group name"
  type        = string
  default     = "rg-email-ml-1"
}

variable "workspace_name" {
  description = "Azure ML workspace name"
  type        = string
  default     = "aml-email-1"
}

variable "acr_name" {
  description = "Azure Container Registry name (globally unique)"
  type        = string
}

variable "storage_account_name" {
  description = "Storage account name (globally unique)"
  type        = string
}
