variable "region" {
  description = "AWS region for all resources."
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment suffix."
  type        = string
  default     = "demo"
}

variable "account_suffix" {
  description = "Globally unique suffix for the evidence bucket name."
  type        = string
}

variable "image_uri" {
  description = "ECR image URI for the arm64 Lambda/ECS image built from aws/Dockerfile.arm64."
  type        = string
}

variable "lambda_memory_mb" {
  description = "Lambda memory (more memory also means more CPU)."
  type        = number
  default     = 2048
}

variable "log_retention_days" {
  description = "CloudWatch retention for Lambda and ECS logs."
  type        = number
  default     = 14
}

variable "force_destroy" {
  description = "Allow terraform destroy to delete the evidence bucket (demo only)."
  type        = bool
  default     = false
}
