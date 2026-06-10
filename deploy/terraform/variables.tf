variable "region" {
  description = "AWS region for the cluster and artifact bucket."
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "EKS cluster name."
  type        = string
  default     = "vision-mlops"
}

variable "kubernetes_version" {
  description = "EKS control-plane version."
  type        = string
  default     = "1.29"
}

variable "serving_instance_types" {
  description = "Instance types for the model-serving node group (CPU inference)."
  type        = list(string)
  default     = ["c6i.xlarge"]
}

variable "artifact_bucket_name" {
  description = "S3 bucket for promoted model artifacts (ONNX exports)."
  type        = string
  default     = "vision-mlops-artifacts"
}
