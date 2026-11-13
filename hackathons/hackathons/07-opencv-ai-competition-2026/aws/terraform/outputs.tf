output "evidence_bucket" {
  description = "Bucket for JSON incident/telemetry records (no raw frames by default)."
  value       = aws_s3_bucket.evidence.bucket
}

output "lambda_function" {
  description = "Arm64 Lambda function name."
  value       = aws_lambda_function.vision.function_name
}

output "ecr_arm64" {
  description = "ECR repository for the Graviton image."
  value       = aws_ecr_repository.arm64.repository_url
}

output "ecr_x86" {
  description = "ECR repository for the x86 reference image."
  value       = aws_ecr_repository.x86.repository_url
}

output "ecs_cluster" {
  description = "ECS cluster for continuous ingestion."
  value       = aws_ecs_cluster.vision.name
}
