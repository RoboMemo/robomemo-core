# ============================================================
# RoboMemo — Terraform AWS Singapore Deployment
# Infrastructure: VPC + ECS Fargate + ALB + S3 + RDS
# Region: ap-southeast-1 (Singapore)
# ============================================================

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket  = "robomemo-terraform-state"
    key     = "singapore/terraform.tfstate"
    region  = "ap-southeast-1"
    encrypt = true
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "RoboMemo"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Compliance  = "ISO27001-GDPR-SOC2"
    }
  }
}

# ─── Variables ─────────────────────────────────────────────────

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "ap-southeast-1" # Singapore
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "app_name" {
  description = "Application name"
  type        = string
  default     = "robomemo"
}

variable "db_password" {
  description = "PostgreSQL database password"
  type        = string
  sensitive   = true
}

variable "jwt_secret" {
  description = "JWT signing secret"
  type        = string
  sensitive   = true
}

variable "encryption_key" {
  description = "AES-256 encryption master key (64 hex chars)"
  type        = string
  sensitive   = true
}

variable "domain_name" {
  description = "Domain for the platform"
  type        = string
  default     = "platform.robomemo.io"
}

# ─── VPC ───────────────────────────────────────────────────────

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = { Name = "${var.app_name}-vpc" }
}

resource "aws_subnet" "public_a" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = true
  tags = { Name = "${var.app_name}-public-a" }
}

resource "aws_subnet" "public_b" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.2.0/24"
  availability_zone       = "${var.aws_region}b"
  map_public_ip_on_launch = true
  tags = { Name = "${var.app_name}-public-b" }
}

resource "aws_subnet" "private_a" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.10.0/24"
  availability_zone = "${var.aws_region}a"
  tags = { Name = "${var.app_name}-private-a" }
}

resource "aws_subnet" "private_b" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.11.0/24"
  availability_zone = "${var.aws_region}b"
  tags = { Name = "${var.app_name}-private-b" }
}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${var.app_name}-igw" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }
  tags = { Name = "${var.app_name}-public-rt" }
}

resource "aws_route_table_association" "public_a" {
  subnet_id      = aws_subnet.public_a.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "public_b" {
  subnet_id      = aws_subnet.public_b.id
  route_table_id = aws_route_table.public.id
}

# NAT Gateway for private subnets
resource "aws_eip" "nat" {
  domain = "vpc"
  tags   = { Name = "${var.app_name}-nat-eip" }
}

resource "aws_nat_gateway" "nat" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public_a.id
  tags          = { Name = "${var.app_name}-nat" }
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat.id
  }
  tags = { Name = "${var.app_name}-private-rt" }
}

resource "aws_route_table_association" "private_a" {
  subnet_id      = aws_subnet.private_a.id
  route_table_id = aws_route_table.private.id
}

resource "aws_route_table_association" "private_b" {
  subnet_id      = aws_subnet.private_b.id
  route_table_id = aws_route_table.private.id
}

# ─── Security Groups ──────────────────────────────────────────

resource "aws_security_group" "alb" {
  name_prefix = "${var.app_name}-alb-"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTP (redirect)"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.app_name}-alb-sg" }
}

resource "aws_security_group" "app" {
  name_prefix = "${var.app_name}-app-"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "From ALB"
    from_port       = 3001
    to_port         = 3001
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.app_name}-app-sg" }
}

resource "aws_security_group" "db" {
  name_prefix = "${var.app_name}-db-"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "PostgreSQL from App"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.app.id]
  }

  tags = { Name = "${var.app_name}-db-sg" }
}

# ─── WAF (Geo-Blocking China/HK) ──────────────────────────────

resource "aws_wafv2_ip_set" "blocked_countries" {
  name               = "${var.app_name}-blocked-ips"
  scope              = "REGIONAL"
  ip_address_version = "IPV4"
  addresses          = [] # Managed via WAF geo-match rule below
}

resource "aws_wafv2_web_acl" "geo_fence" {
  name  = "${var.app_name}-geo-fence"
  scope = "REGIONAL"

  default_action { allow {} }

  # Block China (CN) and Hong Kong (HK)
  rule {
    name     = "block-cn-hk"
    priority = 1

    action { block {} }

    statement {
      geo_match_statement {
        country_codes = ["CN", "HK"]
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${var.app_name}-geo-block"
      sampled_requests_enabled   = true
    }
  }

  # Rate limiting
  rule {
    name     = "rate-limit"
    priority = 2

    action { block {} }

    statement {
      rate_based_statement {
        limit              = 2000
        aggregate_key_type = "IP"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${var.app_name}-rate-limit"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${var.app_name}-waf"
    sampled_requests_enabled   = true
  }
}

# ─── S3 Bucket (Video & Frame Storage) ────────────────────────

resource "aws_s3_bucket" "videos" {
  bucket = "${var.app_name}-videos-${var.aws_region}"
  tags   = { Name = "${var.app_name}-videos" }
}

resource "aws_s3_bucket_versioning" "videos" {
  bucket = aws_s3_bucket.videos.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "videos" {
  bucket = aws_s3_bucket.videos.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "videos" {
  bucket                  = aws_s3_bucket.videos.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ─── ALB (Application Load Balancer) ──────────────────────────

resource "aws_lb" "main" {
  name               = "${var.app_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = [aws_subnet.public_a.id, aws_subnet.public_b.id]

  tags = { Name = "${var.app_name}-alb" }
}

resource "aws_lb_target_group" "app" {
  name        = "${var.app_name}-tg"
  port        = 3001
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    path                = "/api/health"
    healthy_threshold   = 2
    unhealthy_threshold = 5
    interval            = 30
  }
}

resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.main.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = aws_acm_certificate.main.arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}

resource "aws_lb_listener" "http_redirect" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "redirect"
    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}

# Associate WAF with ALB
resource "aws_wafv2_web_acl_association" "alb" {
  resource_arn = aws_lb.main.arn
  web_acl_arn  = aws_wafv2_web_acl.geo_fence.arn
}

# ─── ACM Certificate ──────────────────────────────────────────

resource "aws_acm_certificate" "main" {
  domain_name               = var.domain_name
  subject_alternative_names = ["api.${var.domain_name}"]
  validation_method         = "DNS"

  lifecycle {
    create_before_destroy = true
  }
}

# ─── ECS Cluster + Fargate Service ────────────────────────────

resource "aws_ecs_cluster" "main" {
  name = "${var.app_name}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${var.app_name}"
  retention_in_days = 90 # ISO 27001 log retention
}

resource "aws_iam_role" "ecs_task_execution" {
  name = "${var.app_name}-ecs-exec"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_ecs_task_definition" "app" {
  family                   = "${var.app_name}-app"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 1024
  memory                   = 2048
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn

  container_definitions = jsonencode([{
    name      = "app"
    image     = "${aws_ecr_repository.app.repository_url}:latest"
    essential = true

    portMappings = [{
      containerPort = 3001
      protocol      = "tcp"
    }]

    environment = [
      { name = "NODE_ENV", value = "production" },
      { name = "PORT", value = "3001" },
    ]

    secrets = [
      { name = "JWT_SECRET", valueFrom = aws_ssm_parameter.jwt_secret.arn },
      { name = "ENCRYPTION_MASTER_KEY", valueFrom = aws_ssm_parameter.encryption_key.arn },
      { name = "GEMINI_API_KEY", valueFrom = aws_ssm_parameter.gemini_key.arn },
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.app.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "app"
      }
    }

    healthCheck = {
      command  = ["CMD-SHELL", "wget --no-verbose --tries=1 --spider http://localhost:3001/api/health || exit 1"]
      interval = 30
      timeout  = 5
      retries  = 3
    }
  }])
}

resource "aws_ecs_service" "app" {
  name            = "${var.app_name}-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = 2
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = [aws_subnet.private_a.id, aws_subnet.private_b.id]
    security_groups  = [aws_security_group.app.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = "app"
    container_port   = 3001
  }
}

# ─── ECR Repository ───────────────────────────────────────────

resource "aws_ecr_repository" "app" {
  name                 = var.app_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

# ─── SSM Parameters (Secrets) ─────────────────────────────────

resource "aws_ssm_parameter" "jwt_secret" {
  name  = "/${var.app_name}/jwt-secret"
  type  = "SecureString"
  value = var.jwt_secret
}

resource "aws_ssm_parameter" "encryption_key" {
  name  = "/${var.app_name}/encryption-key"
  type  = "SecureString"
  value = var.encryption_key
}

resource "aws_ssm_parameter" "gemini_key" {
  name  = "/${var.app_name}/gemini-api-key"
  type  = "SecureString"
  value = "placeholder" # Set via AWS Console or CI/CD

  lifecycle { ignore_changes = [value] }
}

# ─── Outputs ───────────────────────────────────────────────────

output "alb_dns" {
  description = "ALB DNS name"
  value       = aws_lb.main.dns_name
}

output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.app.repository_url
}

output "s3_bucket" {
  description = "Video storage bucket"
  value       = aws_s3_bucket.videos.id
}

output "waf_acl_id" {
  description = "WAF Web ACL ID (geo-fence)"
  value       = aws_wafv2_web_acl.geo_fence.id
}

# ============================================================
# UAE Secondary Region — Data Residency Failover
# Region: me-central-1 (UAE Abu Dhabi)
# Purpose: Cross-region replication + DR (no China/HK contact)
# ============================================================

provider "aws" {
  alias  = "uae"
  region = "me-central-1" # AWS UAE (Abu Dhabi)

  default_tags {
    tags = {
      Project     = "RoboMemo"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Compliance  = "ISO27001-GDPR-SOC2"
      Region      = "UAE-Secondary"
    }
  }
}

# ─── UAE S3 Bucket (Replication Destination) ──────────────────

resource "aws_s3_bucket" "videos_uae" {
  provider = aws.uae
  bucket   = "${var.app_name}-videos-me-central-1"
  tags     = { Name = "${var.app_name}-videos-uae" }
}

resource "aws_s3_bucket_versioning" "videos_uae" {
  provider = aws.uae
  bucket   = aws_s3_bucket.videos_uae.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "videos_uae" {
  provider = aws.uae
  bucket   = aws_s3_bucket.videos_uae.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "videos_uae" {
  provider                = aws.uae
  bucket                  = aws_s3_bucket.videos_uae.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ─── Cross-Region Replication: Singapore → UAE ────────────────

resource "aws_iam_role" "s3_replication" {
  name = "${var.app_name}-s3-replication"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRole"
      Principal = { Service = "s3.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "s3_replication" {
  name = "${var.app_name}-s3-replication"
  role = aws_iam_role.s3_replication.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetReplicationConfiguration",
          "s3:ListBucket"
        ]
        Resource = aws_s3_bucket.videos.arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObjectVersionForReplication",
          "s3:GetObjectVersionAcl",
          "s3:GetObjectVersionTagging"
        ]
        Resource = "${aws_s3_bucket.videos.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ReplicateObject",
          "s3:ReplicateDelete",
          "s3:ReplicateTags"
        ]
        Resource = "${aws_s3_bucket.videos_uae.arn}/*"
      }
    ]
  })
}

resource "aws_s3_bucket_replication_configuration" "videos_to_uae" {
  # Depends on versioning being enabled on source
  depends_on = [aws_s3_bucket_versioning.videos]

  bucket = aws_s3_bucket.videos.id
  role   = aws_iam_role.s3_replication.arn

  rule {
    id     = "replicate-all-to-uae"
    status = "Enabled"

    filter {} # Replicate all objects

    destination {
      bucket        = aws_s3_bucket.videos_uae.arn
      storage_class = "STANDARD_IA"

      encryption_configuration {
        replica_kms_key_id = "aws/s3" # UAE region default KMS key
      }
    }

    delete_marker_replication { status = "Enabled" }
  }
}

# ─── UAE WAF Geo-Fence (mirrors Singapore config) ─────────────

resource "aws_wafv2_web_acl" "geo_fence_uae" {
  provider = aws.uae
  name     = "${var.app_name}-geo-fence-uae"
  scope    = "REGIONAL"

  default_action { allow {} }

  rule {
    name     = "block-cn-hk"
    priority = 1
    action { block {} }

    statement {
      geo_match_statement {
        country_codes = ["CN", "HK"]
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${var.app_name}-geo-block-uae"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${var.app_name}-waf-uae"
    sampled_requests_enabled   = true
  }
}

# ─── UAE Outputs ──────────────────────────────────────────────

output "uae_s3_bucket" {
  description = "UAE secondary video bucket (DR replication target)"
  value       = aws_s3_bucket.videos_uae.id
}

output "uae_waf_acl_id" {
  description = "UAE WAF Web ACL ID (geo-fence)"
  value       = aws_wafv2_web_acl.geo_fence_uae.id
}
