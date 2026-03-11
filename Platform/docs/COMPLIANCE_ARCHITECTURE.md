# RoboMemo 数据合规架构设计文档

> Version: 1.0 | Date: 2026-03-10 | Author: Lucy (OpenClaw Agent)
> Status: DRAFT — 待 Vince 审核

---

## 1. 背景与目标

RoboMemo Platform 为 Roboforce 提供机器人视频结构化 VQA 标注服务。客户明确要求：

> **数据和标签全程不可接触到中国（包括香港）**：数据采集员、数据库、服务器均部署于新加坡、阿联酋等海外节点，全程与中国无关。

需要满足的合规标准：
- **ISO 27001** — 信息安全管理体系
- **ISO 27701** — 隐私信息管理体系（PIMS）
- **GDPR** — 欧盟通用数据保护条例
- **SOC 2 Type II** — 服务组织控制（安全性、可用性、处理完整性、机密性）

---

## 2. 威胁模型

### 2.1 数据触及中国/香港的风险点

| # | 风险点 | 当前状态 | 严重性 |
|---|--------|---------|--------|
| T1 | 后端服务器部署在中国 | ❌ 当前 localhost 开发 | Critical |
| T2 | SQLite 数据库文件存放在中国节点 | ❌ 本地文件 | Critical |
| T3 | VLM API 请求经过中国 CDN/代理 | ⚠️ 未控制 | High |
| T4 | 视频文件上传/传输经过中国网络 | ⚠️ 未控制 | High |
| T5 | 开发人员从中国 IP 直接访问生产数据 | ⚠️ 无访问控制 | High |
| T6 | DNS 解析走中国 DNS 服务器 | ⚠️ 未控制 | Medium |
| T7 | 日志/监控数据回传中国 | ⚠️ 无审计日志 | Medium |
| T8 | 备份数据存储在中国云 | ⚠️ 无备份策略 | High |

### 2.2 合规差距分析

| 标准 | 要求 | 当前状态 | 差距 |
|------|------|---------|------|
| ISO 27001 A.8 | 资产管理、数据分类 | 无 | 需要数据分类策略 |
| ISO 27001 A.9 | 访问控制 | 无认证 | 需要 RBAC + MFA |
| ISO 27001 A.10 | 加密 | 无 TLS | 需要传输+静态加密 |
| ISO 27001 A.12 | 运维安全 | 无日志 | 需要审计日志 |
| ISO 27701 7.2 | 数据处理目的限制 | 无隐私策略 | 需要数据处理协议 |
| ISO 27701 7.3 | 数据最小化 | 未实施 | 需要数据保留策略 |
| GDPR Art.17 | 删除权（被遗忘权） | 未实现 | 需要数据删除 API |
| GDPR Art.20 | 数据可携带权 | 仅 JSON 导出 | 需要标准化导出 |
| GDPR Art.33 | 数据泄露通知 | 无机制 | 需要事件响应流程 |
| SOC 2 CC6 | 逻辑与物理访问控制 | 无 | 需要全面访问控制 |
| SOC 2 CC7 | 系统运维 | 无监控 | 需要安全监控 |
| SOC 2 CC8 | 变更管理 | Git 仅 2 commit | 需要正式变更流程 |

---

## 3. 目标架构

### 3.1 部署拓扑

```
                    ┌──────────────────────────────────┐
                    │       GEO-FENCE BOUNDARY         │
                    │  (Singapore / UAE / EU only)      │
                    │                                    │
  ┌─────────┐      │  ┌───────────┐   ┌─────────────┐  │
  │ Overseas │ TLS  │  │  NGINX    │   │  App Server  │  │
  │ Operator │─────────│  WAF +    │──▶│  (Node.js)   │  │
  │ Browser  │      │  │  GeoIP    │   │  + API       │  │
  └─────────┘      │  └───────────┘   └──────┬──────┘  │
                    │                          │         │
                    │         ┌────────────────┼───┐     │
                    │         │                ▼   │     │
                    │         │  ┌─────────────┐   │     │
                    │         │  │ PostgreSQL   │   │     │
                    │         │  │ (encrypted)  │   │     │
                    │         │  └─────────────┘   │     │
                    │         │                     │     │
                    │         │  ┌─────────────┐   │     │
                    │         │  │ Object Store │   │     │
                    │         │  │ (S3/MinIO)   │   │     │
                    │         │  │ SSE-KMS      │   │     │
                    │         │  └─────────────┘   │     │
                    │         │    Private VPC      │     │
                    │         └─────────────────────┘     │
                    │                  │                   │
                    │                  ▼                   │
                    │  ┌───────────────────────────────┐  │
                    │  │   VLM API Proxy (Singapore)   │  │
                    │  │   ┌─────┐ ┌──────┐ ┌──────┐  │  │
                    │  │   │Gemini│ │Claude│ │GPT-4o│  │  │
                    │  │   └─────┘ └──────┘ └──────┘  │  │
                    │  └───────────────────────────────┘  │
                    │                                      │
                    └──────────────────────────────────────┘

        ❌ China / Hong Kong IP — BLOCKED at WAF layer
```

### 3.2 数据流

```
Video Upload                VLM Analysis              Storage
    │                           │                        │
    ▼                           ▼                        ▼
[Operator Browser]        [VLM API Proxy]          [PostgreSQL]
    │ TLS 1.3                   │ TLS 1.3               │ AES-256
    ▼                           ▼                        ▼
[NGINX WAF]               [Gemini/Claude/GPT]      [S3 SSE-KMS]
    │ GeoIP Check               │                        │
    │ ❌ CN/HK blocked          │ ✅ SG/UAE egress       │ ✅ SG/UAE region
    ▼                           ▼                        ▼
[App Server]              [Structured JSON]        [Encrypted at rest]
    │                           │                        │
    ▼                           ▼                        ▼
[Audit Log]               [Grounding Check]        [Backup (same region)]
```

---

## 4. 组件设计

### 4.1 地理围栏层 (Geo-Fence)

#### 4.1.1 NGINX WAF + GeoIP

```nginx
# /etc/nginx/conf.d/geo-fence.conf

# GeoIP2 模块 — MaxMind 数据库
geoip2 /etc/nginx/GeoIP/GeoLite2-Country.mmdb {
    auto_reload 24h;
    $geoip2_country_code source=$remote_addr country iso_code;
}

# 阻断中国和香港
map $geoip2_country_code $blocked_country {
    CN      1;
    HK      1;
    default 0;
}

server {
    listen 443 ssl http2;
    server_name api.robomemo.io;

    # TLS 1.3 only
    ssl_protocols TLSv1.3;
    ssl_certificate     /etc/ssl/certs/robomemo.pem;
    ssl_certificate_key /etc/ssl/private/robomemo.key;

    # Geo-fence enforcement
    if ($blocked_country) {
        return 403 '{"error": "Access denied from this region"}';
    }

    # Forward to app
    location / {
        proxy_pass http://127.0.0.1:3001;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Geo-Country $geoip2_country_code;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

#### 4.1.2 应用层二次校验

```typescript
// middleware/geo-fence.ts
export function geoFenceMiddleware(req: Request, res: Response, next: NextFunction) {
  const country = req.headers['x-geo-country'] as string;
  const clientIp = req.headers['x-real-ip'] as string;

  const BLOCKED_REGIONS = ['CN', 'HK'];

  if (BLOCKED_REGIONS.includes(country?.toUpperCase())) {
    auditLog.write({
      event: 'GEO_FENCE_BLOCK',
      ip: clientIp,
      country,
      path: req.path,
      timestamp: new Date().toISOString(),
    });
    return res.status(403).json({
      error: 'Access denied',
      reason: 'GEOGRAPHIC_RESTRICTION',
    });
  }

  next();
}
```

### 4.2 VLM API 代理 (Outbound Geo-Control)

VLM API 调用必须从海外节点发出，不能经过中国网络。

```typescript
// services/vlm-proxy.ts

const ALLOWED_EGRESS_REGIONS = ['ap-southeast-1', 'me-south-1']; // SG, UAE

interface VLMProxyConfig {
  provider: string;
  apiKey: string;
  // Force outbound request through specified region's proxy
  egressRegion: string;
}

class VLMProxy {
  private proxyUrl: string;

  constructor(config: VLMProxyConfig) {
    // Use region-specific proxy endpoint
    this.proxyUrl = `https://vlm-proxy.${config.egressRegion}.robomemo.io`;
  }

  async analyze(videoFrames: Buffer[], prompt: string): Promise<any> {
    // All VLM API calls go through regional proxy
    // Proxy server is deployed in SG/UAE with no China routing
    const response = await fetch(`${this.proxyUrl}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Provider-Key': this.apiKey, // encrypted in transit
      },
      body: JSON.stringify({ frames: videoFrames, prompt }),
    });

    return response.json();
  }
}
```

### 4.3 数据加密

#### 4.3.1 传输加密 (TLS 1.3)

- 所有 HTTP 通信强制 HTTPS，TLS 1.3
- 内部服务间通信使用 mTLS
- VLM API 调用使用 TLS（由云提供商保证）

#### 4.3.2 静态加密 (AES-256)

```typescript
// services/encryption.ts
import crypto from 'crypto';

const ALGORITHM = 'aes-256-gcm';

export class DataEncryption {
  private key: Buffer;

  constructor(masterKeyHex: string) {
    this.key = Buffer.from(masterKeyHex, 'hex');
  }

  encrypt(plaintext: string): { ciphertext: string; iv: string; tag: string } {
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipheriv(ALGORITHM, this.key, iv);
    let encrypted = cipher.update(plaintext, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    return {
      ciphertext: encrypted,
      iv: iv.toString('hex'),
      tag: cipher.getAuthTag().toString('hex'),
    };
  }

  decrypt(ciphertext: string, ivHex: string, tagHex: string): string {
    const iv = Buffer.from(ivHex, 'hex');
    const tag = Buffer.from(tagHex, 'hex');
    const decipher = crypto.createDecipheriv(ALGORITHM, this.key, iv);
    decipher.setAuthTag(tag);
    let decrypted = decipher.update(ciphertext, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    return decrypted;
  }
}
```

#### 4.3.3 数据库加密

**从 SQLite 迁移到 PostgreSQL**（生产环境）：

```yaml
# docker-compose.yml (Singapore deployment)
services:
  db:
    image: postgres:16
    environment:
      POSTGRES_DB: robomemo
      POSTGRES_USER: robomemo_app
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    volumes:
      - pgdata:/var/lib/postgresql/data  # 加密文件系统 (LUKS)
    command:
      - "-c" 
      - "ssl=on"
      - "-c"
      - "ssl_cert_file=/etc/ssl/server.crt"
      - "-c"
      - "ssl_key_file=/etc/ssl/server.key"
    deploy:
      placement:
        constraints:
          - node.labels.region == ap-southeast-1  # Singapore only

  object-store:
    image: minio/minio
    environment:
      MINIO_KMS_KES_ENDPOINT: https://kes:7373
      MINIO_KMS_KES_KEY_NAME: robomemo-master-key
    command: server /data --console-address ":9001"
    volumes:
      - videodata:/data  # SSE-KMS encrypted
```

### 4.4 访问控制 (RBAC + MFA)

#### 4.4.1 角色定义

| 角色 | 权限 | 数据访问 |
|------|------|---------|
| `annotator` | 查看分配的视频、提交标注 | 仅分配的 episode |
| `reviewer` | 审核标注、批准/驳回 | 所有标注 |
| `data_admin` | 管理数据集、导入/导出 | 所有数据 |
| `platform_admin` | 系统配置、用户管理 | 全部 |
| `auditor` | 只读审计日志 | 审计日志 |

#### 4.4.2 认证方案

```typescript
// middleware/auth.ts
import jwt from 'jsonwebtoken';

interface TokenPayload {
  userId: string;
  role: 'annotator' | 'reviewer' | 'data_admin' | 'platform_admin' | 'auditor';
  region: string;       // 用户所在区域
  permissions: string[];
  mfaVerified: boolean;
}

export function authMiddleware(requiredRole: string[]) {
  return (req: Request, res: Response, next: NextFunction) => {
    const token = req.headers.authorization?.replace('Bearer ', '');
    if (!token) return res.status(401).json({ error: 'Authentication required' });

    try {
      const payload = jwt.verify(token, process.env.JWT_SECRET!) as TokenPayload;

      // Check MFA for sensitive operations
      if (['data_admin', 'platform_admin'].includes(payload.role) && !payload.mfaVerified) {
        return res.status(403).json({ error: 'MFA verification required' });
      }

      // Check role
      if (!requiredRole.includes(payload.role)) {
        return res.status(403).json({ error: 'Insufficient permissions' });
      }

      // Check region (no China/HK users)
      const BLOCKED_REGIONS = ['CN', 'HK'];
      if (BLOCKED_REGIONS.includes(payload.region)) {
        auditLog.write({
          event: 'AUTH_REGION_BLOCK',
          userId: payload.userId,
          region: payload.region,
        });
        return res.status(403).json({ error: 'Access denied from this region' });
      }

      req.user = payload;
      next();
    } catch (e) {
      return res.status(401).json({ error: 'Invalid token' });
    }
  };
}
```

### 4.5 审计日志

#### 4.5.1 日志架构

```typescript
// services/audit-log.ts

interface AuditEvent {
  id: string;
  timestamp: string;
  event: string;          // e.g. 'DATA_ACCESS', 'ANNOTATION_CREATE', 'EXPORT', 'GEO_FENCE_BLOCK'
  userId?: string;
  role?: string;
  ip: string;
  country?: string;
  resource?: string;      // e.g. 'dataset:ds_123', 'annotation:ann_456'
  action: string;         // 'read' | 'create' | 'update' | 'delete' | 'export'
  details?: Record<string, any>;
  result: 'success' | 'denied' | 'error';
}

class AuditLogger {
  // Write to append-only table (no UPDATE/DELETE permissions)
  async write(event: Partial<AuditEvent>): Promise<void> {
    const record: AuditEvent = {
      id: `audit_${Date.now()}_${crypto.randomUUID()}`,
      timestamp: new Date().toISOString(),
      result: 'success',
      ...event,
    } as AuditEvent;

    // PostgreSQL append-only table
    await db.query(
      `INSERT INTO audit_log (id, timestamp, event, user_id, role, ip, country, 
       resource, action, details, result) 
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)`,
      [record.id, record.timestamp, record.event, record.userId, record.role,
       record.ip, record.country, record.resource, record.action,
       JSON.stringify(record.details), record.result]
    );
  }

  // Query for compliance reporting
  async query(filters: {
    startDate?: string;
    endDate?: string;
    event?: string;
    userId?: string;
    result?: string;
  }): Promise<AuditEvent[]> {
    // ... parameterized query
  }
}

export const auditLog = new AuditLogger();
```

#### 4.5.2 审计日志 SQL Schema

```sql
CREATE TABLE audit_log (
  id          TEXT PRIMARY KEY,
  timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  event       TEXT NOT NULL,
  user_id     TEXT,
  role        TEXT,
  ip          INET,
  country     TEXT,
  resource    TEXT,
  action      TEXT NOT NULL,
  details     JSONB,
  result      TEXT NOT NULL CHECK (result IN ('success', 'denied', 'error'))
);

-- Append-only: revoke UPDATE/DELETE from application role
REVOKE UPDATE, DELETE ON audit_log FROM robomemo_app;

-- Partitioned by month for retention
CREATE INDEX idx_audit_timestamp ON audit_log (timestamp);
CREATE INDEX idx_audit_event ON audit_log (event);
CREATE INDEX idx_audit_user ON audit_log (user_id);

-- 保留 3 年（ISO 27001 要求）
-- 使用 pg_partman 或 cron 自动清理超龄分区
```

### 4.6 GDPR 合规

#### 4.6.1 数据主体权利 API

```typescript
// routes/gdpr.ts

// Art.15 — 数据主体访问权
router.get('/gdpr/access/:subjectId', auth(['data_admin', 'platform_admin']), async (req, res) => {
  const data = await collectSubjectData(req.params.subjectId);
  auditLog.write({ event: 'GDPR_ACCESS_REQUEST', resource: `subject:${req.params.subjectId}` });
  res.json(data);
});

// Art.17 — 删除权（被遗忘权）
router.delete('/gdpr/erasure/:subjectId', auth(['platform_admin']), async (req, res) => {
  // 1. 删除所有关联数据
  await eraseSubjectData(req.params.subjectId);
  // 2. 通知所有第三方处理者
  await notifyProcessors(req.params.subjectId, 'erasure');
  // 3. 记录审计
  auditLog.write({ event: 'GDPR_ERASURE', resource: `subject:${req.params.subjectId}` });
  res.json({ success: true, message: 'All subject data erased' });
});

// Art.20 — 数据可携带权
router.get('/gdpr/portability/:subjectId', auth(['data_admin', 'platform_admin']), async (req, res) => {
  const exportData = await exportSubjectData(req.params.subjectId, 'json');
  auditLog.write({ event: 'GDPR_PORTABILITY', resource: `subject:${req.params.subjectId}` });
  res.attachment(`subject_${req.params.subjectId}_data.json`);
  res.json(exportData);
});
```

#### 4.6.2 数据处理记录 (Art.30)

```typescript
// 维护处理活动记录
const PROCESSING_ACTIVITIES = [
  {
    activity: 'Robot Video VQA Annotation',
    purpose: 'Generate structured training labels for robotics AI',
    legalBasis: 'Legitimate interest / Contract',
    dataCategories: ['robot_video', 'sensor_data', 'vqa_annotations'],
    recipients: ['VLM providers (Gemini/Claude/GPT-4o)', 'Annotation reviewers'],
    retention: '5 years from collection',
    transfers: {
      regions: ['ap-southeast-1 (Singapore)', 'me-south-1 (UAE)'],
      mechanism: 'Standard Contractual Clauses (SCC)',
      excludedRegions: ['CN', 'HK'],
    },
    safeguards: ['TLS 1.3', 'AES-256 encryption', 'GeoIP fencing', 'RBAC'],
  },
];
```

### 4.7 数据血统追踪 (Data Lineage)

每个标注记录必须包含完整的数据处理链：

```typescript
interface DataLineage {
  id: string;
  // 数据来源
  source: {
    type: 'upload' | 'simulation' | 'transfer';
    origin: string;         // e.g. 'Singapore Lab Camera #3'
    collectorId: string;    // 采集员 ID
    collectorRegion: string; // 采集员所在区域
    timestamp: string;
  };
  // 处理链
  processing: Array<{
    step: string;           // e.g. 'frame_extraction', 'vlm_analysis', 'manual_review'
    processor: string;      // e.g. 'gemini-2.5-pro', 'annotator:user_123'
    region: string;         // 处理发生的区域
    timestamp: string;
    inputHash: string;      // SHA-256 of input
    outputHash: string;     // SHA-256 of output
  }>;
  // 存储位置
  storage: {
    region: string;
    encrypted: boolean;
    encryptionMethod: string;
  };
}
```

---

## 5. 部署方案

### 5.1 推荐云方案

| 方案 | 提供商 | 区域 | 优势 | 劣势 |
|------|--------|------|------|------|
| **方案 A（推荐）** | AWS | ap-southeast-1 (Singapore) | 最成熟、ISO/SOC 认证齐全 | 成本较高 |
| 方案 B | Azure | UAE North | 距离中东客户近 | 服务种类少于 AWS |
| 方案 C | GCP | asia-southeast1 (Singapore) | Gemini API 原生集成 | SOC 2 认证流程较长 |

### 5.2 AWS Singapore 部署架构

```
AWS ap-southeast-1 (Singapore)
├── VPC (10.0.0.0/16)
│   ├── Public Subnet
│   │   ├── ALB + WAF (GeoIP blocking)
│   │   └── NAT Gateway (outbound VLM API)
│   ├── Private Subnet (App)
│   │   ├── ECS Fargate (Node.js app)
│   │   └── ECS Fargate (VLM proxy)
│   └── Private Subnet (Data)
│       ├── RDS PostgreSQL (Multi-AZ, encrypted)
│       └── S3 (SSE-KMS, bucket policy restricts region)
├── KMS (Customer Managed Key)
├── CloudTrail (audit logging)
├── GuardDuty (threat detection)
├── Config (compliance monitoring)
└── Secrets Manager (API keys, DB credentials)
```

### 5.3 S3 Bucket Policy (地理限制)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyAccessFromBlockedRegions",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:*",
      "Resource": [
        "arn:aws:s3:::robomemo-videos/*",
        "arn:aws:s3:::robomemo-annotations/*"
      ],
      "Condition": {
        "NotIpAddress": {
          "aws:SourceIp": [
            "SINGAPORE_VPC_CIDR",
            "UAE_VPC_CIDR"
          ]
        }
      }
    }
  ]
}
```

### 5.4 基础设施即代码 (Terraform 框架)

```hcl
# main.tf
provider "aws" {
  region = "ap-southeast-1"  # Singapore
}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "robomemo-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["ap-southeast-1a", "ap-southeast-1b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = false  # HA
}

resource "aws_wafv2_web_acl" "geo_fence" {
  name  = "robomemo-geo-fence"
  scope = "REGIONAL"

  default_action { allow {} }

  rule {
    name     = "block-china-hk"
    priority = 1
    action { block {} }

    statement {
      geo_match_statement {
        country_codes = ["CN", "HK"]
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "BlockedGeoRequests"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "RoboMemoWAF"
    sampled_requests_enabled   = true
  }
}

resource "aws_rds_cluster" "db" {
  cluster_identifier = "robomemo-db"
  engine             = "aurora-postgresql"
  engine_version     = "16.1"
  master_username    = "robomemo_app"
  master_password    = var.db_password

  storage_encrypted   = true
  kms_key_id         = aws_kms_key.robomemo.arn

  vpc_security_group_ids = [aws_security_group.db.id]
  db_subnet_group_name   = aws_db_subnet_group.private.name

  backup_retention_period = 35  # 5 weeks
  preferred_backup_window = "03:00-04:00"
}

resource "aws_kms_key" "robomemo" {
  description             = "RoboMemo data encryption key"
  deletion_window_in_days = 30
  enable_key_rotation     = true
}
```

---

## 6. 认证路线图

### 6.1 ISO 27001/27701 认证

| 阶段 | 任务 | 时间 | 交付物 |
|------|------|------|--------|
| Phase 1 | 差距分析 + 风险评估 | 2 周 | 风险登记册、适用性声明 (SoA) |
| Phase 2 | 策略和程序编写 | 4 周 | ISMS 策略文档集 |
| Phase 3 | 控制措施实施 | 8 周 | 技术控制 + 运营流程 |
| Phase 4 | 内部审计 | 2 周 | 内审报告 |
| Phase 5 | 管理评审 | 1 周 | 评审记录 |
| Phase 6 | 外部认证审计 | 2 周 | ISO 证书 |
| **总计** | | **~19 周** | |

### 6.2 SOC 2 Type II

| 阶段 | 任务 | 时间 |
|------|------|------|
| 准备期 | 定义信托服务标准范围 | 2 周 |
| 观察期 | 运行控制措施并收集证据 | **6-12 个月** |
| 审计期 | 外部审计 | 4-6 周 |

> ⚠️ SOC 2 Type II 需要 6-12 个月的运行证据，建议尽早启动。

### 6.3 GDPR 合规

| 任务 | 时间 |
|------|------|
| 数据保护影响评估 (DPIA) | 2 周 |
| 隐私通知 + 数据处理协议 (DPA) | 1 周 |
| 数据主体权利 API 实现 | 2 周 |
| 指定数据保护官 (DPO) | 立即 |
| Cookie/同意管理 | 1 周 |

---

## 7. 实施优先级

### Phase 1 — MVP 合规 (4 周)

**必须在客户交付前完成：**

1. ✅ 海外服务器部署（AWS Singapore）
2. ✅ GeoIP 地理围栏（WAF + 应用层）
3. ✅ TLS 1.3 传输加密
4. ✅ 数据库静态加密（RDS encrypted）
5. ✅ 基本审计日志
6. ✅ VLM API 代理（确保请求从 SG 发出）
7. ✅ 基本 RBAC（annotator / admin）

### Phase 2 — 合规加固 (8 周)

8. MFA 认证
9. 数据血统追踪
10. GDPR 数据主体权利 API
11. 完整审计日志 + 告警
12. 备份加密 + 跨区域容灾（但仅限允许区域）
13. 渗透测试

### Phase 3 — 认证 (12-20 周)

14. ISO 27001/27701 认证流程
15. SOC 2 Type II 观察期启动
16. 第三方安全审计

---

## 8. 成本估算 (AWS Singapore)

| 资源 | 规格 | 月成本 (USD) |
|------|------|-------------|
| ECS Fargate (App) | 2 vCPU, 4GB × 2 | ~$140 |
| ECS Fargate (VLM Proxy) | 1 vCPU, 2GB × 2 | ~$70 |
| RDS Aurora PostgreSQL | db.r6g.large, Multi-AZ | ~$400 |
| S3 (视频+标注) | 500GB | ~$12 |
| ALB + WAF | Standard | ~$50 |
| KMS | 1 key + API calls | ~$5 |
| CloudTrail + GuardDuty | Standard | ~$30 |
| NAT Gateway | 2 AZ | ~$90 |
| **合计** | | **~$800/月** |

> 注：不含 VLM API 调用费用（Gemini/Claude/GPT-4o 按用量计费）。
> ISO 认证审计费用约 $15,000-30,000（一次性）。

---

## 9. 开放问题

| # | 问题 | 决策者 | 状态 |
|---|------|--------|------|
| Q1 | 选择 AWS / Azure / GCP？ | Vince | 待决 |
| Q2 | ISO 认证是否由第三方咨询公司协助？ | Vince | 待决 |
| Q3 | 开发环境是否也需要合规？（当前在 Mac mini 开发） | Vince | 待决 |
| Q4 | 数据保留期限？（建议 5 年） | Vince + 法务 | 待决 |
| Q5 | 是否需要 UAE 作为第二部署区域？ | Vince + 客户 | 待决 |
| Q6 | DPO（数据保护官）由谁担任？ | Vince | 待决 |

---

## 附录 A — 合规检查清单

### ISO 27001 Annex A 控制措施映射

| 控制项 | 描述 | 实施方案 | 状态 |
|--------|------|---------|------|
| A.5.1 | 信息安全策略 | ISMS 策略文档 | ⬜ |
| A.6.1 | 内部组织 | 角色和职责定义 | ⬜ |
| A.7.1 | 人力资源安全 | 背景调查、保密协议 | ⬜ |
| A.8.1 | 资产管理 | 数据分类和标签 | ⬜ |
| A.8.2 | 信息分类 | 四级分类（公开/内部/机密/绝密） | ⬜ |
| A.9.1 | 访问控制 | RBAC + MFA | ⬜ |
| A.9.4 | 系统访问控制 | JWT + 会话管理 | ⬜ |
| A.10.1 | 加密 | TLS 1.3 + AES-256 | ⬜ |
| A.12.4 | 日志和监控 | 审计日志 + CloudWatch | ⬜ |
| A.13.1 | 网络安全 | VPC + 安全组 + WAF | ⬜ |
| A.14.1 | 系统安全开发 | CI/CD + 代码审查 | ⬜ |
| A.16.1 | 事件管理 | 事件响应计划 | ⬜ |
| A.17.1 | 业务连续性 | 多 AZ 部署 + 备份 | ⬜ |
| A.18.1 | 合规性 | 定期合规审查 | ⬜ |

---

*本文档将随项目进展持续更新。下次审查日期：2026-03-24。*
