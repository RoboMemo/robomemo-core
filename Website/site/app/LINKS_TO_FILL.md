# RoboMemo 网站 - 空白链接/按钮清单

以下是网站中所有需要填充的链接和按钮，请告诉我每个应该指向哪里。

---

## 1. Footer 社交链接

**文件**: `src/components/Footer.tsx`

| 图标 | 当前链接 | 用途 | 建议填写 |
|------|----------|------|----------|
| GitHub | `#` | GitHub仓库 | 例如: `https://github.com/robomemo` |
| Twitter/X | `#` | 官方Twitter | 例如: `https://twitter.com/robomemo` |
| LinkedIn | `#` | 公司LinkedIn | 例如: `https://linkedin.com/company/robomemo` |
| Email | `mailto:contact@robomemo.ai` | 联系邮箱 | ✅ 已设置 |

---

## 2. Hero 区域按钮

**文件**: `src/sections/home/Hero.tsx`

| 按钮文字 | 当前状态 | 建议行为 |
|----------|----------|----------|
| "Explore Solutions" / "探索解决方案" | 无功能 | 建议: 滚动到Products页面或Products区块 |
| "Full Video" | 无功能 | 建议: 弹出视频模态框或跳转到YouTube/Bilibili视频 |

---

## 3. DaaS 产品区块按钮

**文件**: `src/sections/products/DaasSection.tsx`

| 按钮文字 | 当前状态 | 建议行为 |
|----------|----------|----------|
| "Browse Datasets" | 无功能 | 建议: 跳转到数据集页面或打开数据集列表 |

---

## 4. SaaS 平台区块按钮

**文件**: `src/sections/products/SaasSection.tsx`

| 按钮文字 | 当前状态 | 建议行为 |
|----------|----------|----------|
| "Try Platform" | 无功能 | 建议: 跳转到SaaS平台登录页或演示页面 |

---

## 5. Robot School 托管服务按钮

**文件**: `src/sections/products/RobotSchoolSection.tsx`

| 按钮文字 | 当前状态 | 建议行为 |
|----------|----------|----------|
| "Get Started" | 无功能 | 建议: 跳转到联系表单或预约演示页面 |

---

## 6. End-Effector 末端执行器按钮

**文件**: `src/sections/products/EndEffectorSection.tsx`

| 按钮文字 | 当前状态 | 建议行为 |
|----------|----------|----------|
| "Learn More" | 无功能 | 建议: 跳转到产品详情页或技术规格PDF |

---

## 7. Footer 导航链接

**文件**: `src/components/Footer.tsx`

当前Footer中的导航链接使用的是锚点形式(`#home`, `#dataset`等)，但由于是单页应用，这些需要改为页面切换。

| 链接文字 | 当前链接 | 建议行为 |
|----------|----------|----------|
| Home / 首页 | `#home` | 切换到首页 |
| Dataset / 数据集 | `#dataset` | 切换到数据集页 |
| Products / 产品 | `#products` | 切换到产品页 |
| About / 关于 | `#about` | 建议: 添加About页面或滚动到首页底部 |

---

## 快速回复格式

您可以直接回复如下格式告诉我：

```
GitHub: https://github.com/yourorg
Twitter: https://twitter.com/yourhandle
LinkedIn: https://linkedin.com/company/yourcompany

Hero - Explore Solutions: 滚动到Products页面
Hero - Full Video: https://youtube.com/watch?v=xxx

DaaS - Browse Datasets: 跳转到Dataset页面
SaaS - Try Platform: https://platform.robomemo.ai
Robot School - Get Started: 弹出联系表单
End-Effector - Learn More: 跳转到产品详情页
```

或者告诉我您希望每个按钮/链接做什么，我来帮您实现！
