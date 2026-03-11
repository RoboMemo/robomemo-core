# RoboMemo 网站技术规范

---

## 1. 组件清单

### shadcn/ui 组件
| 组件 | 用途 | 安装命令 |
|------|------|----------|
| Button | CTA按钮、导航按钮 | `npx shadcn add button` |
| Card | 价值卡片、产品卡片 | `npx shadcn add card` |
| Badge | 标签、状态指示 | `npx shadcn add badge` |
| Tabs | 数据集分类、产品切换 | `npx shadcn add tabs` |
| Dialog | 图片灯箱、详情弹窗 | `npx shadcn add dialog` |
| Sheet | 移动端导航菜单 | `npx shadcn add sheet` |
| Separator | 分割线 | `npx shadcn add separator` |
| ScrollArea | 滚动区域 | `npx shadcn add scroll-area` |

### 自定义组件
| 组件 | 用途 | 位置 |
|------|------|------|
| Navigation | 顶部导航栏 | `components/Navigation.tsx` |
| LanguageSwitcher | 语言切换 | `components/LanguageSwitcher.tsx` |
| VideoBackground | Hero视频背景 | `components/VideoBackground.tsx` |
| AnimatedCounter | 数字计数动画 | `components/AnimatedCounter.tsx` |
| ScrollReveal | 滚动显示包装器 | `components/ScrollReveal.tsx` |
| FlywheelDiagram | 飞轮图表 | `components/FlywheelDiagram.tsx` |
| RobotFormCard | 机器人形态卡片 | `components/RobotFormCard.tsx` |
| ProductShowcase | 产品展示 | `components/ProductShowcase.tsx` |
| DatasetGallery | 数据集画廊 | `components/DatasetGallery.tsx` |
| Footer | 页脚 | `components/Footer.tsx` |

---

## 2. 动画实现规划

| 动画 | 库 | 实现方式 | 复杂度 |
|------|-----|----------|--------|
| 页面加载序列 | Framer Motion | AnimatePresence + stagger | Medium |
| 滚动触发显示 | Framer Motion | useInView + motion.div | Low |
| Hero文字动画 | Framer Motion | variants + staggerChildren | Medium |
| 按钮悬停效果 | CSS/Tailwind | hover:scale + transition | Low |
| 卡片悬停效果 | CSS/Tailwind | hover:translateY + border | Low |
| 导航背景渐变 | CSS + React | scroll listener + opacity | Low |
| 数字计数动画 | Framer Motion | useSpring + animate | Medium |
| 飞轮旋转动画 | Framer Motion | rotate animation loop | Medium |
| 图片悬停缩放 | CSS/Tailwind | hover:scale + overflow-hidden | Low |
| 链接下划线动画 | CSS | pseudo-element + width | Low |
| 语言切换过渡 | Framer Motion | AnimatePresence | Low |
| 移动端菜单 | Framer Motion + Sheet | slide-in animation | Medium |

---

## 3. 动画库选择

### 主要库: Framer Motion
- 用于: 页面加载、滚动触发、复杂序列动画
- 原因: React原生支持，声明式API，性能优秀

### 辅助: CSS/Tailwind
- 用于: 简单悬停效果、过渡动画
- 原因: 轻量，无需JS，性能最佳

### 安装命令
```bash
npm install framer-motion
```

---

## 4. 项目文件结构

```
/mnt/okcomputer/output/app/
├── public/
│   ├── videos/
│   │   └── hero-bg.mp4
│   └── images/
│       ├── hero-robot.jpg
│       ├── robot-forms/
│       ├── dataset/
│       ├── products/
│       └── icons/
├── src/
│   ├── components/
│   │   ├── ui/              # shadcn组件
│   │   ├── Navigation.tsx
│   │   ├── Footer.tsx
│   │   ├── LanguageSwitcher.tsx
│   │   ├── VideoBackground.tsx
│   │   ├── ScrollReveal.tsx
│   │   ├── AnimatedCounter.tsx
│   │   ├── FlywheelDiagram.tsx
│   │   ├── RobotFormCard.tsx
│   │   ├── ProductShowcase.tsx
│   │   └── DatasetGallery.tsx
│   ├── sections/
│   │   ├── home/
│   │   │   ├── Hero.tsx
│   │   │   ├── ValueProposition.tsx
│   │   │   ├── TaskFocus.tsx
│   │   │   ├── GrowthCurve.tsx
│   │   │   ├── MultiFormSupport.tsx
│   │   │   ├── ScaleAIModel.tsx
│   │   │   └── LanxiangAnalogy.tsx
│   │   ├── dataset/
│   │   │   ├── DatasetHero.tsx
│   │   │   ├── DatasetGallery.tsx
│   │   │   └── DatasetFeatures.tsx
│   │   └── products/
│   │       ├── ProductsHero.tsx
│   │       ├── DaasSection.tsx
│   │       ├── SaasSection.tsx
│   │       ├── RobotSchoolSection.tsx
│   │       └── EndEffectorSection.tsx
│   ├── context/
│   │   └── LanguageContext.tsx
│   ├── lib/
│   │   ├── utils.ts
│   │   └── content.ts       # 中英文内容
│   ├── hooks/
│   │   └── useScrollPosition.ts
│   ├── pages/
│   │   ├── Home.tsx
│   │   ├── Dataset.tsx
│   │   └── Products.tsx
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── index.html
├── tailwind.config.js
├── vite.config.ts
└── package.json
```

---

## 5. 依赖清单

### 核心依赖 (已包含)
- React 18+
- TypeScript
- Vite
- Tailwind CSS
- shadcn/ui

### 额外依赖
```bash
# 动画
npm install framer-motion

# 图标
npm install lucide-react

# 路由
npm install react-router-dom

# 工具
npm install clsx tailwind-merge
```

---

## 6. 路由结构

| 路径 | 页面 | 说明 |
|------|------|------|
| `/` | Home | 默认重定向到 `/en` |
| `/en` | Home (English) | 英文首页 |
| `/zh` | Home (Chinese) | 中文首页 |
| `/en/dataset` | Dataset (English) | 英文数据集页 |
| `/zh/dataset` | Dataset (Chinese) | 中文数据集页 |
| `/en/products` | Products (English) | 英文产品页 |
| `/zh/products` | Products (Chinese) | 中文产品页 |

---

## 7. 内容数据结构

```typescript
// lib/content.ts
export interface Content {
  nav: {
    home: string;
    dataset: string;
    products: string;
    about: string;
  };
  hero: {
    title: string;
    subtitle: string;
    cta: string;
  };
  // ... 其他区块
}

export const content: Record<'en' | 'zh', Content> = {
  en: { /* 英文内容 */ },
  zh: { /* 中文内容 */ }
};
```

---

## 8. 性能优化

### 图片优化
- 使用WebP格式
- 懒加载非首屏图片
- 响应式图片srcset

### 动画优化
- 使用 `transform` 和 `opacity`
- 添加 `will-change` 提示
- 使用 `useReducedMotion` 支持

### 代码优化
- 路由懒加载
- 组件按需加载
- Tree shaking

---

## 9. 构建配置

### vite.config.ts
```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
});
```

### tailwind.config.js
```javascript
module.exports = {
  darkMode: 'class',
  content: ['./src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        background: '#000000',
        foreground: '#FFFFFF',
        accent: {
          blue: '#3B82F6',
          purple: '#8B5CF6',
        },
      },
    },
  },
  plugins: [],
};
```

---

## 10. 开发流程

1. **初始化项目**
   ```bash
   bash /app/.kimi/skills/webapp-building/scripts/init-webapp.sh "RoboMemo"
   ```

2. **安装依赖**
   ```bash
   cd /mnt/okcomputer/output/app
   npm install framer-motion lucide-react react-router-dom
   ```

3. **安装shadcn组件**
   ```bash
   npx shadcn add button card badge tabs dialog sheet separator scroll-area
   ```

4. **创建目录结构**
   ```bash
   mkdir -p src/components/ui src/sections/home src/sections/dataset src/sections/products src/context src/hooks src/pages public/videos public/images
   ```

5. **开发组件和页面**
   - 按依赖顺序创建组件
   - 从底层组件开始（ScrollReveal, AnimatedCounter）
   - 然后创建区块组件
   - 最后组装页面

6. **构建和测试**
   ```bash
   npm run build
   ```

7. **部署**
   ```bash
   # 使用deploy工具部署dist目录
   ```
