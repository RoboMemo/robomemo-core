import { createContext, useContext, useState, type ReactNode } from 'react';

type Language = 'en' | 'zh';

interface LanguageContextType {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: (key: string) => string;
}

const content: Record<Language, Record<string, string>> = {
  en: {
    // Navigation
    'nav.home': 'Home',
    'nav.dataset': 'Dataset',
    'nav.products': 'Products',
    'nav.about': 'About',
    
    // Hero
    'hero.title': 'Robot Memory Maker For GPT3 Moments of EmBodied AI',
    'hero.subtitle': 'Fuel for Humanoid Robots. No Shovels Required.',
    'hero.cta': 'Explore Solutions',
    
    // Value Proposition
    'value.title': 'Core Value Proposition',
    'value.fullbody.title': 'Full-Body Humanoid Motion',
    'value.fullbody.desc': 'Service industry focused - restaurants, homes. Trillion-dollar market opportunity.',
    'value.data.title': 'High-Efficiency Embodied Data',
    'value.data.desc': 'Professional self-labeling toolchain for embodied AI data. Closed-loop training with Verified Reward Model.',
    'value.cross.title': 'Cross-Embodiment Augmentation',
    'value.cross.desc': 'Solving data silos from robot heterogeneity and poor generalization across scenarios.',
    
    // Task Focus
    'task.title': 'Action-Level Industry Intersection',
    'task.desc': 'Focused on high-demand tasks like screws securing - the universal need across manufacturing and assembly.',
    
    // Growth Curve
    'growth.title': 'Data-Model-Skill Flywheel',
    'growth.desc': 'Closed-loop embodied "blue-collar" skill improvement. From pre-training to real robot validation.',
    'growth.step1': 'Data Collection',
    'growth.step2': 'Model Training',
    'growth.step3': 'Real Robot Validation',
    
    // Multi-Form
    'form.title': 'Serving All Robot Forms',
    'form.desc': 'From single-arm to dual-arm, wheeled to quadruped. Dimensionality reduction services for all.',
    'form.single': 'Single Arm',
    'form.dual': 'Dual Arm',
    'form.wheeled': 'Wheeled',
    'form.quadruped': 'Quadruped',
    
    // ScaleAI
    'scale.title': 'From Labor-Intensive to AI-Powered',
    'scale.traditional': 'Traditional: Manual labeling, high cost, slow iteration',
    'scale.new': 'New: Self-labeling model with Human Feedback. Tool + DaaS.',
    
    // Lanxiang
    'lanxiang.title': 'The Vocational Academy of Embodied AI',
    'lanxiang.step1': 'Generic Textbooks',
    'lanxiang.step2': 'Personalized Learning',
    'lanxiang.step3': 'Learning by Doing',
    'lanxiang.desc1': 'Labeled Data',
    'lanxiang.desc2': 'Online Learning',
    'lanxiang.desc3': 'Embodied AI Infrastructure',
    
    // Dataset Page
    'dataset.hero.title': 'AgiBot World Dataset',
    'dataset.hero.desc': 'High-quality embodied intelligence dataset for robotics research',
    'dataset.stats.tasks': 'Tasks',
    'dataset.stats.hours': 'Hours',
    'dataset.stats.samples': 'Samples',
    'dataset.stats.type': 'Type',
    'dataset.features.title': 'Dataset Features',
    'dataset.feature1': 'Real-world data',
    'dataset.feature2': 'Dual-arm manipulation',
    'dataset.feature3': 'Multi-modal sensors',
    'dataset.feature4': 'Anonymized & compressed',
    'dataset.feature5': 'Camera extrinsic corrected',
    
    // Products Page
    'products.hero.title': 'Our Products',
    'daas.title': 'DaaS - Data as a Service',
    'daas.desc': 'Pre-processed, high-quality embodied datasets for specific tasks. Including RLHF data packages.',
    'daas.feature1': 'Task-specific datasets',
    'daas.feature2': 'Pre-labeled and verified',
    'daas.feature3': 'Ready for training',
    
    'saas.title': 'SaaS Platform',
    'saas.desc': 'Self-service tools for automated labeling, data cleaning, and cross-embodiment augmentation.',
    'saas.feature1': 'Auto-labeling',
    'saas.feature2': 'Data cleaning',
    'saas.feature3': 'Cross-embodiment aug',
    
    'school.title': 'Robot School - Managed Service',
    'school.desc': 'End-to-end service for specific tasks and robot embodiments. Data collection, model training, and real robot validation.',
    'school.step1': 'Data Collection',
    'school.step2': 'Model Training',
    'school.step3': 'Deployment & Validation',
    
    'ee.title': 'Plug-in End-Effector + Skills',
    'ee.desc': 'Electric screwdriver end-effector with integrated sensors. Install and instantly gain fast screwing capabilities.',
    'ee.skill': 'Screws Securing Skill',
    
    // Footer
    'footer.tagline': 'Robot Memory Maker for Embodied AI',
    'footer.links': 'Quick Links',
    'footer.contact': 'Contact',
    'footer.copyright': '© 2025 RoboMemo. All rights reserved.',
  },
  zh: {
    // Navigation
    'nav.home': '首页',
    'nav.dataset': '数据集',
    'nav.products': '产品',
    'nav.about': '关于',
    
    // Hero
    'hero.title': '具身智能的机器人记忆制造者',
    'hero.subtitle': '为人形机器人供给能源，无需购买铲子',
    'hero.cta': '探索解决方案',
    
    // Value Proposition
    'value.title': '核心价值主张',
    'value.fullbody.title': '全身拟人动作',
    'value.fullbody.desc': '服务业聚焦 - 餐厅、家庭。万亿市值机遇。',
    'value.data.title': '高效高质具身数据',
    'value.data.desc': '专业自标注工具链，基于Verified Reward Model的闭环高效数据采集。',
    'value.cross.title': '跨本体/跨Context增广',
    'value.cross.desc': '解决机器人本体异构性造成的数据孤岛与场景泛化性问题。',
    
    // Task Focus
    'task.title': 'Action级别业界需求交集',
    'task.desc': '聚焦业界共同需求，如拧螺丝/螺母紧固 - 制造业与装配线的通用需求。',
    
    // Growth Curve
    'growth.title': '数据-模型-技能飞轮',
    'growth.desc': '闭环的具身"蓝领"技能提升。从预训练到实机验证。',
    'growth.step1': '数据采集',
    'growth.step2': '模型训练',
    'growth.step3': '实机验证',
    
    // Multi-Form
    'form.title': '服务多形态机器人',
    'form.desc': '从单臂到双臂，轮式到四足。为所有形态提供降维打击服务。',
    'form.single': '单臂',
    'form.dual': '双臂',
    'form.wheeled': '轮式',
    'form.quadruped': '四足',
    
    // ScaleAI
    'scale.title': '从劳动密集到AI驱动',
    'scale.traditional': '传统模式：人工标注，高成本，慢迭代',
    'scale.new': '新模式：Self-labeling模型配合人工反馈。工具+DaaS服务。',
    
    // Lanxiang
    'lanxiang.title': '具身AI的蓝翔技校',
    'lanxiang.step1': '泛用教材',
    'lanxiang.step2': '因材施教',
    'lanxiang.step3': '练中学',
    'lanxiang.desc1': '标注数据',
    'lanxiang.desc2': '在线学习',
    'lanxiang.desc3': '具身AI基础设施',
    
    // Dataset Page
    'dataset.hero.title': 'AgiBot World 数据集',
    'dataset.hero.desc': '用于机器人研究的高质量具身智能数据集',
    'dataset.stats.tasks': '任务数',
    'dataset.stats.hours': '时长',
    'dataset.stats.samples': '样本数',
    'dataset.stats.type': '类型',
    'dataset.features.title': '数据集特性',
    'dataset.feature1': '真实世界数据',
    'dataset.feature2': '双臂操作',
    'dataset.feature3': '多模态传感器',
    'dataset.feature4': '匿名化与压缩',
    'dataset.feature5': '相机外参校正',
    
    // Products Page
    'products.hero.title': '产品服务',
    'daas.title': 'DaaS - 数据即服务',
    'daas.desc': '预处理、高质量、针对特定任务的具身数据集。包括RLHF数据包。',
    'daas.feature1': '任务特定数据集',
    'daas.feature2': '预标注与验证',
    'daas.feature3': '即训即用',
    
    'saas.title': 'SaaS平台',
    'saas.desc': '自动化标注、数据清洗和跨本体增广的自助工具平台。',
    'saas.feature1': '自动标注',
    'saas.feature2': '数据清洗',
    'saas.feature3': '跨本体增广',
    
    'school.title': '机器人学校托管服务',
    'school.desc': '针对特定任务和机器人本体的端到端服务。数据采集、模型训练、真机验证。',
    'school.step1': '数据采集',
    'school.step2': '模型训练',
    'school.step3': '部署与验证',
    
    'ee.title': '即插即用末端执行器+技能',
    'ee.desc': '带集成传感器的电动螺丝刀末端执行器。安装即获得快速拧螺丝能力。',
    'ee.skill': '螺丝紧固技能',
    
    // Footer
    'footer.tagline': '具身AI的机器人记忆制造者',
    'footer.links': '快速链接',
    'footer.contact': '联系我们',
    'footer.copyright': '© 2025 RoboMemo. 保留所有权利。',
  }
};

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [language, setLanguage] = useState<Language>('en');

  const t = (key: string): string => {
    return content[language][key] || key;
  };

  return (
    <LanguageContext.Provider value={{ language, setLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  const context = useContext(LanguageContext);
  if (context === undefined) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
}
