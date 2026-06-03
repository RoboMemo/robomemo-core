import { useState, useEffect } from 'react';
import {
  Database,
  Play,
  Tag,
  BarChart3,
  Cpu,
  Sparkles,
  Settings,
  Menu,
  Sun,
  Volume2,
  Bell,
  BrainCircuit,
  ShoppingCart,
  GitBranch
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Sheet, SheetContent, SheetTrigger } from '@/components/ui/sheet';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import DatasetManager from '@/sections/DatasetManager';
import DataCollection from '@/sections/DataCollection';
import DataVisualization from '@/sections/DataVisualization';
import SimulatorControl from '@/sections/SimulatorControl';
import DataAugmentation from '@/sections/DataAugmentation';
import AnnotationWorkbench from '@/sections/AnnotationWorkbench';
import TrainingManager from '@/sections/TrainingManager';
import PlatformStats from '@/sections/PlatformStats';
import Web3Marketplace from '@/sections/Web3Marketplace';
import DataPipeline from '@/sections/DataPipeline';
import './App.css';

type Tab = 'pipeline' | 'datasets' | 'collection' | 'annotation' | 'visualization' | 'simulators' | 'augmentation' | 'training' | 'stats' | 'marketplace';

const tabs = [
  { id: 'pipeline' as Tab,      label: '数据流水线', icon: GitBranch },
  { id: 'datasets' as Tab,      label: '数据集',     icon: Database },
  { id: 'collection' as Tab,    label: '数据采集',   icon: Play },
  { id: 'annotation' as Tab,    label: '标注工作台', icon: Tag },
  { id: 'visualization' as Tab, label: '可视化',     icon: BarChart3 },
  { id: 'simulators' as Tab,    label: '模拟器',     icon: Cpu },
  { id: 'augmentation' as Tab,  label: '数据增强',   icon: Sparkles },
  { id: 'training' as Tab,      label: '训练管理',   icon: BrainCircuit },
  { id: 'stats' as Tab,         label: '统计',       icon: BarChart3 },
  { id: 'marketplace' as Tab,   label: 'Web3 市场',  icon: ShoppingCart },
];

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('pipeline');
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  // Allow DataPipeline to navigate to marketplace via custom event
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail as Tab;
      if (detail) setActiveTab(detail);
    };
    window.addEventListener('navigate', handler);
    return () => window.removeEventListener('navigate', handler);
  }, []);
  const [settings, setSettings] = useState({
    darkMode: false,
    soundEnabled: true,
    notifications: true,
    autoSave: true,
    maxFileSize: 1024,
    apiEndpoint: 'http://localhost:3001'
  });

  const handleSettingChange = (key: string, value: any) => {
    setSettings(prev => ({
      ...prev,
      [key]: value
    }));
    localStorage.setItem('platformSettings', JSON.stringify({
      ...settings,
      [key]: value
    }));
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'pipeline':
        return <DataPipeline />;
      case 'datasets':
        return <DatasetManager />;
      case 'collection':
        return <DataCollection />;
      case 'annotation':
        return <AnnotationWorkbench />;
      case 'visualization':
        return <DataVisualization />;
      case 'simulators':
        return <SimulatorControl />;
      case 'augmentation':
        return <DataAugmentation />;
      case 'training':
        return <TrainingManager />;
      case 'stats':
        return <PlatformStats />;
      case 'marketplace':
        return <Web3Marketplace />;
      default:
        return <DataPipeline />;
    }
  };

  return (
    <div className="min-h-screen bg-background flex">
      {/* Desktop Sidebar */}
      <aside className="hidden lg:flex w-64 flex-col border-r bg-card">
        <div className="p-6 border-b">
          <h1 className="text-xl font-bold text-primary">Embodied Data Platform</h1>
          <p className="text-xs text-muted-foreground mt-1">v1.0.0</p>
        </div>
        
        <nav className="flex-1 p-4 space-y-1">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                }`}
              >
                <Icon className="w-5 h-5" />
                {tab.label}
              </button>
            );
          })}
        </nav>
        
        <div className="p-4 border-t">
          <Dialog open={isSettingsOpen} onOpenChange={setIsSettingsOpen}>
            <DialogTrigger asChild>
              <Button variant="outline" className="w-full" size="sm">
                <Settings className="w-4 h-4 mr-2" />
                Settings
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-lg">
              <DialogHeader>
                <DialogTitle>Platform Settings</DialogTitle>
              </DialogHeader>
              <div className="space-y-6 pt-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Sun className="w-4 h-4" />
                    <Label className="text-sm font-medium">Dark Mode</Label>
                  </div>
                  <Switch 
                    checked={settings.darkMode}
                    onCheckedChange={(v) => handleSettingChange('darkMode', v)}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Volume2 className="w-4 h-4" />
                    <Label className="text-sm font-medium">Sound Effects</Label>
                  </div>
                  <Switch 
                    checked={settings.soundEnabled}
                    onCheckedChange={(v) => handleSettingChange('soundEnabled', v)}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Bell className="w-4 h-4" />
                    <Label className="text-sm font-medium">Notifications</Label>
                  </div>
                  <Switch 
                    checked={settings.notifications}
                    onCheckedChange={(v) => handleSettingChange('notifications', v)}
                  />
                </div>

                <div className="border-t pt-4">
                  <Label className="text-sm font-medium block mb-1">后端服务器地址</Label>
                  <p className="text-xs text-muted-foreground mb-2">
                    连接到 Mac Mini 上运行的 RoboMemoClaw 后端。<br />
                    公网：<span className="font-mono">https://destitute-navigate-street.ngrok-free.dev</span><br />
                    本地：<span className="font-mono">http://localhost:3001</span>
                  </p>
                  <input
                    type="text"
                    value={settings.apiEndpoint}
                    onChange={(e) => handleSettingChange('apiEndpoint', e.target.value)}
                    className="w-full px-3 py-2 border border-input rounded-md text-sm font-mono"
                  />
                  <div className="flex gap-2 mt-2">
                    <button
                      onClick={() => handleSettingChange('apiEndpoint', 'http://localhost:3001')}
                      className="text-xs px-2 py-1 rounded border border-input hover:bg-accent"
                    >
                      本地
                    </button>
                    <button
                      onClick={() => handleSettingChange('apiEndpoint', 'https://destitute-navigate-street.ngrok-free.dev')}
                      className="text-xs px-2 py-1 rounded border border-input hover:bg-accent"
                    >
                      公网 ngrok
                    </button>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <Label className="text-sm font-medium">Auto-Save</Label>
                  <Switch 
                    checked={settings.autoSave}
                    onCheckedChange={(v) => handleSettingChange('autoSave', v)}
                  />
                </div>

                <div className="pt-4 border-t">
                  <Button variant="outline" className="w-full" size="sm">
                    Reset to Defaults
                  </Button>
                </div>
              </div>
            </DialogContent>
          </Dialog>
        </div>
      </aside>

      {/* Mobile Header */}
      <div className="lg:hidden fixed top-0 left-0 right-0 z-50 bg-card border-b">
        <div className="flex items-center justify-between p-4">
          <h1 className="text-lg font-bold">Embodied Data Platform</h1>
          <Sheet open={isMobileMenuOpen} onOpenChange={setIsMobileMenuOpen}>
            <SheetTrigger asChild>
              <Button variant="ghost" size="icon">
                <Menu className="w-6 h-6" />
              </Button>
            </SheetTrigger>
            <SheetContent side="left" className="w-64 p-0">
              <div className="p-6 border-b">
                <h1 className="text-xl font-bold">Embodied Data Platform</h1>
              </div>
              <nav className="p-4 space-y-1">
                {tabs.map((tab) => {
                  const Icon = tab.icon;
                  return (
                    <button
                      key={tab.id}
                      onClick={() => {
                        setActiveTab(tab.id);
                        setIsMobileMenuOpen(false);
                      }}
                      className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors ${
                        activeTab === tab.id
                          ? 'bg-primary text-primary-foreground'
                          : 'text-muted-foreground hover:bg-accent'
                      }`}
                    >
                      <Icon className="w-5 h-5" />
                      {tab.label}
                    </button>
                  );
                })}
              </nav>
            </SheetContent>
          </Sheet>
        </div>
      </div>

      {/* Main Content */}
      <main className="flex-1 overflow-auto lg:pt-0 pt-16">
        <div className="p-6 lg:p-8">
          {renderContent()}
        </div>
      </main>
    </div>
  );
}

export default App;
