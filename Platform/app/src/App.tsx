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
  Wand2,
  Sun,
  Volume2,
  Bell,
  BrainCircuit,
  LogOut,
  Users,
  CheckSquare,
  Briefcase,
  ShieldCheck,
  Package,
  Zap,
  Bot,
  DollarSign
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Sheet, SheetContent, SheetTrigger } from '@/components/ui/sheet';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import LoginPage from '@/components/LoginPage';
import DatasetManager from '@/sections/DatasetManager';
import DataCollection from '@/sections/DataCollection';
import DataAnnotation from '@/sections/DataAnnotation';
import DataVisualization from '@/sections/DataVisualization';
import SimulatorControl from '@/sections/SimulatorControl';
import DataAugmentation from '@/sections/DataAugmentation';
import AutoAnnotation from '@/sections/AutoAnnotation';
import StructuredVQA from '@/sections/StructuredVQA';
import PlatformStats from '@/sections/PlatformStats';
import UserManagement from '@/sections/UserManagement';
import MyTasks from '@/sections/MyTasks';
import TaskManagement from '@/sections/TaskManagement';
import QualityControl from '@/sections/QualityControl';
import OrderManagement from '@/sections/OrderManagement';
import BatchOperations from '@/sections/BatchOperations';
import RoboForceIntegration from '@/sections/RoboForceIntegration';
import BillingManagement from '@/sections/BillingManagement';import api from '@/services/api';
import type { User } from '@/types';
import './App.css';

type Tab = 'datasets' | 'collection' | 'annotation' | 'visualization' | 'simulators' | 'augmentation' | 'autoannotation' | 'structuredvqa' | 'stats' | 'users' | 'mytasks' | 'tasks' | 'quality' | 'orders' | 'batch' | 'roboforce' | 'billing';

const tabs = [
  { id: 'datasets' as Tab, label: 'Datasets', icon: Database },
  { id: 'collection' as Tab, label: 'Collection', icon: Play },
  { id: 'annotation' as Tab, label: 'Annotation', icon: Tag },
  { id: 'autoannotation' as Tab, label: 'Auto-Annotation', icon: Wand2 },
  { id: 'structuredvqa' as Tab, label: 'Structured VQA', icon: BrainCircuit },
  { id: 'visualization' as Tab, label: 'Visualization', icon: BarChart3 },
  { id: 'simulators' as Tab, label: 'Simulators', icon: Cpu },
  { id: 'augmentation' as Tab, label: 'Augmentation', icon: Sparkles },
  { id: 'stats' as Tab, label: 'Statistics', icon: BarChart3 },
  { id: 'mytasks' as Tab, label: 'My Tasks', icon: CheckSquare },
  { id: 'tasks' as Tab, label: 'Task Management', icon: Briefcase },
  { id: 'quality' as Tab, label: 'Quality Control', icon: ShieldCheck },
  { id: 'orders' as Tab, label: 'Orders', icon: Package },
  { id: 'batch' as Tab, label: 'Batch Ops', icon: Zap },
  { id: 'roboforce' as Tab, label: 'RoboForce', icon: Bot },
  { id: 'billing' as Tab, label: 'Billing', icon: DollarSign },  { id: 'users' as Tab, label: 'Users', icon: Users },
];

function App() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<Tab>('datasets');
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [settings, setSettings] = useState({
    darkMode: false,
    soundEnabled: true,
    notifications: true,
    autoSave: true,
    maxFileSize: 1024,
    apiEndpoint: 'http://localhost:3001'
  });

  useEffect(() => {
    checkAuth();
  }, []);

  const checkAuth = async () => {
    try {
      // Try existing token first
      const token = api.getToken();
      if (token) {
        const me = await api.getMe();
        setUser(me);
        setLoading(false);
        return;
      }
      // Auto-login with default admin credentials
      const result = await api.login('admin@robomemo.io', 'admin123');
      setUser(result.user);
    } catch (err) {
      // If auto-login fails, still let user in as guest admin
      setUser({ id: 'guest', email: 'admin@robomemo.io', name: 'Platform Admin', role: 'platform_admin', region: 'SG' } as User);
    } finally {
      setLoading(false);
    }
  };

  const handleLogin = (newUser: User) => {
    setUser(newUser);
  };

  const handleLogout = () => {
    api.logout();
    setUser(null);
  };

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
      case 'datasets':
        return <DatasetManager />;
      case 'collection':
        return <DataCollection />;
      case 'annotation':
        return <DataAnnotation />;
      case 'visualization':
        return <DataVisualization />;
      case 'simulators':
        return <SimulatorControl />;
      case 'augmentation':
        return <DataAugmentation />;
      case 'autoannotation':
        return <AutoAnnotation />;
      case 'structuredvqa':
        return <StructuredVQA />;
      case 'stats':
        return <PlatformStats />;
      case 'mytasks':
        return <MyTasks />;
      case 'tasks':
        return user?.role === 'platform_admin' || user?.role === 'reviewer' || user?.role === 'data_admin' ? <TaskManagement /> : <div className="text-center py-12 text-muted-foreground">Access denied</div>;
      case 'quality':
        return user?.role === 'platform_admin' || user?.role === 'reviewer' ? <QualityControl /> : <div className="text-center py-12 text-muted-foreground">Access denied</div>;
      case 'orders':
        return user?.role === 'platform_admin' || user?.role === 'data_admin' ? <OrderManagement /> : <div className="text-center py-12 text-muted-foreground">Access denied</div>;
      case 'batch':
        return user?.role === 'platform_admin' || user?.role === 'data_admin' ? <BatchOperations /> : <div className="text-center py-12 text-muted-foreground">Access denied</div>;
      case 'roboforce':
      case 'billing':
        return <BillingManagement />;        return user?.role === 'platform_admin' || user?.role === 'data_admin' ? <RoboForceIntegration /> : <div className="text-center py-12 text-muted-foreground">Access denied</div>;
      case 'users':
        return user?.role === 'platform_admin' ? <UserManagement /> : <div className="text-center py-12 text-muted-foreground">Access denied</div>;
      default:
        return <DatasetManager />;
    }
  };

  if (loading) {
    return <div className="min-h-screen flex items-center justify-center">Loading...</div>;
  }

  // Login page removed — auto-login with default admin

  const visibleTabs = tabs.filter(tab => {
    if (tab.id === 'users') return user.role === 'platform_admin';
    if (tab.id === 'tasks') return user.role === 'platform_admin' || user.role === 'reviewer' || user.role === 'data_admin';
    if (tab.id === 'quality') return user.role === 'platform_admin' || user.role === 'reviewer';
    if (tab.id === 'orders') return user.role === 'platform_admin' || user.role === 'data_admin';
    if (tab.id === 'batch') return user.role === 'platform_admin' || user.role === 'data_admin';
    if (tab.id === 'roboforce') return user.role === 'platform_admin' || user.role === 'data_admin';
    if (tab.id === 'billing') return user.role === 'platform_admin' || user.role === 'data_admin';    return true;
  });

  return (
    <div className="min-h-screen bg-background flex">
      {/* Desktop Sidebar */}
      <aside className="hidden lg:flex w-64 flex-col border-r bg-card">
        <div className="p-6 border-b">
          <div className="flex items-center gap-3 mb-1">
            <svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect width="36" height="36" rx="8" fill="hsl(var(--primary))" opacity="0.12"/>
              <rect x="8" y="10" width="20" height="14" rx="3" stroke="hsl(var(--primary))" strokeWidth="2" fill="none"/>
              <circle cx="14" cy="17" r="2" fill="hsl(var(--primary))"/>
              <circle cx="22" cy="17" r="2" fill="hsl(var(--primary))"/>
              <line x1="14" y1="22" x2="22" y2="22" stroke="hsl(var(--primary))" strokeWidth="1.5" strokeLinecap="round"/>
              <line x1="12" y1="8" x2="12" y2="10" stroke="hsl(var(--primary))" strokeWidth="2" strokeLinecap="round"/>
              <line x1="24" y1="8" x2="24" y2="10" stroke="hsl(var(--primary))" strokeWidth="2" strokeLinecap="round"/>
              <rect x="26" y="15" width="4" height="3" rx="1" fill="hsl(var(--primary))" opacity="0.5"/>
              <rect x="6" y="15" width="4" height="3" rx="1" fill="hsl(var(--primary))" opacity="0.5"/>
            </svg>
            <div>
              <h1 className="text-lg font-bold text-primary leading-tight">RoboMemo</h1>
              <p className="text-[10px] text-muted-foreground">Embodied Data Platform</p>
            </div>
          </div>
          <p className="text-xs text-muted-foreground">v1.0.0</p>
          <div className="mt-3 pt-3 border-t">
            <p className="text-xs font-medium text-muted-foreground">{user.name}</p>
            <p className="text-xs text-muted-foreground capitalize">{user.role.replace('_', ' ')}</p>
          </div>
        </div>
        
        <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
          {visibleTabs.map((tab) => {
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
        
        <div className="p-4 border-t space-y-2">
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
                    <Label className="text-sm font-medium">Sound</Label>
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
              </div>
            </DialogContent>
          </Dialog>
          <Button variant="outline" className="w-full" size="sm" onClick={handleLogout}>
            <LogOut className="w-4 h-4 mr-2" />
            Logout
          </Button>
        </div>
      </aside>

      {/* Mobile Header */}
      <div className="lg:hidden fixed top-0 left-0 right-0 z-50 bg-card border-b">
        <div className="flex items-center justify-between p-4">
          <div>
            <h1 className="text-lg font-bold">RoboMemo</h1>
            <p className="text-xs text-muted-foreground">{user.name}</p>
          </div>
          <Sheet open={isMobileMenuOpen} onOpenChange={setIsMobileMenuOpen}>
            <SheetTrigger asChild>
              <Button variant="ghost" size="icon">
                <Menu className="w-6 h-6" />
              </Button>
            </SheetTrigger>
            <SheetContent side="left" className="w-64 p-0">
              <div className="p-6 border-b">
                <h1 className="text-xl font-bold">RoboMemo</h1>
                <p className="text-xs text-muted-foreground mt-2">{user.name}</p>
              </div>
              <nav className="p-4 space-y-1">
                {visibleTabs.map((tab) => {
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
              <div className="p-4 border-t">
                <Button variant="outline" className="w-full" size="sm" onClick={handleLogout}>
                  <LogOut className="w-4 h-4 mr-2" />
                  Logout
                </Button>
              </div>
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
