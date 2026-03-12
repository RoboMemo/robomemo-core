import { useState, useEffect } from 'react';
import { Plus, Trash2, Package, DollarSign, Clock, CheckCircle2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import api from '@/services/api';
import type { Order, OrderStats, OrderStatus, Dataset } from '@/types';

const STATUS_COLORS: Record<OrderStatus, string> = {
  draft:       'bg-slate-100 text-slate-700',
  pending:     'bg-blue-100 text-blue-700',
  in_progress: 'bg-yellow-100 text-yellow-700',
  review:      'bg-purple-100 text-purple-700',
  completed:   'bg-green-100 text-green-700',
  cancelled:   'bg-red-100 text-red-700',
};

export default function OrderManagement() {
  const [orders, setOrders] = useState<Order[]>([]);
  const [stats, setStats] = useState<OrderStats | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [viewMode, setViewMode] = useState<'table' | 'card'>('card');
  const [form, setForm] = useState({
    title: '', description: '', clientName: '', clientContact: '',
    priority: 'normal', datasetId: '', totalEpisodes: 0,
    dueDate: '', budget: 0,
  });

  useEffect(() => {
    loadAll();
  }, []);

  const loadAll = async () => {
    try {
      const [ordersData, statsData, datasetsData] = await Promise.all([
        api.getOrders(),
        api.getOrderStats(),
        api.getDatasets().catch(() => []),
      ]);
      setOrders(ordersData);
      setStats(statsData);
      setDatasets(datasetsData);
    } catch (err) {
      console.error('Failed to load orders:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async () => {
    if (!form.title) return;
    try {
      await api.createOrder({
        title: form.title,
        description: form.description || undefined,
        clientName: form.clientName || undefined,
        clientContact: form.clientContact || undefined,
        priority: form.priority,
        datasetId: form.datasetId || undefined,
        totalEpisodes: form.totalEpisodes,
        dueDate: form.dueDate || undefined,
        budget: form.budget || undefined,
      });
      setForm({ title: '', description: '', clientName: '', clientContact: '', priority: 'normal', datasetId: '', totalEpisodes: 0, dueDate: '', budget: 0 });
      setIsDialogOpen(false);
      loadAll();
    } catch (err) {
      console.error('Failed to create order:', err);
    }
  };

  const handleStatusChange = async (id: string, status: string) => {
    try {
      await api.updateOrderStatus(id, status);
      loadAll();
    } catch (err) {
      console.error('Failed to update order status:', err);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this order?')) return;
    try {
      await api.deleteOrder(id);
      loadAll();
    } catch (err) {
      console.error('Failed to delete order:', err);
    }
  };

  if (loading) return <div className="text-center py-12 text-muted-foreground">Loading...</div>;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Order Management</h1>
          <p className="text-muted-foreground mt-1">Track annotation orders and client projects</p>
        </div>
        <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
          <DialogTrigger asChild>
            <Button><Plus className="w-4 h-4 mr-2" /> New Order</Button>
          </DialogTrigger>
          <DialogContent className="max-w-lg">
            <DialogHeader><DialogTitle>Create Order</DialogTitle></DialogHeader>
            <div className="space-y-4 pt-4 max-h-[60vh] overflow-y-auto">
              <div>
                <Label>Title</Label>
                <Input placeholder="Order title" value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} />
              </div>
              <div>
                <Label>Description</Label>
                <Textarea placeholder="Details..." value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>Client Name</Label>
                  <Input placeholder="Client" value={form.clientName} onChange={(e) => setForm({ ...form, clientName: e.target.value })} />
                </div>
                <div>
                  <Label>Client Contact</Label>
                  <Input placeholder="email@..." value={form.clientContact} onChange={(e) => setForm({ ...form, clientContact: e.target.value })} />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>Priority</Label>
                  <Select value={form.priority} onValueChange={(v) => setForm({ ...form, priority: v })}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="low">Low</SelectItem>
                      <SelectItem value="normal">Normal</SelectItem>
                      <SelectItem value="high">High</SelectItem>
                      <SelectItem value="urgent">Urgent</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label>Dataset</Label>
                  <Select value={form.datasetId} onValueChange={(v) => setForm({ ...form, datasetId: v })}>
                    <SelectTrigger><SelectValue placeholder="Select..." /></SelectTrigger>
                    <SelectContent>
                      {datasets.map((d) => (
                        <SelectItem key={d.id} value={d.id}>{d.name}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <Label>Total Episodes</Label>
                  <Input type="number" value={form.totalEpisodes} onChange={(e) => setForm({ ...form, totalEpisodes: parseInt(e.target.value) || 0 })} />
                </div>
                <div>
                  <Label>Budget ($)</Label>
                  <Input type="number" value={form.budget} onChange={(e) => setForm({ ...form, budget: parseFloat(e.target.value) || 0 })} />
                </div>
                <div>
                  <Label>Due Date</Label>
                  <Input type="date" value={form.dueDate} onChange={(e) => setForm({ ...form, dueDate: e.target.value })} />
                </div>
              </div>
              <Button onClick={handleCreate} className="w-full">Create Order</Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <Card>
            <CardContent className="p-4 flex items-center gap-3">
              <Package className="w-8 h-8 text-blue-500 opacity-50" />
              <div>
                <div className="text-2xl font-bold">{stats.total}</div>
                <div className="text-xs text-muted-foreground">Total Orders</div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4 flex items-center gap-3">
              <Clock className="w-8 h-8 text-yellow-500 opacity-50" />
              <div>
                <div className="text-2xl font-bold">{stats.byStatus['in_progress'] || 0}</div>
                <div className="text-xs text-muted-foreground">In Progress</div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4 flex items-center gap-3">
              <CheckCircle2 className="w-8 h-8 text-green-500 opacity-50" />
              <div>
                <div className="text-2xl font-bold">{stats.byStatus['completed'] || 0}</div>
                <div className="text-xs text-muted-foreground">Completed</div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4 flex items-center gap-3">
              <DollarSign className="w-8 h-8 text-emerald-500 opacity-50" />
              <div>
                <div className="text-2xl font-bold">${stats.totalBudget.toLocaleString()}</div>
                <div className="text-xs text-muted-foreground">Total Budget</div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Order Cards */}
      {orders.length === 0 ? (
        <Card>
          <CardContent className="py-12 text-center text-muted-foreground">
            <Package className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p className="text-lg font-medium">No orders yet</p>
            <p className="text-sm mt-1">Create your first order to get started</p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {orders.map((order) => {
            const progress = order.totalEpisodes > 0
              ? Math.round((order.completedEpisodes / order.totalEpisodes) * 100) : 0;
            return (
              <Card key={order.id} className="hover:shadow-md transition-shadow">
                <CardHeader className="pb-2">
                  <div className="flex items-start justify-between">
                    <CardTitle className="text-base">{order.title}</CardTitle>
                    <Button variant="ghost" size="sm" onClick={() => handleDelete(order.id)}>
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                  <div className="flex gap-2">
                    <Badge className={STATUS_COLORS[order.status]}>
                      {order.status.replace('_', ' ')}
                    </Badge>
                    {order.priority !== 'normal' && (
                      <Badge variant="outline">{order.priority}</Badge>
                    )}
                  </div>
                </CardHeader>
                <CardContent className="space-y-3">
                  {order.description && (
                    <p className="text-sm text-muted-foreground line-clamp-2">{order.description}</p>
                  )}
                  {order.clientName && (
                    <div className="text-sm"><span className="font-medium">Client:</span> {order.clientName}</div>
                  )}
                  <div>
                    <div className="flex justify-between text-xs text-muted-foreground mb-1">
                      <span>{order.completedEpisodes} / {order.totalEpisodes} episodes</span>
                      <span>{progress}%</span>
                    </div>
                    <Progress value={progress} className="h-2" />
                  </div>
                  {order.budget && (
                    <div className="text-sm">
                      <span className="font-medium">Budget:</span> ${order.budget.toLocaleString()}
                      {order.actualCost > 0 && ` · Cost: $${order.actualCost.toLocaleString()}`}
                    </div>
                  )}
                  {order.dueDate && (
                    <div className="text-xs text-muted-foreground">
                      Due: {new Date(order.dueDate).toLocaleDateString()}
                    </div>
                  )}

                  {/* Quick status change */}
                  <div className="flex gap-1 pt-2">
                    {order.status === 'draft' && (
                      <Button size="sm" variant="outline" onClick={() => handleStatusChange(order.id, 'pending')}>Submit</Button>
                    )}
                    {order.status === 'pending' && (
                      <Button size="sm" variant="outline" onClick={() => handleStatusChange(order.id, 'in_progress')}>Start</Button>
                    )}
                    {order.status === 'in_progress' && (
                      <Button size="sm" variant="outline" onClick={() => handleStatusChange(order.id, 'review')}>Review</Button>
                    )}
                    {order.status === 'review' && (
                      <Button size="sm" variant="outline" onClick={() => handleStatusChange(order.id, 'completed')}>Complete</Button>
                    )}
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
