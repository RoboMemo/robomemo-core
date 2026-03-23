import { useState, useEffect } from 'react';
import { Plus, Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import api from '@/services/api';
import type { Task, TaskStats, User, Dataset, TaskStatus, TaskPriority, TaskType } from '@/types';

const STATUS_COLORS: Record<TaskStatus, string> = {
  pending:     'bg-slate-100 text-slate-700',
  assigned:    'bg-blue-100 text-blue-700',
  in_progress: 'bg-yellow-100 text-yellow-700',
  completed:   'bg-green-100 text-green-700',
  rejected:    'bg-red-100 text-red-700',
};

const PRIORITY_COLORS: Record<TaskPriority, string> = {
  low:    'bg-slate-100 text-slate-600',
  normal: 'bg-blue-100 text-blue-600',
  high:   'bg-orange-100 text-orange-600',
  urgent: 'bg-red-100 text-red-600',
};

const STATUSES: TaskStatus[] = ['pending', 'assigned', 'in_progress', 'completed', 'rejected'];

export default function TaskManagement() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [stats, setStats] = useState<TaskStats | null>(null);
  const [users, setUsers] = useState<User[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [form, setForm] = useState({
    title: '', description: '', type: 'annotation' as TaskType,
    priority: 'normal' as TaskPriority, assignedTo: '', datasetId: '', dueDate: ''
  });

  useEffect(() => {
    loadAll();
  }, []);

  const loadAll = async () => {
    try {
      const [tasksData, statsData, usersData, datasetsData] = await Promise.all([
        api.getTasks(),
        api.getTaskStats(),
        api.getUsers().catch(() => []),
        api.getDatasets().catch(() => []),
      ]);
      setTasks(tasksData);
      setStats(statsData);
      setUsers(usersData);
      setDatasets(datasetsData);
    } catch (err) {
      console.error('Failed to load tasks:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async () => {
    if (!form.title) return;
    try {
      await api.createTask({
        title: form.title,
        description: form.description || undefined,
        type: form.type,
        priority: form.priority,
        assignedTo: form.assignedTo || undefined,
        datasetId: form.datasetId || undefined,
        dueDate: form.dueDate || undefined,
        status: form.assignedTo ? 'assigned' : 'pending',
      });
      setForm({ title: '', description: '', type: 'annotation', priority: 'normal', assignedTo: '', datasetId: '', dueDate: '' });
      setIsDialogOpen(false);
      loadAll();
    } catch (err) {
      console.error('Failed to create task:', err);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this task?')) return;
    try {
      await api.deleteTask(id);
      loadAll();
    } catch (err) {
      console.error('Failed to delete task:', err);
    }
  };

  if (loading) {
    return <div className="text-center py-12 text-muted-foreground">Loading...</div>;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Task Management</h1>
          <p className="text-muted-foreground mt-1">Create, assign, and track annotation tasks</p>
        </div>
        <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
          <DialogTrigger asChild>
            <Button><Plus className="w-4 h-4 mr-2" /> New Task</Button>
          </DialogTrigger>
          <DialogContent className="max-w-lg">
            <DialogHeader><DialogTitle>Create Task</DialogTitle></DialogHeader>
            <div className="space-y-4 pt-4">
              <div>
                <Label>Title</Label>
                <Input placeholder="Annotate dataset X" value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} />
              </div>
              <div>
                <Label>Description</Label>
                <Textarea placeholder="Details..." value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>Type</Label>
                  <Select value={form.type} onValueChange={(v) => setForm({ ...form, type: v as TaskType })}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="annotation">Annotation</SelectItem>
                      <SelectItem value="review">Review</SelectItem>
                      <SelectItem value="vqa">VQA</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label>Priority</Label>
                  <Select value={form.priority} onValueChange={(v) => setForm({ ...form, priority: v as TaskPriority })}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="low">Low</SelectItem>
                      <SelectItem value="normal">Normal</SelectItem>
                      <SelectItem value="high">High</SelectItem>
                      <SelectItem value="urgent">Urgent</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div>
                <Label>Assign To</Label>
                <Select value={form.assignedTo} onValueChange={(v) => setForm({ ...form, assignedTo: v })}>
                  <SelectTrigger><SelectValue placeholder="Unassigned" /></SelectTrigger>
                  <SelectContent>
                    {users.map((u) => (
                      <SelectItem key={u.id} value={u.id}>{u.name} ({u.role})</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label>Dataset</Label>
                <Select value={form.datasetId} onValueChange={(v) => setForm({ ...form, datasetId: v })}>
                  <SelectTrigger><SelectValue placeholder="Select dataset" /></SelectTrigger>
                  <SelectContent>
                    {datasets.map((d) => (
                      <SelectItem key={d.id} value={d.id}>{d.name}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label>Due Date</Label>
                <Input type="date" value={form.dueDate} onChange={(e) => setForm({ ...form, dueDate: e.target.value })} />
              </div>
              <Button onClick={handleCreate} className="w-full">Create Task</Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold">{stats.total}</div>
              <div className="text-sm text-muted-foreground">Total Tasks</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold text-yellow-600">{stats.byStatus['in_progress'] || 0}</div>
              <div className="text-sm text-muted-foreground">In Progress</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold text-green-600">{stats.byStatus['completed'] || 0}</div>
              <div className="text-sm text-muted-foreground">Completed</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold text-slate-600">{stats.byStatus['pending'] || 0}</div>
              <div className="text-sm text-muted-foreground">Pending</div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Kanban view */}
      <Tabs defaultValue="kanban">
        <TabsList>
          <TabsTrigger value="kanban">Kanban</TabsTrigger>
          <TabsTrigger value="list">List</TabsTrigger>
        </TabsList>

        <TabsContent value="kanban">
          <div className="grid grid-cols-1 sm:grid-cols-3 lg:grid-cols-5 gap-4 mt-4">
            {STATUSES.map((status) => {
              const filtered = tasks.filter(t => t.status === status);
              return (
                <div key={status}>
                  <h3 className="text-sm font-medium mb-2 capitalize flex items-center gap-2">
                    <Badge className={STATUS_COLORS[status]}>{status.replace('_', ' ')}</Badge>
                    <span className="text-muted-foreground">{filtered.length}</span>
                  </h3>
                  <div className="space-y-2">
                    {filtered.map((task) => (
                      <Card key={task.id} className="cursor-pointer hover:shadow-md transition-shadow">
                        <CardContent className="p-3">
                          <h4 className="font-medium text-sm mb-1 line-clamp-2">{task.title}</h4>
                          <div className="flex flex-wrap gap-1">
                            <Badge className={PRIORITY_COLORS[task.priority]} variant="outline">
                              {task.priority}
                            </Badge>
                            <Badge variant="outline">{task.type}</Badge>
                          </div>
                          {task.dueDate && (
                            <div className="text-xs text-muted-foreground mt-1">
                              Due: {new Date(task.dueDate).toLocaleDateString()}
                            </div>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                    {filtered.length === 0 && (
                      <div className="text-xs text-muted-foreground text-center py-4 border border-dashed rounded-lg">
                        No tasks
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </TabsContent>

        <TabsContent value="list">
          <div className="space-y-2 mt-4">
            {tasks.length === 0 ? (
              <Card>
                <CardContent className="py-12 text-center text-muted-foreground">
                  No tasks created yet
                </CardContent>
              </Card>
            ) : (
              tasks.map((task) => (
                <Card key={task.id}>
                  <CardContent className="p-4 flex items-center justify-between">
                    <div>
                      <h4 className="font-medium">{task.title}</h4>
                      <div className="flex gap-2 mt-1">
                        <Badge className={STATUS_COLORS[task.status]}>{task.status.replace('_', ' ')}</Badge>
                        <Badge className={PRIORITY_COLORS[task.priority]}>{task.priority}</Badge>
                        <Badge variant="outline">{task.type}</Badge>
                      </div>
                    </div>
                    <Button variant="ghost" size="sm" onClick={() => handleDelete(task.id)}>
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </CardContent>
                </Card>
              ))
            )}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
