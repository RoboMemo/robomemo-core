import { useState, useEffect } from 'react';
import { ClipboardList, Play, CheckCircle2, XCircle, Clock, AlertTriangle, ChevronDown, ChevronUp } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import api from '@/services/api';
import type { Task, TaskStatus, TaskPriority } from '@/types';

const STATUS_CONFIG: Record<TaskStatus, { color: string; icon: any }> = {
  pending:     { color: 'bg-slate-100 text-slate-700', icon: Clock },
  assigned:    { color: 'bg-blue-100 text-blue-700', icon: ClipboardList },
  in_progress: { color: 'bg-yellow-100 text-yellow-700', icon: Play },
  completed:   { color: 'bg-green-100 text-green-700', icon: CheckCircle2 },
  rejected:    { color: 'bg-red-100 text-red-700', icon: XCircle },
};

const PRIORITY_CONFIG: Record<TaskPriority, string> = {
  low:    'bg-slate-100 text-slate-600',
  normal: 'bg-blue-100 text-blue-600',
  high:   'bg-orange-100 text-orange-600',
  urgent: 'bg-red-100 text-red-600',
};

export default function MyTasks() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  useEffect(() => {
    loadTasks();
  }, []);

  const loadTasks = async () => {
    try {
      const data = await api.getMyTasks();
      setTasks(data);
    } catch (err) {
      console.error('Failed to load tasks:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleStatusChange = async (taskId: string, status: string) => {
    try {
      await api.updateTaskStatus(taskId, status);
      loadTasks();
    } catch (err) {
      console.error('Failed to update task status:', err);
    }
  };

  if (loading) {
    return <div className="text-center py-12 text-muted-foreground">Loading tasks...</div>;
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">My Tasks</h1>
        <p className="text-muted-foreground mt-1">Your assigned annotation and review tasks</p>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        {(['pending', 'assigned', 'in_progress', 'completed', 'rejected'] as TaskStatus[]).map((s) => {
          const count = tasks.filter(t => t.status === s).length;
          const cfg = STATUS_CONFIG[s];
          const Icon = cfg.icon;
          return (
            <Card key={s}>
              <CardContent className="p-4 flex items-center gap-3">
                <Icon className="w-5 h-5 text-muted-foreground" />
                <div>
                  <div className="text-xl font-bold">{count}</div>
                  <div className="text-xs text-muted-foreground capitalize">{s.replace('_', ' ')}</div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {tasks.length === 0 ? (
        <Card>
          <CardContent className="py-12 text-center text-muted-foreground">
            <ClipboardList className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p className="text-lg font-medium">No tasks assigned</p>
            <p className="text-sm mt-1">Tasks will appear here once assigned to you</p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-3">
          {tasks.map((task) => {
            const cfg = STATUS_CONFIG[task.status];
            const Icon = cfg.icon;
            const expanded = expandedId === task.id;
            return (
              <Card key={task.id}>
                <CardContent className="p-4">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <button onClick={() => setExpandedId(expanded ? null : task.id)} className="flex items-center gap-1">
                          {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                        </button>
                        <h3 className="font-semibold text-lg">{task.title}</h3>
                      </div>
                      <div className="flex flex-wrap items-center gap-2 mb-2">
                        <Badge className={cfg.color}>
                          <Icon className="w-3 h-3 mr-1" />
                          {task.status.replace('_', ' ')}
                        </Badge>
                        <Badge className={PRIORITY_CONFIG[task.priority]}>
                          {task.priority === 'urgent' && <AlertTriangle className="w-3 h-3 mr-1" />}
                          {task.priority}
                        </Badge>
                        <Badge variant="outline">{task.type}</Badge>
                        {task.dueDate && (
                          <span className="text-xs text-muted-foreground">
                            Due: {new Date(task.dueDate).toLocaleDateString()}
                          </span>
                        )}
                      </div>
                      {task.description && (
                        <p className="text-sm text-muted-foreground">{task.description}</p>
                      )}
                    </div>

                    <div className="flex gap-2 shrink-0">
                      {(task.status === 'pending' || task.status === 'assigned') && (
                        <Button size="sm" onClick={() => handleStatusChange(task.id, 'in_progress')}>
                          <Play className="w-3 h-3 mr-1" /> Start
                        </Button>
                      )}
                      {task.status === 'in_progress' && (
                        <>
                          <Button size="sm" onClick={() => handleStatusChange(task.id, 'completed')}>
                            <CheckCircle2 className="w-3 h-3 mr-1" /> Complete
                          </Button>
                          <Button size="sm" variant="outline" onClick={() => handleStatusChange(task.id, 'rejected')}>
                            <XCircle className="w-3 h-3 mr-1" /> Reject
                          </Button>
                        </>
                      )}
                    </div>
                  </div>

                  {expanded && (
                    <div className="mt-4 pt-4 border-t space-y-2 text-sm">
                      <div className="grid grid-cols-2 gap-2">
                        <div><span className="font-medium">Type:</span> {task.type}</div>
                        <div><span className="font-medium">Priority:</span> {task.priority}</div>
                        {task.datasetId && <div><span className="font-medium">Dataset:</span> {task.datasetId}</div>}
                        {task.episodeIds && task.episodeIds.length > 0 && (
                          <div><span className="font-medium">Episodes:</span> {task.episodeIds.length}</div>
                        )}
                        {task.startedAt && <div><span className="font-medium">Started:</span> {new Date(task.startedAt).toLocaleString()}</div>}
                        {task.completedAt && <div><span className="font-medium">Completed:</span> {new Date(task.completedAt).toLocaleString()}</div>}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
