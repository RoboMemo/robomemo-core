import { useState, useEffect } from 'react';
import { DollarSign, TrendingUp, AlertCircle, Plus, Edit2, Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import api from '@/services/api';

interface BillingRate {
  id: string;
  type: string;
  rate: number;
  currency: string;
  unit: string;
  description: string;
  updatedAt: string;
}

interface BillingRecord {
  id: string;
  orderId: string;
  type: string;
  amount: number;
  currency: string;
  ratePerEpisode: number;
  episodesCount: number;
  status: string;
  note: string;
  createdAt: string;
}

interface BillingSummary {
  total: number;
  totalAmount: number;
  pendingAmount: number;
  paidAmount: number;
  byStatus: Array<{ status: string; count: number; total: number }>;
  byType: Array<{ type: string; count: number; total: number }>;
}

export default function BillingManagement() {
  const [rates, setRates] = useState<BillingRate[]>([]);
  const [records, setRecords] = useState<BillingRecord[]>([]);
  const [summary, setSummary] = useState<BillingSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [editingRate, setEditingRate] = useState<BillingRate | null>(null);
  const [newRate, setNewRate] = useState({ rate: '', description: '' });

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const [ratesData, recordsData, summaryData] = await Promise.all([
        api.fetch('/billing/rates'),
        api.fetch('/billing'),
        api.fetch('/billing/summary'),
      ]);
      setRates(ratesData);
      setRecords(recordsData);
      setSummary(summaryData);
    } catch (error) {
      console.error('Failed to load billing data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleUpdateRate = async (type: string) => {
    if (!newRate.rate) return;
    try {
      await api.fetch(`/billing/rates/${type}`, {
        method: 'PUT',
        body: JSON.stringify({ rate: parseFloat(newRate.rate), description: newRate.description }),
      });
      setEditingRate(null);
      setNewRate({ rate: '', description: '' });
      loadData();
    } catch (error) {
      console.error('Failed to update rate:', error);
    }
  };

  if (loading) {
    return <div className="text-center py-8">Loading billing data...</div>;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Billing & Cost Tracking</h1>
      </div>

      {/* Summary Cards */}
      {summary && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Total Amount</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">${summary.totalAmount.toFixed(2)}</div>
              <p className="text-xs text-muted-foreground mt-1">{summary.total} records</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Pending</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-amber-600">${summary.pendingAmount.toFixed(2)}</div>
              <p className="text-xs text-muted-foreground mt-1">Awaiting payment</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Paid</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">${summary.paidAmount.toFixed(2)}</div>
              <p className="text-xs text-muted-foreground mt-1">Completed</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">By Type</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-1">
                {summary.byType.slice(0, 2).map(t => (
                  <div key={t.type} className="text-xs flex justify-between">
                    <span>{t.type}</span>
                    <span className="font-semibold">${t.total.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Billing Rates */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <DollarSign className="w-5 h-5" />
            Billing Rates
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {rates.map(rate => (
              <div key={rate.id} className="flex items-center justify-between p-3 border rounded-lg">
                <div className="flex-1">
                  <div className="font-medium">{rate.type}</div>
                  <div className="text-sm text-muted-foreground">{rate.description}</div>
                </div>
                <div className="text-right mr-4">
                  <div className="font-bold">${rate.rate.toFixed(2)}</div>
                  <div className="text-xs text-muted-foreground">{rate.unit}</div>
                </div>
                <Dialog open={editingRate?.id === rate.id} onOpenChange={(open) => {
                  if (!open) setEditingRate(null);
                  else setEditingRate(rate);
                }}>
                  <DialogTrigger asChild>
                    <Button variant="ghost" size="sm">
                      <Edit2 className="w-4 h-4" />
                    </Button>
                  </DialogTrigger>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>Edit Rate: {rate.type}</DialogTitle>
                    </DialogHeader>
                    <div className="space-y-4">
                      <div>
                        <Label>Rate ({rate.currency})</Label>
                        <Input
                          type="number"
                          step="0.01"
                          defaultValue={rate.rate}
                          onChange={(e) => setNewRate({ ...newRate, rate: e.target.value })}
                        />
                      </div>
                      <div>
                        <Label>Description</Label>
                        <Input
                          defaultValue={rate.description}
                          onChange={(e) => setNewRate({ ...newRate, description: e.target.value })}
                        />
                      </div>
                      <Button onClick={() => handleUpdateRate(rate.type)} className="w-full">
                        Save Changes
                      </Button>
                    </div>
                  </DialogContent>
                </Dialog>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Billing Records */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5" />
            Billing Records
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2 px-2">Type</th>
                  <th className="text-left py-2 px-2">Episodes</th>
                  <th className="text-left py-2 px-2">Amount</th>
                  <th className="text-left py-2 px-2">Status</th>
                  <th className="text-left py-2 px-2">Date</th>
                </tr>
              </thead>
              <tbody>
                {records.map(record => (
                  <tr key={record.id} className="border-b hover:bg-muted/50">
                    <td className="py-2 px-2">{record.type}</td>
                    <td className="py-2 px-2">{record.episodesCount}</td>
                    <td className="py-2 px-2 font-semibold">${record.amount.toFixed(2)}</td>
                    <td className="py-2 px-2">
                      <Badge variant={record.status === 'paid' ? 'default' : 'secondary'}>
                        {record.status}
                      </Badge>
                    </td>
                    <td className="py-2 px-2 text-muted-foreground">
                      {new Date(record.createdAt).toLocaleDateString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
