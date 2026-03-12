import { useState, useEffect } from 'react';
import { CheckCircle2, XCircle, AlertCircle, TrendingUp, Users, BarChart3 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import api from '@/services/api';
import type { Review, ReviewStats, QualityDashboard, AnnotatorQuality } from '@/types';

export default function QualityControl() {
  const [dashboard, setDashboard] = useState<QualityDashboard | null>(null);
  const [stats, setStats] = useState<ReviewStats | null>(null);
  const [annotators, setAnnotators] = useState<AnnotatorQuality[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedReview, setSelectedReview] = useState<Partial<Review> | null>(null);
  const [reviewForm, setReviewForm] = useState({ score: 75, feedback: '', status: 'approved' as const });

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [dashData, statsData] = await Promise.all([
        api.getQualityDashboard(),
        api.getReviewStats(),
      ]);
      setDashboard(dashData);
      setStats(statsData);
      
      // Load annotator quality for each reviewer
      if (statsData.byReviewer) {
        const qualities = await Promise.all(
          statsData.byReviewer.map(r => api.getAnnotatorQuality(r.reviewerId))
        );
        setAnnotators(qualities);
      }
    } catch (err) {
      console.error('Failed to load quality data:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitReview = async () => {
    if (!selectedReview) return;
    try {
      await api.createReview({
        ...selectedReview,
        status: reviewForm.status,
        score: reviewForm.score,
        feedback: reviewForm.feedback,
      });
      setSelectedReview(null);
      setReviewForm({ score: 75, feedback: '', status: 'approved' });
      loadData();
    } catch (err) {
      console.error('Failed to submit review:', err);
    }
  };

  if (loading) {
    return <div className="text-center py-12 text-muted-foreground">Loading quality data...</div>;
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Quality Control</h1>
        <p className="text-muted-foreground mt-1">Monitor annotation quality and reviewer performance</p>
      </div>

      {/* Dashboard Stats */}
      {dashboard && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-2xl font-bold">{dashboard.qualityScore.toFixed(1)}</div>
                  <div className="text-xs text-muted-foreground">Quality Score</div>
                </div>
                <TrendingUp className="w-8 h-8 text-green-500 opacity-50" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-2xl font-bold">{dashboard.completedTasks}</div>
                  <div className="text-xs text-muted-foreground">Completed Tasks</div>
                </div>
                <CheckCircle2 className="w-8 h-8 text-blue-500 opacity-50" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-2xl font-bold">{dashboard.reviewBacklog}</div>
                  <div className="text-xs text-muted-foreground">Pending Reviews</div>
                </div>
                <AlertCircle className="w-8 h-8 text-yellow-500 opacity-50" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-2xl font-bold">{dashboard.total}</div>
                  <div className="text-xs text-muted-foreground">Total Reviews</div>
                </div>
                <BarChart3 className="w-8 h-8 text-purple-500 opacity-50" />
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      <Tabs defaultValue="reviewers">
        <TabsList>
          <TabsTrigger value="reviewers">Reviewer Performance</TabsTrigger>
          <TabsTrigger value="status">Review Status</TabsTrigger>
        </TabsList>

        <TabsContent value="reviewers">
          <div className="space-y-3 mt-4">
            {annotators.length === 0 ? (
              <Card>
                <CardContent className="py-8 text-center text-muted-foreground">
                  No reviewer data yet
                </CardContent>
              </Card>
            ) : (
              annotators.map((ann) => (
                <Card key={ann.userId}>
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div>
                        <h3 className="font-medium">{ann.userId}</h3>
                        <p className="text-sm text-muted-foreground">
                          {ann.totalReviews} reviews • {ann.approvalRate}% approval rate
                        </p>
                      </div>
                      <div className="text-right">
                        <div className="text-2xl font-bold text-green-600">
                          {ann.averageScore?.toFixed(1) || 'N/A'}
                        </div>
                        <div className="text-xs text-muted-foreground">Avg Score</div>
                      </div>
                    </div>
                    <div className="grid grid-cols-3 gap-2 text-sm">
                      <div className="bg-green-50 p-2 rounded">
                        <div className="font-medium text-green-700">{ann.approved}</div>
                        <div className="text-xs text-green-600">Approved</div>
                      </div>
                      <div className="bg-red-50 p-2 rounded">
                        <div className="font-medium text-red-700">{ann.rejected}</div>
                        <div className="text-xs text-red-600">Rejected</div>
                      </div>
                      <div className="bg-blue-50 p-2 rounded">
                        <div className="font-medium text-blue-700">{ann.totalReviews}</div>
                        <div className="text-xs text-blue-600">Total</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </div>
        </TabsContent>

        <TabsContent value="status">
          {stats && (
            <div className="space-y-3 mt-4">
              {Object.entries(stats.byStatus).map(([status, count]) => {
                const colors: Record<string, string> = {
                  pending: 'bg-slate-100 text-slate-700',
                  approved: 'bg-green-100 text-green-700',
                  rejected: 'bg-red-100 text-red-700',
                  needs_revision: 'bg-yellow-100 text-yellow-700',
                };
                return (
                  <Card key={status}>
                    <CardContent className="p-4 flex items-center justify-between">
                      <div>
                        <Badge className={colors[status] || 'bg-gray-100'}>
                          {status.replace('_', ' ')}
                        </Badge>
                      </div>
                      <div className="text-2xl font-bold">{count}</div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          )}
        </TabsContent>
      </Tabs>

      {/* Review Dialog */}
      {selectedReview && (
        <Dialog open={!!selectedReview} onOpenChange={() => setSelectedReview(null)}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Submit Review</DialogTitle>
            </DialogHeader>
            <div className="space-y-4 pt-4">
              <div>
                <Label>Score: {reviewForm.score}</Label>
                <Slider
                  value={[reviewForm.score]}
                  onValueChange={(v) => setReviewForm({ ...reviewForm, score: v[0] })}
                  min={0}
                  max={100}
                  step={1}
                  className="mt-2"
                />
              </div>
              <div>
                <Label>Feedback</Label>
                <Textarea
                  placeholder="Enter review feedback..."
                  value={reviewForm.feedback}
                  onChange={(e) => setReviewForm({ ...reviewForm, feedback: e.target.value })}
                />
              </div>
              <div>
                <Label>Status</Label>
                <select
                  value={reviewForm.status}
                  onChange={(e) => setReviewForm({ ...reviewForm, status: e.target.value as any })}
                  className="w-full px-3 py-2 border rounded-lg"
                >
                  <option value="approved">Approved</option>
                  <option value="rejected">Rejected</option>
                  <option value="needs_revision">Needs Revision</option>
                </select>
              </div>
              <Button onClick={handleSubmitReview} className="w-full">
                Submit Review
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
}
