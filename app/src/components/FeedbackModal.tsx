import { useState, useEffect } from 'react';
import { submitFeedback, getFeedbackList } from '../services/api';
import type { FeedbackSubmission, FeedbackItem } from '../types';
import { 
  X, 
  Star, 
  Send, 
  CheckCircle2, 
  MessageSquareHeart, 
  Heart, 
  User,
  Tag,
  FileText
} from 'lucide-react';

interface FeedbackModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function FeedbackModal({ isOpen, onClose }: FeedbackModalProps) {
  const [form, setForm] = useState<FeedbackSubmission>({
    user_name: '',
    category: 'general',
    rating: 5,
    comment: '',
  });
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [submittedMessage, setSubmittedMessage] = useState<string | null>(null);
  const [communityList, setCommunityList] = useState<FeedbackItem[]>([]);

  useEffect(() => {
    if (isOpen) {
      getFeedbackList().then(setCommunityList);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.comment.trim()) return;

    setSubmitting(true);
    const res = await submitFeedback(form);
    setSubmittedMessage(res.message);
    setSubmitting(false);

    getFeedbackList().then(setCommunityList);
    setForm({ user_name: '', category: 'general', rating: 5, comment: '' });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-[#030712]/85 backdrop-blur-md">
      <div className="bg-[#0b0f19] border border-slate-800 rounded-3xl w-full max-w-2xl max-h-[90vh] overflow-y-auto shadow-2xl flex flex-col">
        
        {/* Modal Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-800/80 sticky top-0 bg-[#0b0f19]/95 backdrop-blur z-10">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-cyan-500/10 text-cyan-400 rounded-2xl border border-cyan-500/20 shadow-sm">
              <MessageSquareHeart className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base sm:text-lg font-heading font-extrabold text-slate-100">
                Global Community Notes & Scenarios
              </h3>
              <p className="text-xs text-slate-400">
                Share scenario validation reports, local testing benchmarks, and suggestions.
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800 transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 space-y-6">
          
          {/* Submission Form */}
          <form onSubmit={handleSubmit} className="bg-[#070b14] p-5 rounded-2xl border border-slate-800/80 space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                  <User className="w-3.5 h-3.5 text-cyan-400" />
                  Your Name / Organization
                </label>
                <input
                  type="text"
                  placeholder="e.g. EcoWater Lab, Jane Doe"
                  value={form.user_name}
                  onChange={e => setForm(f => ({ ...f, user_name: e.target.value }))}
                  className="w-full bg-[#030712] border border-slate-800 rounded-xl px-3.5 py-2.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                  <Tag className="w-3.5 h-3.5 text-cyan-400" />
                  Feedback Category
                </label>
                <select
                  value={form.category}
                  onChange={e => setForm(f => ({ ...f, category: e.target.value as any }))}
                  className="w-full bg-[#030712] border border-slate-800 rounded-xl px-3.5 py-2.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 cursor-pointer"
                >
                  <option value="forecast_accuracy">Forecast Accuracy & Validation</option>
                  <option value="scenario_preset">Custom Scenario Preset</option>
                  <option value="feature_request">Feature Request / Idea</option>
                  <option value="bug_report">Bug Report</option>
                  <option value="general">General Community Note</option>
                </select>
              </div>

            </div>

            {/* Rating Stars */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-slate-300">Experience Rating</label>
              <div className="flex items-center gap-2">
                {[1, 2, 3, 4, 5].map(star => (
                  <button
                    key={star}
                    type="button"
                    onClick={() => setForm(f => ({ ...f, rating: star }))}
                    className="p-1 text-slate-700 hover:text-amber-400 transition-colors cursor-pointer"
                  >
                    <Star
                      className={`w-5 h-5 ${
                        star <= form.rating ? 'text-amber-400 fill-amber-400 drop-shadow-[0_0_6px_rgba(251,191,36,0.5)]' : 'text-slate-700'
                      }`}
                    />
                  </button>
                ))}
                <span className="text-xs text-slate-400 ml-2 font-mono font-medium">{form.rating} / 5 Stars</span>
              </div>
            </div>

            {/* Message / Scenario */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                <FileText className="w-3.5 h-3.5 text-cyan-400" />
                Notes / Scenario Configuration
              </label>
              <textarea
                required
                rows={3}
                placeholder="Share your experience, suggest a scenario, or report real-world water testing results..."
                value={form.comment}
                onChange={e => setForm(f => ({ ...f, comment: e.target.value }))}
                className="w-full bg-[#030712] border border-slate-800 rounded-xl px-3.5 py-2.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 leading-relaxed"
              />
            </div>

            {/* Success Alert */}
            {submittedMessage && (
              <div className="flex items-center gap-2.5 p-3.5 bg-emerald-500/10 text-emerald-300 border border-emerald-500/30 rounded-2xl text-xs">
                <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-400" />
                <span>{submittedMessage}</span>
              </div>
            )}

            <button
              type="submit"
              disabled={submitting}
              className="w-full flex items-center justify-center gap-2 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-bold text-xs rounded-2xl shadow-lg shadow-cyan-500/25 transition-all cursor-pointer disabled:opacity-50"
            >
              <Send className="w-4 h-4" />
              {submitting ? 'Transmitting Feedback...' : 'Publish Community Note'}
            </button>
          </form>

          {/* Worldwide Community Feed */}
          <div className="space-y-3.5">
            <h4 className="text-xs font-heading font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
              <Heart className="w-4 h-4 text-rose-400" />
              Recent Worldwide Community Notes ({communityList.length})
            </h4>

            <div className="space-y-2.5 max-h-60 overflow-y-auto pr-1">
              {communityList.length === 0 ? (
                <p className="text-xs text-slate-500 text-center py-6">No community notes yet. Be the first to share!</p>
              ) : (
                communityList.map(item => (
                  <div key={item.id} className="bg-[#070b14] p-3.5 rounded-2xl border border-slate-800/80 space-y-1.5">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-bold text-slate-200">{item.user_name}</span>
                      <div className="flex items-center gap-0.5 text-amber-400">
                        {Array.from({ length: item.rating }).map((_, i) => (
                          <Star key={i} className="w-3 h-3 fill-amber-400" />
                        ))}
                      </div>
                    </div>
                    <p className="text-xs text-slate-300 leading-relaxed">{item.comment}</p>
                    <div className="flex items-center justify-between text-[10px] text-slate-500 pt-1 border-t border-slate-800/40">
                      <span className="capitalize text-cyan-400/90 font-medium">{item.category.replace('_', ' ')}</span>
                      <span className="font-mono">{new Date(item.received_at).toLocaleDateString()}</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

        </div>

      </div>
    </div>
  );
}
