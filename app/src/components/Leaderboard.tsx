import { useState, useEffect } from 'react';
import { getModelMetadata } from '../services/api';
import type { ModelMetadata } from '../types';
import { 
  Trophy, 
  Award, 
  Cpu, 
  CheckCircle2, 
  Flame, 
  Shield
} from 'lucide-react';

export default function Leaderboard() {
  const [metadata, setMetadata] = useState<ModelMetadata | null>(null);

  useEffect(() => {
    getModelMetadata().then(setMetadata);
  }, []);

  const leaderboard = metadata?.leaderboard || [];
  const topFeatures = metadata?.top_features || [];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      
      {/* Header */}
      <div className="glass-panel p-5 rounded-3xl border border-slate-800/80 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-amber-500/10 text-amber-400 border border-amber-500/20">
              <Trophy className="w-4 h-4" />
            </span>
            <h2 className="text-lg sm:text-xl font-heading font-extrabold text-slate-100">
              Model Benchmark Telemetry & Leaderboard
            </h2>
          </div>
          <p className="text-xs text-slate-400 mt-1 max-w-2xl">
            Empirical out-of-fold evaluations against held-out chronological test sets (365 days) with strict zero-temporal-leakage guarantees.
          </p>
        </div>

        <div className="flex items-center gap-2.5 text-xs text-slate-400">
          <span className="px-3 py-1.5 bg-[#030712] border border-slate-800 rounded-xl font-medium">
            Test Horizon: 365 Days
          </span>
          <span className="px-3 py-1.5 bg-cyan-950/80 text-cyan-400 border border-cyan-800/80 rounded-xl font-mono font-bold">
            v{metadata?.version || '0.2.0'}
          </span>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Leaderboard Table (8 cols) */}
        <div className="lg:col-span-8 glass-panel p-6 rounded-3xl border border-slate-800 space-y-4 shadow-xl">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-heading font-bold text-slate-200 flex items-center gap-2">
              <Award className="w-4 h-4 text-cyan-400" />
              Candidate Regressor Performance Comparison
            </h3>
            <span className="text-[11px] text-slate-400">Lower RMSE is better</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-[#030712] text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800/80">
                <tr>
                  <th className="px-4 py-3.5 font-bold rounded-l-xl">Rank & Model</th>
                  <th className="px-3 py-3.5 font-bold">Family</th>
                  <th className="px-3 py-3.5 font-bold">RMSE ↓</th>
                  <th className="px-3 py-3.5 font-bold">MAE ↓</th>
                  <th className="px-3 py-3.5 font-bold">R² Score ↑</th>
                  <th className="px-3 py-3.5 font-bold rounded-r-xl">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-300">
                {leaderboard.map((row, idx) => {
                  const isTop = idx === 0;
                  return (
                    <tr key={row.model} className={isTop ? 'bg-cyan-500/5 font-semibold' : 'hover:bg-slate-800/20'}>
                      <td className="px-4 py-3.5 flex items-center gap-2.5">
                        {isTop ? (
                          <span className="w-6 h-6 rounded-full bg-amber-400/20 text-amber-300 border border-amber-400/50 flex items-center justify-center font-bold text-xs shadow-sm shadow-amber-400/30">
                            1
                          </span>
                        ) : (
                          <span className="w-6 h-6 rounded-full bg-slate-900 text-slate-400 border border-slate-800 flex items-center justify-center font-medium text-xs">
                            {idx + 1}
                          </span>
                        )}
                        <span className={isTop ? 'text-cyan-300 font-bold' : 'text-slate-200'}>
                          {row.model}
                        </span>
                      </td>
                      <td className="px-3 py-3.5 text-slate-400">{row.family}</td>
                      <td className="px-3 py-3.5 font-mono font-bold text-emerald-400">{row.rmse.toFixed(2)}</td>
                      <td className="px-3 py-3.5 font-mono text-slate-300">{row.mae.toFixed(2)}</td>
                      <td className="px-3 py-3.5 font-mono text-cyan-400">{(row.r2 * 100).toFixed(1)}%</td>
                      <td className="px-3 py-3.5">
                        <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold border ${
                          row.status === 'Production'
                            ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                            : row.status === 'Candidate'
                            ? 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30'
                            : 'bg-slate-900 text-slate-400 border-slate-800'
                        }`}>
                          {row.status}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Feature Importance Column (4 cols) */}
        <div className="lg:col-span-4 glass-panel p-6 rounded-3xl border border-slate-800 space-y-4 shadow-xl">
          <h3 className="text-sm font-heading font-bold text-slate-200 flex items-center gap-2">
            <Cpu className="w-4 h-4 text-cyan-400" />
            Causal Feature Importances
          </h3>
          <p className="text-[11px] text-slate-400">
            Multi-model stability selection and global TreeSHAP rankings.
          </p>

          <div className="space-y-3.5 pt-2">
            {topFeatures.map((f, i) => (
              <div key={f.feature} className="space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 text-[11px] font-medium truncate max-w-[200px]" title={f.feature}>
                    {i + 1}. {f.feature}
                  </span>
                  <span className="text-cyan-400 font-mono font-bold">{(f.importance * 100).toFixed(0)}%</span>
                </div>
                <div className="w-full bg-[#030712] h-2 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full"
                    style={{ width: `${f.importance * 100 * 3.5}%` }}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="pt-4 border-t border-slate-800/80 space-y-2.5 text-[11px] text-slate-400">
            <div className="flex items-center gap-2.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
              <span>Zero-leakage chronological split validation</span>
            </div>
            <div className="flex items-center gap-2.5">
              <Flame className="w-4 h-4 text-amber-400 shrink-0" />
              <span>Optuna Bayesian hyperparameter search (100+ trials)</span>
            </div>
            <div className="flex items-center gap-2.5">
              <Shield className="w-4 h-4 text-sky-400 shrink-0" />
              <span>Autoregressive error correction on ensemble residuals</span>
            </div>
          </div>
        </div>

      </div>

    </div>
  );
}
