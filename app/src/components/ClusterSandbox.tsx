import { useState, useEffect } from 'react';
import { predictCluster } from '../services/api';
import type { ClusterPredictionRequest, ClusterPredictionResponse, Hemisphere } from '../types';
import { 
  Users, 
  Lightbulb, 
  Target, 
  ShieldCheck, 
  SlidersHorizontal,
  Leaf,
  Home,
  Sun,
  Building2,
  Activity,
  Layers
} from 'lucide-react';

export default function ClusterSandbox() {
  const [params, setParams] = useState<ClusterPredictionRequest>({
    hemisphere: 'north',
    log_per_capita_usage: 4.8,
    dry_day_spike_factor: 1.25,
    efficiency_penalty_ratio: 0.15,
    landscape_demand_index: 0.45,
  });

  const [result, setResult] = useState<ClusterPredictionResponse | null>(null);

  useEffect(() => {
    let isMounted = true;
    const fetchCluster = async () => {
      const res = await predictCluster(params);
      if (isMounted) {
        setResult(res);
      }
    };
    fetchCluster();
    return () => { isMounted = false; };
  }, [params]);

  const setPreset = (preset: Partial<ClusterPredictionRequest>) => {
    setParams(prev => ({ ...prev, ...preset }));
  };

  const getArchetypeIcon = (id: number) => {
    if (id === 0) return <Leaf className="w-5 h-5 text-emerald-400" />;
    if (id === 1) return <Home className="w-5 h-5 text-cyan-400" />;
    if (id === 2) return <Sun className="w-5 h-5 text-amber-400" />;
    return <Building2 className="w-5 h-5 text-rose-400" />;
  };

  const getArchetypeBadgeColor = (id: number) => {
    if (id === 0) return 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30';
    if (id === 1) return 'bg-cyan-500/10 text-cyan-300 border-cyan-500/30';
    if (id === 2) return 'bg-amber-500/10 text-amber-300 border-amber-500/30';
    return 'bg-rose-500/10 text-rose-300 border-rose-500/30';
  };

  // SVG Radar Polygon points computation
  const scores = result?.behavior_scores || {
    'Per-Capita Volume': 50,
    'Dry-Day Spike': 40,
    'Tier Penalty Adherence': 20,
    'Landscape Irrigation': 45,
  };

  const center = 100;
  const maxR = 75;
  const keys = Object.keys(scores);
  const angles = keys.map((_, i) => (i * 2 * Math.PI) / keys.length - Math.PI / 2);

  const polygonPoints = keys.map((k, i) => {
    const val = (scores[k] ?? 50) / 100;
    const r = val * maxR;
    const x = center + r * Math.cos(angles[i]);
    const y = center + r * Math.sin(angles[i]);
    return `${x},${y}`;
  }).join(' ');

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      
      {/* Header & Preset Toolbar */}
      <div className="glass-panel p-5 rounded-3xl border border-slate-800/80 flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              <Users className="w-4 h-4" />
            </span>
            <h2 className="text-lg sm:text-xl font-heading font-extrabold text-slate-100">
              Unsupervised Behavioral Clustering Studio (k=4)
            </h2>
          </div>
          <p className="text-xs text-slate-400 mt-1 max-w-2xl">
            K-Means customer demand profiling mapping consumer water use into 4 macro behavioral archetypes for precision conservation interventions.
          </p>
        </div>

        {/* Quick Presets */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-slate-400 font-semibold uppercase tracking-wider">Presets:</span>
          <button
            onClick={() => setPreset({ log_per_capita_usage: 3.8, dry_day_spike_factor: 1.02, efficiency_penalty_ratio: 0.02, landscape_demand_index: 0.12 })}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-900 hover:bg-slate-800 text-emerald-300 text-xs font-semibold rounded-xl border border-emerald-800/60 transition-all cursor-pointer hover:border-emerald-500"
          >
            <Leaf className="w-3.5 h-3.5" />
            Conservationist
          </button>
          <button
            onClick={() => setPreset({ log_per_capita_usage: 4.8, dry_day_spike_factor: 1.22, efficiency_penalty_ratio: 0.14, landscape_demand_index: 0.42 })}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-900 hover:bg-slate-800 text-cyan-300 text-xs font-semibold rounded-xl border border-cyan-800/60 transition-all cursor-pointer hover:border-cyan-500"
          >
            <Home className="w-3.5 h-3.5" />
            Average Suburban
          </button>
          <button
            onClick={() => setPreset({ log_per_capita_usage: 5.4, dry_day_spike_factor: 1.68, efficiency_penalty_ratio: 0.38, landscape_demand_index: 0.88 })}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-900 hover:bg-slate-800 text-amber-300 text-xs font-semibold rounded-xl border border-amber-800/60 transition-all cursor-pointer hover:border-amber-500"
          >
            <Sun className="w-3.5 h-3.5" />
            Landscape Irrigation Heavy
          </button>
          <button
            onClick={() => setPreset({ log_per_capita_usage: 6.2, dry_day_spike_factor: 2.15, efficiency_penalty_ratio: 0.75, landscape_demand_index: 0.65 })}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-900 hover:bg-slate-800 text-rose-300 text-xs font-semibold rounded-xl border border-rose-800/60 transition-all cursor-pointer hover:border-rose-500"
          >
            <Building2 className="w-3.5 h-3.5" />
            High-Volume Commercial
          </button>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Sliders Column (6 cols) */}
        <div className="lg:col-span-6 glass-panel p-6 rounded-3xl border border-slate-800 space-y-6">
          <div className="flex items-center justify-between pb-4 border-b border-slate-800/80">
            <h3 className="text-sm font-heading font-bold text-slate-200 flex items-center gap-2">
              <SlidersHorizontal className="w-4 h-4 text-cyan-400" />
              Engineered Behavioral Attributes
            </h3>
            
            <div className="flex items-center bg-[#030712] p-1 rounded-xl border border-slate-800 text-xs">
              <button
                onClick={() => setParams(p => ({ ...p, hemisphere: 'north' as Hemisphere }))}
                className={`px-3 py-1 rounded-lg font-semibold transition-all cursor-pointer ${
                  params.hemisphere === 'north' ? 'bg-cyan-500 text-white shadow-sm' : 'text-slate-400'
                }`}
              >
                North Centroids
              </button>
              <button
                onClick={() => setParams(p => ({ ...p, hemisphere: 'south' as Hemisphere }))}
                className={`px-3 py-1 rounded-lg font-semibold transition-all cursor-pointer ${
                  params.hemisphere === 'south' ? 'bg-cyan-500 text-white shadow-sm' : 'text-slate-400'
                }`}
              >
                South Centroids
              </button>
            </div>
          </div>

          <div className="space-y-4">
            
            {/* Log Per-Capita Volume */}
            <div className="bg-[#070b14] p-4 rounded-2xl border border-slate-800/80 space-y-2 hover:border-slate-700 transition-colors">
              <div className="flex justify-between items-center text-xs">
                <span className="text-slate-300 font-semibold flex items-center gap-1.5">
                  <Activity className="w-3.5 h-3.5 text-cyan-400" />
                  Log Per-Capita Usage
                </span>
                <span className="text-cyan-400 font-mono font-bold bg-cyan-950/80 px-2 py-0.5 rounded border border-cyan-800/60">
                  {params.log_per_capita_usage.toFixed(2)} log(L)
                </span>
              </div>
              <input
                type="range"
                min="2.5"
                max="7.0"
                step="0.05"
                value={params.log_per_capita_usage}
                onChange={e => setParams(p => ({ ...p, log_per_capita_usage: parseFloat(e.target.value) }))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
              />
              <p className="text-[11px] text-slate-400">Log-transformed daily consumption volume per occupant.</p>
            </div>

            {/* Dry-Day Demand Spike */}
            <div className="bg-[#070b14] p-4 rounded-2xl border border-slate-800/80 space-y-2 hover:border-slate-700 transition-colors">
              <div className="flex justify-between items-center text-xs">
                <span className="text-slate-300 font-semibold flex items-center gap-1.5">
                  <Sun className="w-3.5 h-3.5 text-amber-400" />
                  Dry-Day Spike Factor
                </span>
                <span className="text-amber-400 font-mono font-bold bg-amber-950/80 px-2 py-0.5 rounded border border-amber-800/60">
                  {params.dry_day_spike_factor.toFixed(2)}x
                </span>
              </div>
              <input
                type="range"
                min="0.8"
                max="3.0"
                step="0.02"
                value={params.dry_day_spike_factor}
                onChange={e => setParams(p => ({ ...p, dry_day_spike_factor: parseFloat(e.target.value) }))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-400"
              />
              <p className="text-[11px] text-slate-400">Relative surge in water consumption during drought & heat spells.</p>
            </div>

            {/* Efficiency Penalty Ratio */}
            <div className="bg-[#070b14] p-4 rounded-2xl border border-slate-800/80 space-y-2 hover:border-slate-700 transition-colors">
              <div className="flex justify-between items-center text-xs">
                <span className="text-slate-300 font-semibold flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-rose-400" />
                  Tier Penalty Adherence
                </span>
                <span className="text-rose-400 font-mono font-bold bg-rose-950/80 px-2 py-0.5 rounded border border-rose-800/60">
                  {(params.efficiency_penalty_ratio * 100).toFixed(1)}%
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="1.0"
                step="0.01"
                value={params.efficiency_penalty_ratio}
                onChange={e => setParams(p => ({ ...p, efficiency_penalty_ratio: parseFloat(e.target.value) }))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-rose-400"
              />
              <p className="text-[11px] text-slate-400">Percentage of billing periods triggering tiered penalty rates.</p>
            </div>

            {/* Landscape Demand Index */}
            <div className="bg-[#070b14] p-4 rounded-2xl border border-slate-800/80 space-y-2 hover:border-slate-700 transition-colors">
              <div className="flex justify-between items-center text-xs">
                <span className="text-slate-300 font-semibold flex items-center gap-1.5">
                  <Leaf className="w-3.5 h-3.5 text-emerald-400" />
                  Landscape Irrigation Demand
                </span>
                <span className="text-emerald-400 font-mono font-bold bg-emerald-950/80 px-2 py-0.5 rounded border border-emerald-800/60">
                  {params.landscape_demand_index.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="1.0"
                step="0.01"
                value={params.landscape_demand_index}
                onChange={e => setParams(p => ({ ...p, landscape_demand_index: parseFloat(e.target.value) }))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-400"
              />
              <p className="text-[11px] text-slate-400">Sensitivity of outdoor landscape watering to ambient weather fluctuations.</p>
            </div>

          </div>
        </div>

        {/* Results & Radar Column (6 cols) */}
        <div className="lg:col-span-6 space-y-6">
          
          {/* Classification Result Card */}
          <div className="glass-panel p-6 rounded-3xl border border-slate-800 space-y-4 shadow-xl">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-3 bg-[#030712] rounded-2xl border border-slate-800 shadow-inner">
                  {getArchetypeIcon(result?.cluster_id ?? 1)}
                </div>
                <div>
                  <span className={`px-3 py-0.5 text-xs font-bold rounded-full border ${getArchetypeBadgeColor(result?.cluster_id ?? 1)}`}>
                    Cluster {result?.cluster_id ?? 1}
                  </span>
                  <h3 className="text-base font-heading font-extrabold text-slate-100 mt-1">
                    {result?.archetype_name || 'Evaluating Archetype...'}
                  </h3>
                </div>
              </div>
            </div>

            <p className="text-xs text-slate-300 bg-[#070b14] p-3.5 rounded-2xl border border-slate-800/80 leading-relaxed">
              {result?.description}
            </p>

            {/* Dynamic SVG Radar Spider Chart */}
            <div className="bg-[#070b14] p-5 rounded-2xl border border-slate-800/80 flex flex-col items-center justify-center">
              <span className="text-[11px] text-slate-400 uppercase tracking-widest font-heading font-bold mb-2 flex items-center gap-1.5">
                <Target className="w-3.5 h-3.5 text-cyan-400" />
                4-Axis Behavioral Profile Radar
              </span>

              <svg className="w-52 h-52" viewBox="0 0 200 200">
                {/* Background Grid Rings */}
                {[0.25, 0.5, 0.75, 1.0].map((level, i) => (
                  <polygon
                    key={i}
                    points={angles.map(a => `${center + level * maxR * Math.cos(a)},${center + level * maxR * Math.sin(a)}`).join(' ')}
                    fill="none"
                    stroke="#1e293b"
                    strokeWidth="1"
                    strokeDasharray={i < 3 ? '3 3' : 'none'}
                  />
                ))}

                {/* Axes */}
                {angles.map((a, i) => (
                  <line
                    key={i}
                    x1={center}
                    y1={center}
                    x2={center + maxR * Math.cos(a)}
                    y2={center + maxR * Math.sin(a)}
                    stroke="#1e293b"
                    strokeWidth="1.5"
                  />
                ))}

                {/* Radar Polygon */}
                <polygon
                  points={polygonPoints}
                  fill="rgba(6, 182, 212, 0.2)"
                  stroke="#06B6D4"
                  strokeWidth="2.5"
                  className="transition-all duration-500"
                />

                {/* Vertex Dots */}
                {keys.map((k, i) => {
                  const val = (scores[k] ?? 50) / 100;
                  const r = val * maxR;
                  const x = center + r * Math.cos(angles[i]);
                  const y = center + r * Math.sin(angles[i]);
                  return (
                    <circle
                      key={k}
                      cx={x}
                      cy={y}
                      r="4"
                      fill="#38BDF8"
                      stroke="#030712"
                      strokeWidth="1.5"
                    />
                  );
                })}
              </svg>

              {/* Axis Legend */}
              <div className="grid grid-cols-2 gap-x-6 gap-y-1.5 text-[11px] text-slate-400 mt-2 font-medium">
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-cyan-400"></span> Top: Per-Capita Volume</span>
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-amber-400"></span> Right: Dry-Day Surge</span>
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-rose-400"></span> Bottom: Penalty Ratio</span>
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-400"></span> Left: Landscape Irrigation</span>
              </div>
            </div>

            {/* Targeted Recommendations */}
            <div className="space-y-2.5 pt-1">
              <h4 className="text-xs font-heading font-bold text-slate-200 uppercase tracking-wider flex items-center gap-1.5">
                <Lightbulb className="w-4 h-4 text-amber-400" />
                Targeted Conservation Actions & Rebates
              </h4>
              <div className="space-y-2">
                {result?.recommendations.map((rec, idx) => (
                  <div key={idx} className="flex items-start gap-2.5 text-xs text-slate-300 bg-[#070b14] p-2.5 rounded-xl border border-slate-800/80">
                    <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                    <span className="leading-relaxed">{rec}</span>
                  </div>
                ))}
              </div>
            </div>

          </div>

        </div>

      </div>

    </div>
  );
}
