import { useState, useEffect } from 'react';
import { runSimulation } from '../services/api';
import type { SimulationRequest, SimulationResponse, Hemisphere } from '../types';
import { 
  CloudRain, 
  Play, 
  Calendar, 
  Activity, 
  Droplet, 
  Flame, 
  ShieldAlert, 
  SunMedium,
  TrendingDown,
  BarChart3,
  Compass
} from 'lucide-react';

export default function SimulationStudio() {
  const [params, setParams] = useState<SimulationRequest>({
    hemisphere: 'north',
    days: 90,
    scenario: 'baseline',
  });

  const [result, setResult] = useState<SimulationResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const executeSimulation = async () => {
    setLoading(true);
    const res = await runSimulation(params);
    setResult(res);
    setLoading(false);
  };

  useEffect(() => {
    executeSimulation();
  }, [params]);

  const timeline = result?.timeline || [];
  const svgWidth = 720;
  const svgHeight = 240;
  const padding = 35;

  const maxRain = Math.max(10, ...timeline.map(t => t.rainfall_mm));
  const minWqi = Math.min(20, ...timeline.map(t => t.wqi_score));
  const maxWqi = 100;

  const getX = (index: number) => padding + (index / Math.max(1, timeline.length - 1)) * (svgWidth - 2 * padding);
  const getYWqi = (val: number) => (svgHeight - padding) - ((val - minWqi) / (maxWqi - minWqi)) * (svgHeight - 2 * padding);

  const wqiPath = timeline.map((t, i) => `${i === 0 ? 'M' : 'L'} ${getX(i)} ${getYWqi(t.wqi_score)}`).join(' ');

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      
      {/* Header */}
      <div className="glass-panel p-5 rounded-3xl border border-slate-800/80 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              <CloudRain className="w-4 h-4" />
            </span>
            <h2 className="text-lg sm:text-xl font-heading font-extrabold text-slate-100">
              Physical Hydrological Climate Simulator
            </h2>
          </div>
          <p className="text-xs text-slate-400 mt-1 max-w-2xl">
            Simulate Markov-chain stochastic rainfall, SCS Curve Number runoff discharge, and first-flush contaminant washoff kinetics across multi-month timelines.
          </p>
        </div>

        <button
          onClick={executeSimulation}
          disabled={loading}
          className="flex items-center justify-center gap-2 px-5 py-2.5 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white text-xs font-bold rounded-2xl shadow-lg shadow-cyan-500/25 transition-all cursor-pointer disabled:opacity-50"
        >
          <Play className="w-4 h-4 fill-current" />
          {loading ? 'Simulating Dynamic Scenario...' : 'Run Scenario Simulation'}
        </button>
      </div>

      {/* Scenario Controls */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        
        {/* Scenario Selection */}
        <div className="glass-panel p-5 rounded-3xl border border-slate-800 space-y-3">
          <label className="text-xs font-heading font-bold text-slate-200 uppercase tracking-wider">Macro Scenario</label>
          <div className="grid grid-cols-2 gap-2">
            {[
              { id: 'baseline', label: 'Baseline', icon: SunMedium },
              { id: 'intense_storm', label: 'Storm Surge', icon: Droplet },
              { id: 'drought_heatwave', label: 'Drought Spike', icon: Flame },
              { id: 'conservation_mandate', label: 'Mandate', icon: ShieldAlert },
            ].map(s => {
              const Icon = s.icon;
              return (
                <button
                  key={s.id}
                  onClick={() => setParams(p => ({ ...p, scenario: s.id as any }))}
                  className={`flex items-center gap-2 p-2.5 text-xs font-semibold rounded-xl border transition-all cursor-pointer ${
                    params.scenario === s.id
                      ? 'bg-cyan-500/20 border-cyan-400 text-cyan-200 shadow-sm'
                      : 'bg-[#070b14] border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{s.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Time Horizon */}
        <div className="glass-panel p-5 rounded-3xl border border-slate-800 space-y-3">
          <label className="text-xs font-heading font-bold text-slate-200 uppercase tracking-wider flex items-center gap-1.5">
            <Calendar className="w-4 h-4 text-cyan-400" />
            Simulation Horizon
          </label>
          <div className="flex gap-2">
            {[30, 60, 90, 180, 365].map(d => (
              <button
                key={d}
                onClick={() => setParams(p => ({ ...p, days: d }))}
                className={`flex-1 py-2 text-xs font-mono font-bold rounded-xl border transition-all cursor-pointer ${
                  params.days === d
                    ? 'bg-cyan-500 text-white border-cyan-400 shadow-sm'
                    : 'bg-[#070b14] text-slate-400 border-slate-800 hover:text-slate-200'
                }`}
              >
                {d}d
              </button>
            ))}
          </div>
        </div>

        {/* Hemisphere Selector */}
        <div className="glass-panel p-5 rounded-3xl border border-slate-800 space-y-3">
          <label className="text-xs font-heading font-bold text-slate-200 uppercase tracking-wider flex items-center gap-1.5">
            <Compass className="w-4 h-4 text-cyan-400" />
            Geographic Hemisphere
          </label>
          <div className="grid grid-cols-2 gap-2">
            {(['north', 'south'] as Hemisphere[]).map(h => (
              <button
                key={h}
                onClick={() => setParams(p => ({ ...p, hemisphere: h }))}
                className={`py-2 text-xs font-semibold rounded-xl border capitalize transition-all cursor-pointer ${
                  params.hemisphere === h
                    ? 'bg-cyan-500 text-white border-cyan-400 shadow-sm'
                    : 'bg-[#070b14] text-slate-400 border-slate-800 hover:text-slate-200'
                }`}
              >
                {h}ern
              </button>
            ))}
          </div>
        </div>

      </div>

      {/* Summary KPI Cards */}
      {result && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="glass-panel p-5 rounded-3xl border border-slate-800 hover:border-cyan-500/40 transition-colors">
            <span className="text-[11px] font-heading font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
              <Activity className="w-3.5 h-3.5 text-cyan-400" />
              Average Projected WQI
            </span>
            <div className="text-3xl font-heading font-extrabold font-mono text-cyan-400 mt-1.5">
              {result.summary.avg_wqi.toFixed(1)}
            </div>
            <span className="text-[10px] text-slate-400">Mean across {params.days} days</span>
          </div>

          <div className="glass-panel p-5 rounded-3xl border border-slate-800 hover:border-rose-500/40 transition-colors">
            <span className="text-[11px] font-heading font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
              <TrendingDown className="w-3.5 h-3.5 text-rose-400" />
              Critical WQI Trough
            </span>
            <div className="text-3xl font-heading font-extrabold font-mono text-rose-400 mt-1.5">
              {result.summary.min_wqi.toFixed(1)}
            </div>
            <span className="text-[10px] text-slate-400">Peak environmental stress</span>
          </div>

          <div className="glass-panel p-5 rounded-3xl border border-slate-800 hover:border-sky-500/40 transition-colors">
            <span className="text-[11px] font-heading font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
              <Droplet className="w-3.5 h-3.5 text-sky-400" />
              Total Precipitation
            </span>
            <div className="text-3xl font-heading font-extrabold font-mono text-sky-400 mt-1.5">
              {result.summary.total_rainfall_mm.toFixed(1)} mm
            </div>
            <span className="text-[10px] text-slate-400">Cumulative basin rainfall</span>
          </div>

          <div className="glass-panel p-5 rounded-3xl border border-slate-800 hover:border-amber-500/40 transition-colors">
            <span className="text-[11px] font-heading font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
              <BarChart3 className="w-3.5 h-3.5 text-amber-400" />
              Storm Inundations
            </span>
            <div className="text-3xl font-heading font-extrabold font-mono text-amber-400 mt-1.5">
              {result.summary.storm_events}
            </div>
            <span className="text-[10px] text-slate-400">Events &gt; 25mm washoff</span>
          </div>
        </div>
      )}

      {/* SVG Interactive Multi-Layer Chart */}
      <div className="glass-panel p-6 rounded-3xl border border-slate-800 space-y-4 shadow-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <h3 className="text-sm font-heading font-bold text-slate-200 flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            Projected WQI Trajectory & Precipitation Inundation Curve
          </h3>
          <div className="flex items-center gap-5 text-xs">
            <span className="flex items-center gap-1.5 text-cyan-400 font-semibold">
              <span className="w-3 h-1 bg-cyan-400 inline-block rounded-full"></span> Water Quality (0–100)
            </span>
            <span className="flex items-center gap-1.5 text-sky-400 font-semibold">
              <span className="w-2.5 h-2.5 bg-sky-500/40 inline-block rounded-sm"></span> Daily Rainfall (mm)
            </span>
          </div>
        </div>

        <div className="w-full overflow-x-auto">
          <svg className="w-full h-64" viewBox={`0 0 ${svgWidth} ${svgHeight}`}>
            {/* Gridlines */}
            {[25, 50, 75, 100].map(level => {
              const y = getYWqi(level);
              return (
                <g key={level}>
                  <line x1={padding} y1={y} x2={svgWidth - padding} y2={y} stroke="#1e293b" strokeWidth="1" strokeDasharray="4 4" />
                  <text x={padding - 8} y={y + 3} fill="#64748b" fontSize="10" fontFamily="Fira Code" textAnchor="end">{level}</text>
                </g>
              );
            })}

            {/* Rainfall Inundation Bars */}
            {timeline.map((t, i) => {
              if (t.rainfall_mm <= 0) return null;
              const x = getX(i);
              const barHeight = (t.rainfall_mm / maxRain) * (svgHeight - 2 * padding) * 0.45;
              const y = svgHeight - padding - barHeight;
              return (
                <rect
                  key={i}
                  x={x - 1.5}
                  y={y}
                  width="3"
                  height={barHeight}
                  fill="rgba(56, 189, 248, 0.4)"
                  className="transition-all hover:fill-sky-400"
                />
              );
            })}

            {/* Continuous WQI Curve */}
            <path
              d={wqiPath}
              fill="none"
              stroke="#06B6D4"
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="drop-shadow-[0_0_8px_rgba(6,182,212,0.6)]"
            />
          </svg>
        </div>
      </div>

    </div>
  );
}
