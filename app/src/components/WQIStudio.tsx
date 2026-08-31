import { useState, useEffect } from 'react';
import { predictWQI } from '../services/api';
import type { WQIPredictionRequest, WQIPredictionResponse, Hemisphere, ModelFamily } from '../types';
import { 
  Sparkles, 
  AlertTriangle, 
  CheckCircle2, 
  TrendingUp, 
  TrendingDown, 
  Cpu, 
  SlidersHorizontal,
  Info,
  Layers,
  CloudSun,
  CloudLightning,
  Flame,
  Droplet,
  Thermometer,
  Compass
} from 'lucide-react';

export default function WQIStudio() {
  const [params, setParams] = useState<WQIPredictionRequest>({
    hemisphere: 'north',
    daily_rainfall_mm: 8.5,
    daily_max_temp: 26.0,
    antecedent_moisture_condition: 28.0,
    total_suspended_solids_mg_l: 45.0,
    nutrient_load_index: 25.0,
    consecutive_dry_days: 2,
    consumer_demand_cluster: 1,
    day_of_year: 180,
    include_shap: true,
    model_family: 'ensemble',
  });

  const [result, setResult] = useState<WQIPredictionResponse | null>(null);

  useEffect(() => {
    let isMounted = true;
    const fetchPrediction = async () => {
      const res = await predictWQI(params);
      if (isMounted) {
        setResult(res);
      }
    };
    fetchPrediction();
    return () => { isMounted = false; };
  }, [params]);

  const setPreset = (preset: Partial<WQIPredictionRequest>) => {
    setParams(prev => ({ ...prev, ...preset }));
  };

  const getScoreTheme = (score: number) => {
    if (score >= 85) return { stroke: '#10B981', text: 'text-emerald-400', bg: 'bg-emerald-500/10 border-emerald-500/30', glow: 'shadow-emerald-500/20' };
    if (score >= 70) return { stroke: '#06B6D4', text: 'text-cyan-400', bg: 'bg-cyan-500/10 border-cyan-500/30', glow: 'shadow-cyan-500/20' };
    if (score >= 50) return { stroke: '#F59E0B', text: 'text-amber-400', bg: 'bg-amber-500/10 border-amber-500/30', glow: 'shadow-amber-500/20' };
    if (score >= 30) return { stroke: '#F97316', text: 'text-orange-400', bg: 'bg-orange-500/10 border-orange-500/30', glow: 'shadow-orange-500/20' };
    return { stroke: '#F43F5E', text: 'text-rose-400', bg: 'bg-rose-500/10 border-rose-500/30', glow: 'shadow-rose-500/20' };
  };

  const scoreTheme = getScoreTheme(result?.wqi_score ?? 75);

  // SVG Gauge calculations
  const radius = 72;
  const circumference = 2 * Math.PI * radius;
  const scorePercent = Math.min(1, Math.max(0, (result?.wqi_score ?? 75) / 100));
  const strokeDashoffset = circumference - (scorePercent * 0.75 * circumference);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      
      {/* Top Banner & Quick Presets */}
      <div className="glass-panel p-5 rounded-3xl border border-slate-800/80 flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              <Sparkles className="w-4 h-4" />
            </span>
            <h2 className="text-lg sm:text-xl font-heading font-extrabold text-slate-100">
              Water Quality Index (WQI) Forecasting Studio
            </h2>
          </div>
          <p className="text-xs text-slate-400 mt-1 max-w-2xl">
            Multi-model stacking regressor with Autoregressive AR(1) residual correction, physical SCS runoff kinetics, and local SHAP decomposition.
          </p>
        </div>

        {/* Climate Presets */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-slate-400 font-semibold uppercase tracking-wider">Presets:</span>
          
          <button
            onClick={() => setPreset({ daily_rainfall_mm: 2.5, daily_max_temp: 22.0, total_suspended_solids_mg_l: 22.0, nutrient_load_index: 12.0, consecutive_dry_days: 1 })}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-900 hover:bg-slate-800 text-slate-200 text-xs font-semibold rounded-xl border border-slate-700/80 transition-all cursor-pointer hover:border-slate-500"
          >
            <CloudSun className="w-3.5 h-3.5 text-amber-400" />
            Baseline Equilibrium
          </button>
          
          <button
            onClick={() => setPreset({ daily_rainfall_mm: 72.0, daily_max_temp: 20.5, antecedent_moisture_condition: 82.0, total_suspended_solids_mg_l: 260.0, nutrient_load_index: 92.0, consecutive_dry_days: 0 })}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-cyan-950/60 hover:bg-cyan-900/60 text-cyan-300 text-xs font-semibold rounded-xl border border-cyan-800/80 transition-all cursor-pointer hover:border-cyan-500"
          >
            <CloudLightning className="w-3.5 h-3.5 text-cyan-400" />
            Storm Surge Washoff
          </button>
          
          <button
            onClick={() => setPreset({ daily_rainfall_mm: 0.0, daily_max_temp: 39.0, antecedent_moisture_condition: 10.0, total_suspended_solids_mg_l: 85.0, nutrient_load_index: 78.0, consecutive_dry_days: 24 })}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-rose-950/60 hover:bg-rose-900/60 text-rose-300 text-xs font-semibold rounded-xl border border-rose-800/80 transition-all cursor-pointer hover:border-rose-500"
          >
            <Flame className="w-3.5 h-3.5 text-rose-400" />
            Heatwave Drought
          </button>
        </div>
      </div>

      {/* Main Grid: Controls on Left, Live Output on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Controls Column (7 cols) */}
        <div className="lg:col-span-7 glass-panel p-6 rounded-3xl border border-slate-800 space-y-6">
          
          {/* Header & Hemisphere / Model Selector */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-800/80">
            <h3 className="text-sm font-heading font-bold text-slate-200 flex items-center gap-2">
              <SlidersHorizontal className="w-4 h-4 text-cyan-400" />
              Physical & Environmental Parameters
            </h3>
            
            <div className="flex items-center gap-2.5">
              {/* Hemisphere Switcher */}
              <div className="flex items-center bg-[#030712] p-1 rounded-xl border border-slate-800 text-xs">
                <button
                  onClick={() => setParams(p => ({ ...p, hemisphere: 'north' as Hemisphere }))}
                  className={`px-3 py-1 rounded-lg font-semibold transition-all cursor-pointer ${
                    params.hemisphere === 'north' ? 'bg-cyan-500 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  North
                </button>
                <button
                  onClick={() => setParams(p => ({ ...p, hemisphere: 'south' as Hemisphere }))}
                  className={`px-3 py-1 rounded-lg font-semibold transition-all cursor-pointer ${
                    params.hemisphere === 'south' ? 'bg-cyan-500 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  South
                </button>
              </div>

              {/* Model Family Dropdown */}
              <div className="relative">
                <select
                  value={params.model_family}
                  onChange={e => setParams(p => ({ ...p, model_family: e.target.value as ModelFamily }))}
                  className="bg-[#030712] text-slate-200 border border-slate-800 rounded-xl px-3 py-1.5 text-xs font-semibold focus:outline-none focus:border-cyan-500 cursor-pointer"
                >
                  <option value="ensemble">Stacked Ensemble + AR(1)</option>
                  <option value="lightgbm">LightGBM Regressor</option>
                  <option value="xgboost">XGBoost Regressor</option>
                  <option value="catboost">CatBoost Regressor</option>
                  <option value="linear">Ridge Regularized Linear</option>
                </select>
              </div>
            </div>
          </div>

          {/* Interactive Sliders Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            
            {/* Daily Rainfall */}
            <div className="bg-[#070b14] p-4 rounded-2xl border border-slate-800/80 space-y-2 hover:border-slate-700 transition-colors">
              <div className="flex justify-between items-center text-xs">
                <span className="text-slate-300 font-semibold flex items-center gap-1.5">
                  <Droplet className="w-3.5 h-3.5 text-cyan-400" />
                  Daily Precipitation
                </span>
                <span className="text-cyan-400 font-mono font-bold bg-cyan-950/80 px-2 py-0.5 rounded border border-cyan-800/60">
                  {params.daily_rainfall_mm} mm
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="120"
                step="0.5"
                value={params.daily_rainfall_mm}
                onChange={e => setParams(p => ({ ...p, daily_rainfall_mm: parseFloat(e.target.value) }))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
              />
            </div>

            {/* Daily Max Temp */}
            <div className="bg-[#070b14] p-4 rounded-2xl border border-slate-800/80 space-y-2 hover:border-slate-700 transition-colors">
              <div className="flex justify-between items-center text-xs">
                <span className="text-slate-300 font-semibold flex items-center gap-1.5">
                  <Thermometer className="w-3.5 h-3.5 text-amber-400" />
                  Max Ambient Temp
                </span>
                <span className="text-amber-400 font-mono font-bold bg-amber-950/80 px-2 py-0.5 rounded border border-amber-800/60">
                  {params.daily_max_temp} °C
                </span>
              </div>
              <input
                type="range"
                min="5"
                max="45"
                step="0.5"
                value={params.daily_max_temp}
                onChange={e => setParams(p => ({ ...p, daily_max_temp: parseFloat(e.target.value) }))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-400"
              />
            </div>

            {/* Total Suspended Solids */}
            <div className="bg-[#070b14] p-4 rounded-2xl border border-slate-800/80 space-y-2 hover:border-slate-700 transition-colors">
              <div className="flex justify-between items-center text-xs">
                <span className="text-slate-300 font-semibold flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-orange-400" />
                  Total Suspended Solids (TSS)
                </span>
                <span className="text-orange-400 font-mono font-bold bg-orange-950/80 px-2 py-0.5 rounded border border-orange-800/60">
                  {params.total_suspended_solids_mg_l} mg/L
                </span>
              </div>
              <input
                type="range"
                min="10"
                max="350"
                step="1"
                value={params.total_suspended_solids_mg_l}
                onChange={e => setParams(p => ({ ...p, total_suspended_solids_mg_l: parseFloat(e.target.value) }))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-orange-400"
              />
            </div>

            {/* Nutrient & Biological Load */}
            <div className="bg-[#070b14] p-4 rounded-2xl border border-slate-800/80 space-y-2 hover:border-slate-700 transition-colors">
              <div className="flex justify-between items-center text-xs">
                <span className="text-slate-300 font-semibold flex items-center gap-1.5">
                  <Flame className="w-3.5 h-3.5 text-rose-400" />
                  Nutrient & Bio Index
                </span>
                <span className="text-rose-400 font-mono font-bold bg-rose-950/80 px-2 py-0.5 rounded border border-rose-800/60">
                  {params.nutrient_load_index}
                </span>
              </div>
              <input
                type="range"
                min="5"
                max="120"
                step="1"
                value={params.nutrient_load_index}
                onChange={e => setParams(p => ({ ...p, nutrient_load_index: parseFloat(e.target.value) }))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-rose-400"
              />
            </div>

            {/* Antecedent Moisture Condition */}
            <div className="bg-[#070b14] p-4 rounded-2xl border border-slate-800/80 space-y-2 hover:border-slate-700 transition-colors">
              <div className="flex justify-between items-center text-xs">
                <span className="text-slate-300 font-semibold flex items-center gap-1.5">
                  <Compass className="w-3.5 h-3.5 text-sky-400" />
                  Antecedent 5-Day Moisture
                </span>
                <span className="text-sky-400 font-mono font-bold bg-sky-950/80 px-2 py-0.5 rounded border border-sky-800/60">
                  {params.antecedent_moisture_condition} %
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                step="1"
                value={params.antecedent_moisture_condition}
                onChange={e => setParams(p => ({ ...p, antecedent_moisture_condition: parseFloat(e.target.value) }))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-400"
              />
            </div>

            {/* Consecutive Dry Days */}
            <div className="bg-[#070b14] p-4 rounded-2xl border border-slate-800/80 space-y-2 hover:border-slate-700 transition-colors">
              <div className="flex justify-between items-center text-xs">
                <span className="text-slate-300 font-semibold flex items-center gap-1.5">
                  <Flame className="w-3.5 h-3.5 text-indigo-400" />
                  Consecutive Dry Days
                </span>
                <span className="text-indigo-400 font-mono font-bold bg-indigo-950/80 px-2 py-0.5 rounded border border-indigo-800/60">
                  {params.consecutive_dry_days} days
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="30"
                step="1"
                value={params.consecutive_dry_days}
                onChange={e => setParams(p => ({ ...p, consecutive_dry_days: parseInt(e.target.value) }))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-400"
              />
            </div>

          </div>

          {/* Consumer Archetype Segment Selector */}
          <div className="bg-[#070b14] p-4 rounded-2xl border border-slate-800/80 space-y-3">
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-300 font-semibold">Consumer Behavioral Pressure Archetype</span>
              <span className="text-cyan-400 font-bold">
                {params.consumer_demand_cluster === 0 && 'Cluster 0: Low-Volume Conservationist'}
                {params.consumer_demand_cluster === 1 && 'Cluster 1: Standard Suburban Consumer'}
                {params.consumer_demand_cluster === 2 && 'Cluster 2: Landscape Irrigation Heavy'}
                {params.consumer_demand_cluster === 3 && 'Cluster 3: High-Volume Commercial/Multi'}
              </span>
            </div>
            <div className="grid grid-cols-4 gap-2">
              {[0, 1, 2, 3].map(c => (
                <button
                  key={c}
                  onClick={() => setParams(p => ({ ...p, consumer_demand_cluster: c }))}
                  className={`py-2 text-xs font-bold rounded-xl border transition-all cursor-pointer ${
                    params.consumer_demand_cluster === c
                      ? 'bg-cyan-500/25 border-cyan-400 text-cyan-200 shadow-md shadow-cyan-500/10'
                      : 'bg-[#030712] border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700'
                  }`}
                >
                  Cluster {c}
                </button>
              ))}
            </div>
          </div>

        </div>

        {/* Live Output & SHAP Waterfall Column (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          
          {/* Main Score Card & Radial Gauge */}
          <div className="glass-panel p-6 rounded-3xl border border-slate-800 flex flex-col items-center justify-center text-center relative overflow-hidden shadow-xl">
            <div className="absolute inset-0 bg-gradient-to-b from-cyan-500/10 via-transparent to-transparent pointer-events-none" />
            
            {/* SVG Circular Gauge */}
            <div className="relative w-48 h-48 flex items-center justify-center">
              <svg className="w-full h-full transform -rotate-90" viewBox="0 0 170 170">
                {/* Background Ring */}
                <circle
                  cx="85"
                  cy="85"
                  r={radius}
                  stroke="#1e293b"
                  strokeWidth="14"
                  fill="transparent"
                  strokeDasharray={`${circumference * 0.75} ${circumference * 0.25}`}
                  strokeLinecap="round"
                />
                {/* Score Arc */}
                <circle
                  cx="85"
                  cy="85"
                  r={radius}
                  stroke={scoreTheme.stroke}
                  strokeWidth="14"
                  fill="transparent"
                  strokeDasharray={`${circumference * 0.75} ${circumference * 0.25}`}
                  strokeDashoffset={strokeDashoffset}
                  strokeLinecap="round"
                  className="transition-all duration-700 ease-out"
                />
              </svg>

              {/* Centered Score */}
              <div className="absolute flex flex-col items-center justify-center">
                <span className={`text-5xl font-heading font-extrabold font-mono tracking-tight ${scoreTheme.text}`}>
                  {result ? result.wqi_score.toFixed(1) : '--'}
                </span>
                <span className="text-[10px] text-slate-400 uppercase tracking-widest font-bold mt-1">
                  WQI Index
                </span>
              </div>
            </div>

            {/* Classification & Confidence Interval */}
            <div className="mt-3 flex items-center gap-2.5">
              <span className={`px-4 py-1 rounded-full text-xs font-extrabold border ${scoreTheme.bg} ${scoreTheme.text} ${scoreTheme.glow}`}>
                {result?.classification || 'Evaluating'}
              </span>
              {result && (
                <span className="text-xs text-slate-400 font-mono font-medium">
                  95% CI: [{result.confidence_interval[0]}, {result.confidence_interval[1]}]
                </span>
              )}
            </div>

            <div className="mt-2.5 text-[11px] text-slate-400 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              <span>{result?.model_used}</span>
            </div>
          </div>

          {/* Real-Time SHAP Waterfall Contribution */}
          <div className="glass-panel p-5 rounded-3xl border border-slate-800 space-y-3.5">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-heading font-bold text-slate-200 uppercase tracking-wider flex items-center gap-1.5">
                <Compass className="w-4 h-4 text-cyan-400" />
                Live SHAP Driver Decomposition
              </h4>
              <span className="text-[10px] text-slate-400 font-mono">Ref: 92.0 pts</span>
            </div>
            
            <p className="text-[11px] text-slate-400 leading-relaxed">
              Marginal point impact on Water Quality Index deviation from baseline pristine conditions:
            </p>

            <div className="space-y-2.5 pt-1">
              {result?.shap_breakdown && Object.entries(result.shap_breakdown).map(([feature, val]) => {
                const isPositive = val >= 0;
                const pct = Math.min(100, Math.abs(val) * 4.0);
                return (
                  <div key={feature} className="space-y-1">
                    <div className="flex justify-between text-xs font-medium">
                      <span className="text-slate-300 flex items-center gap-1.5 text-[11px]">
                        {isPositive ? <TrendingUp className="w-3.5 h-3.5 text-emerald-400" /> : <TrendingDown className="w-3.5 h-3.5 text-rose-400" />}
                        {feature}
                      </span>
                      <span className={`font-mono text-xs font-bold ${isPositive ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {isPositive ? `+${val.toFixed(2)}` : val.toFixed(2)} pts
                      </span>
                    </div>
                    {/* Bar visualizer */}
                    <div className="w-full bg-[#030712] h-2 rounded-full overflow-hidden flex">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          isPositive
                            ? 'bg-gradient-to-r from-emerald-500 to-teal-400 shadow-sm shadow-emerald-500/50'
                            : 'bg-gradient-to-r from-rose-500 to-red-600 shadow-sm shadow-rose-500/50'
                        }`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Operational Advisories */}
          <div className="glass-panel p-5 rounded-3xl border border-slate-800 space-y-2.5">
            <h4 className="text-xs font-heading font-bold text-slate-200 uppercase tracking-wider flex items-center gap-1.5">
              <Info className="w-4 h-4 text-cyan-400" />
              Treatment & Operational Advisories
            </h4>
            <div className="space-y-2">
              {result?.advisories.map((adv, idx) => (
                <div key={idx} className="flex items-start gap-2.5 text-xs text-slate-300 bg-[#070b14] p-2.5 rounded-xl border border-slate-800/80">
                  {adv.includes('detected') || adv.includes('hazard') || adv.includes('surge') ? (
                    <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                  ) : (
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                  )}
                  <span className="leading-relaxed">{adv}</span>
                </div>
              ))}
            </div>
          </div>

        </div>

      </div>

    </div>
  );
}
