import { useEffect, useState } from 'react';
import { subscribeCloudStatus } from '../services/api';
import type { CloudStatus } from '../types';
import { 
  Waves, 
  Users, 
  CloudRain, 
  Trophy, 
  MessageSquareHeart,
  Radio,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';

export type ActiveTab = 'wqi' | 'cluster' | 'simulation' | 'leaderboard';

interface NavbarProps {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  onOpenFeedback: () => void;
}

export default function Navbar({ activeTab, setActiveTab, onOpenFeedback }: NavbarProps) {
  const [status, setStatus] = useState<CloudStatus>('waking_up');

  useEffect(() => {
    return subscribeCloudStatus(setStatus);
  }, []);

  const navItems: Array<{ id: ActiveTab; label: string; icon: typeof Waves; subtitle: string }> = [
    { id: 'wqi', label: 'WQI Mission Control', icon: Waves, subtitle: 'Physical & Supervised Forecasting' },
    { id: 'cluster', label: 'Behavioral Clusters', icon: Users, subtitle: 'Unsupervised Consumer Science' },
    { id: 'simulation', label: 'Climate Simulator', icon: CloudRain, subtitle: 'Markov & SCS Runoff Trace' },
    { id: 'leaderboard', label: 'Model Leaderboard', icon: Trophy, subtitle: 'Telemetry & Benchmarks' },
  ];

  return (
    <header className="w-full bg-[#070b14]/90 backdrop-blur-xl border-b border-slate-800/80 sticky top-0 z-50 transition-colors">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-18 flex items-center justify-between">
        
        {/* Brand Identity */}
        <div className="flex items-center gap-3.5">
          <div className="relative group cursor-pointer">
            <div className="absolute -inset-1 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-2xl blur opacity-30 group-hover:opacity-60 transition duration-300"></div>
            <div className="relative w-11 h-11 rounded-xl bg-gradient-to-b from-slate-900 to-slate-950 border border-cyan-500/40 flex items-center justify-center shadow-lg shadow-cyan-950">
              <Waves className="w-6 h-6 text-cyan-400 stroke-[2.2]" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-heading font-extrabold text-xl tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-sky-300 to-indigo-300">
                Hydroloom AI
              </span>
              <span className="px-2 py-0.5 text-[10px] font-mono font-bold bg-cyan-950/80 text-cyan-400 border border-cyan-800/70 rounded-md tracking-wider">
                PROD v0.2.0
              </span>
            </div>
            <p className="text-[11px] text-slate-400 font-medium hidden sm:block">
              Continuous Water Quality Index Intelligence & Hydro-Climatic Forecasting
            </p>
          </div>
        </div>

        {/* Navigation Tabs (Desktop Segmented Control) */}
        <nav className="hidden md:flex items-center gap-1.5 bg-[#030712]/90 p-1.5 rounded-2xl border border-slate-800/90 shadow-inner">
          {navItems.map(item => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`relative flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold transition-all duration-200 cursor-pointer ${
                  isActive
                    ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg shadow-cyan-500/25 border border-cyan-400/40'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-white' : 'text-cyan-400/70'}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Right Section: Status & Feedback */}
        <div className="flex items-center gap-3">
          
          {/* Cloud Pulse Status Badge */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#030712] border border-slate-800 text-xs shadow-sm">
            {status === 'connected' && (
              <>
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
                </span>
                <span className="text-emerald-400 font-semibold text-[11px] flex items-center gap-1">
                  <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                  <span className="hidden lg:inline">Render Cloud Online</span>
                </span>
              </>
            )}
            {status === 'waking_up' && (
              <>
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-amber-500"></span>
                </span>
                <span className="text-amber-400 font-semibold text-[11px] flex items-center gap-1">
                  <Radio className="w-3 h-3 text-amber-400 animate-spin" />
                  <span className="hidden lg:inline">Hybrid Engine (Warming Cloud)</span>
                </span>
              </>
            )}
            {status === 'offline_hybrid' && (
              <>
                <AlertCircle className="w-3 h-3 text-sky-400" />
                <span className="text-sky-400 font-semibold text-[11px] hidden lg:inline">
                  Local Hybrid Execution
                </span>
              </>
            )}
          </div>

          {/* Feedback Trigger Button */}
          <button
            onClick={onOpenFeedback}
            className="flex items-center gap-2 px-3.5 py-2 bg-gradient-to-r from-slate-900 to-slate-800 hover:from-slate-800 hover:to-slate-700 text-slate-100 border border-slate-700/80 rounded-xl text-xs font-bold transition-all shadow-md hover:shadow-cyan-500/10 cursor-pointer"
          >
            <MessageSquareHeart className="w-4 h-4 text-cyan-400" />
            <span className="hidden sm:inline">Community Notes</span>
          </button>
        </div>

      </div>

      {/* Mobile Navigation Bar */}
      <div className="md:hidden flex items-center gap-1.5 px-4 py-2.5 overflow-x-auto border-t border-slate-800/80 bg-[#030712]/95">
        {navItems.map(item => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold whitespace-nowrap transition-colors ${
                isActive
                  ? 'bg-cyan-500 text-white shadow-md shadow-cyan-500/20'
                  : 'text-slate-400 hover:text-slate-200 bg-slate-900/60 border border-slate-800/60'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              {item.label}
            </button>
          );
        })}
      </div>
    </header>
  );
}
