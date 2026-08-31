import { useState } from 'react';
import Navbar, { type ActiveTab } from './components/Navbar';
import WQIStudio from './components/WQIStudio';
import ClusterSandbox from './components/ClusterSandbox';
import SimulationStudio from './components/SimulationStudio';
import Leaderboard from './components/Leaderboard';
import FeedbackModal from './components/FeedbackModal';

function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('wqi');
  const [isFeedbackOpen, setIsFeedbackOpen] = useState<boolean>(false);

  return (
    <div className="min-h-screen w-screen bg-[#030712] text-slate-100 flex flex-col antialiased selection:bg-cyan-500 selection:text-white relative overflow-x-hidden">
      
      {/* Ambient Lighting Background Accents */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl" />
        <div className="absolute top-1/3 right-1/4 w-96 h-96 bg-blue-600/10 rounded-full blur-3xl" />
        <div className="absolute bottom-10 left-1/3 w-80 h-80 bg-indigo-500/5 rounded-full blur-3xl" />
      </div>

      {/* Global Top Navbar */}
      <div className="relative z-20">
        <Navbar
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          onOpenFeedback={() => setIsFeedbackOpen(true)}
        />
      </div>

      {/* Main Studio Views */}
      <main className="flex-1 w-full relative z-10 py-2">
        {activeTab === 'wqi' && <WQIStudio />}
        {activeTab === 'cluster' && <ClusterSandbox />}
        {activeTab === 'simulation' && <SimulationStudio />}
        {activeTab === 'leaderboard' && <Leaderboard />}
      </main>

      {/* Global Community Feedback Modal */}
      <FeedbackModal
        isOpen={isFeedbackOpen}
        onClose={() => setIsFeedbackOpen(false)}
      />

    </div>
  );
}

export default App;
