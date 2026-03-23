import { useState, useEffect } from 'react';
import { KeyRound, ArrowRight, Database } from 'lucide-react';

const ACCESS_CODE = '60602656';
const STORAGE_KEY = 'robomemo_access';

export function useAccessGate() {
  const [authorized, setAuthorized] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === ACCESS_CODE) {
      setAuthorized(true);
    }
  }, []);

  const authorize = (code: string): boolean => {
    if (code === ACCESS_CODE) {
      localStorage.setItem(STORAGE_KEY, code);
      setAuthorized(true);
      return true;
    }
    return false;
  };

  const logout = () => {
    localStorage.removeItem(STORAGE_KEY);
    setAuthorized(false);
  };

  return { authorized, authorize, logout };
}

export default function AccessGate({ onAuthorized }: { onAuthorized: () => void }) {
  const [code, setCode] = useState('');
  const [error, setError] = useState(false);
  const [shaking, setShaking] = useState(false);
  const { authorize } = useAccessGate();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (authorize(code.trim())) {
      onAuthorized();
    } else {
      setError(true);
      setShaking(true);
      setTimeout(() => setShaking(false), 500);
      setTimeout(() => setError(false), 3000);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-primary/10 rounded-2xl mb-4">
            <Database className="w-8 h-8 text-primary" />
          </div>
          <h1 className="text-2xl font-bold text-slate-900">RoboMemo</h1>
          <p className="text-sm text-slate-500 mt-1">Embodied Intelligence Data Platform</p>
        </div>

        {/* Card */}
        <div className={`bg-white rounded-2xl shadow-lg border border-slate-200 p-8 transition-transform ${shaking ? 'animate-shake' : ''}`}>
          <div className="flex items-center gap-3 mb-6">
            <div className="flex items-center justify-center w-10 h-10 bg-amber-50 rounded-xl">
              <KeyRound className="w-5 h-5 text-amber-600" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-slate-900">Access Required</h2>
              <p className="text-sm text-slate-500">Enter your trial access code</p>
            </div>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="space-y-4">
              <div>
                <input
                  type="text"
                  inputMode="numeric"
                  pattern="[0-9]*"
                  maxLength={8}
                  value={code}
                  onChange={(e) => {
                    setCode(e.target.value.replace(/\D/g, ''));
                    setError(false);
                  }}
                  placeholder="Enter 8-digit code"
                  className={`w-full px-4 py-3 text-center text-2xl font-mono tracking-[0.3em] border rounded-xl outline-none transition-colors ${
                    error 
                      ? 'border-red-300 bg-red-50 text-red-600' 
                      : 'border-slate-200 bg-slate-50 text-slate-900 focus:border-primary focus:bg-white'
                  }`}
                  autoFocus
                />
                {error && (
                  <p className="text-sm text-red-500 mt-2 text-center">Invalid access code</p>
                )}
              </div>

              <button
                type="submit"
                disabled={code.length < 8}
                className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-primary text-primary-foreground rounded-xl font-medium text-sm transition-all hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Enter Platform
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </form>
        </div>

        <p className="text-center text-xs text-slate-400 mt-6">
          Contact your administrator for access credentials
        </p>
      </div>

      <style>{`
        @keyframes shake {
          0%, 100% { transform: translateX(0); }
          10%, 30%, 50%, 70%, 90% { transform: translateX(-4px); }
          20%, 40%, 60%, 80% { transform: translateX(4px); }
        }
        .animate-shake { animation: shake 0.5s ease-in-out; }
      `}</style>
    </div>
  );
}
