import { useLanguage } from '@/context/LanguageContext';
import { ScrollReveal } from '@/components/ScrollReveal';
import { X, Check, Users, Bot } from 'lucide-react';

export function ScaleAIModel() {
  const { t } = useLanguage();

  return (
    <section className="py-24 bg-black relative overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-blue-900/5 to-transparent" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <ScrollReveal>
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold text-white mb-6">
              {t('scale.title')}
            </h2>
          </div>
        </ScrollReveal>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-4xl mx-auto">
          {/* Traditional Model */}
          <ScrollReveal delay={0.1}>
            <div className="bg-white/5 border border-white/10 rounded-2xl p-8 opacity-60">
              <div className="flex items-center space-x-3 mb-6">
                <div className="w-12 h-12 rounded-xl bg-red-500/20 flex items-center justify-center">
                  <Users className="h-6 w-6 text-red-400" />
                </div>
                <span className="text-red-400 font-medium">Traditional</span>
              </div>
              
              <ul className="space-y-4">
                <li className="flex items-start space-x-3">
                  <X className="h-5 w-5 text-red-400 mt-0.5 flex-shrink-0" />
                  <span className="text-white/60">{t('scale.traditional')}</span>
                </li>
              </ul>
            </div>
          </ScrollReveal>

          {/* New Model */}
          <ScrollReveal delay={0.2}>
            <div className="bg-gradient-to-br from-blue-500/10 to-purple-500/10 border border-blue-500/30 rounded-2xl p-8">
              <div className="flex items-center space-x-3 mb-6">
                <div className="w-12 h-12 rounded-xl bg-blue-500/20 flex items-center justify-center">
                  <Bot className="h-6 w-6 text-blue-400" />
                </div>
                <span className="text-blue-400 font-medium">AI-Powered</span>
              </div>
              
              <ul className="space-y-4">
                <li className="flex items-start space-x-3">
                  <Check className="h-5 w-5 text-green-400 mt-0.5 flex-shrink-0" />
                  <span className="text-white/80">{t('scale.new')}</span>
                </li>
              </ul>
            </div>
          </ScrollReveal>
        </div>
      </div>
    </section>
  );
}
