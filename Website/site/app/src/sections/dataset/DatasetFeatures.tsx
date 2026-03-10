import { useLanguage } from '@/context/LanguageContext';
import { ScrollReveal } from '@/components/ScrollReveal';
import { Globe, ScanEye, Shield, Camera, Cpu } from 'lucide-react';

export function DatasetFeatures() {
  const { t } = useLanguage();

  const features = [
    { icon: Globe, label: t('dataset.feature1') },
    { icon: Cpu, label: t('dataset.feature2') },
    { icon: ScanEye, label: t('dataset.feature3') },
    { icon: Shield, label: t('dataset.feature4') },
    { icon: Camera, label: t('dataset.feature5') },
  ];

  return (
    <section className="py-24 bg-black">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <ScrollReveal>
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold text-white mb-6">
              {t('dataset.features.title')}
            </h2>
          </div>
        </ScrollReveal>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <ScrollReveal key={feature.label} delay={index * 0.1}>
              <div className="bg-white/5 border border-white/10 rounded-xl p-6 flex items-center space-x-4 hover:border-white/20 hover:bg-white/[0.07] transition-all duration-300">
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-blue-500/20 to-purple-500/20 flex items-center justify-center flex-shrink-0">
                  <feature.icon className="h-6 w-6 text-blue-400" />
                </div>
                <span className="text-white font-medium">{feature.label}</span>
              </div>
            </ScrollReveal>
          ))}
        </div>

        {/* Dataset Reference */}
        <ScrollReveal delay={0.3}>
          <div className="mt-16 text-center">
            <p className="text-white/50 text-sm mb-4">
              Based on
            </p>
            <a
              href="https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center space-x-2 text-blue-400 hover:text-blue-300 transition-colors"
            >
              <span className="text-lg font-medium">agibot-world/AgiBotWorld-Alpha</span>
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
            </a>
          </div>
        </ScrollReveal>
      </div>
    </section>
  );
}
