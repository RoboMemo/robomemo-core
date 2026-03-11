import { useLanguage } from '@/context/LanguageContext';
import { ScrollReveal } from '@/components/ScrollReveal';
import { Database, Brain, CheckCircle } from 'lucide-react';

export function GrowthCurve() {
  const { t } = useLanguage();

  const steps = [
    {
      icon: Database,
      title: t('growth.step1'),
      description: 'High-quality embodied data collection',
    },
    {
      icon: Brain,
      title: t('growth.step2'),
      description: 'AI model training with feedback loops',
    },
    {
      icon: CheckCircle,
      title: t('growth.step3'),
      description: 'Deploy and validate on real robots',
    },
  ];

  return (
    <section className="py-24 bg-black relative overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-full blur-3xl" />
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <ScrollReveal>
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold text-white mb-6">
              {t('growth.title')}
            </h2>
            <p className="text-lg text-white/70 max-w-2xl mx-auto">
              {t('growth.desc')}
            </p>
          </div>
        </ScrollReveal>

        {/* Flywheel Diagram */}
        <div className="relative">
          {/* Connection Lines (Desktop) */}
          <div className="hidden lg:block absolute top-1/2 left-0 right-0 h-0.5 bg-gradient-to-r from-blue-500/30 via-purple-500/30 to-blue-500/30 -translate-y-1/2" />
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {steps.map((step, index) => (
              <ScrollReveal key={step.title} delay={index * 0.15}>
                <div className="relative">
                  {/* Card */}
                  <div className="bg-white/5 border border-white/10 rounded-2xl p-8 hover:border-white/20 hover:bg-white/[0.07] transition-all duration-300 text-center group">
                    {/* Icon */}
                    <div className="w-20 h-20 rounded-full bg-gradient-to-br from-blue-500/20 to-purple-500/20 flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-300">
                      <step.icon className="h-10 w-10 text-blue-400" />
                    </div>
                    
                    {/* Step Number */}
                    <div className="absolute -top-3 left-1/2 -translate-x-1/2 w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white text-sm font-bold">
                      {index + 1}
                    </div>
                    
                    <h3 className="text-xl font-semibold text-white mb-3">
                      {step.title}
                    </h3>
                    <p className="text-white/60">
                      {step.description}
                    </p>
                  </div>
                </div>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
