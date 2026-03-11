import { useLanguage } from '@/context/LanguageContext';
import { ScrollReveal } from '@/components/ScrollReveal';
import { Cloud, Wand2, Sparkles, Layers, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';

export function SaasSection() {
  const { t } = useLanguage();

  const features = [
    { icon: Wand2, label: t('saas.feature1') },
    { icon: Sparkles, label: t('saas.feature2') },
    { icon: Layers, label: t('saas.feature3') },
  ];

  return (
    <section className="py-24 bg-black relative overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-to-r from-purple-900/5 via-transparent to-blue-900/5" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Visual */}
          <ScrollReveal direction="left">
            <div className="relative order-2 lg:order-1">
              <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
                {/* Mock UI */}
                <div className="space-y-4">
                  <div className="flex items-center space-x-2 pb-4 border-b border-white/10">
                    <div className="w-3 h-3 rounded-full bg-red-500" />
                    <div className="w-3 h-3 rounded-full bg-yellow-500" />
                    <div className="w-3 h-3 rounded-full bg-green-500" />
                  </div>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="col-span-2 space-y-3">
                      <div className="h-24 bg-white/5 rounded-lg flex items-center justify-center">
                        <span className="text-white/40 text-sm">Video Preview</span>
                      </div>
                      <div className="flex space-x-2">
                        <div className="flex-1 h-8 bg-blue-500/20 rounded flex items-center justify-center">
                          <span className="text-blue-400 text-xs">Auto Label</span>
                        </div>
                        <div className="flex-1 h-8 bg-white/5 rounded flex items-center justify-center">
                          <span className="text-white/40 text-xs">Clean</span>
                        </div>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="h-16 bg-white/5 rounded" />
                      <div className="h-16 bg-white/5 rounded" />
                      <div className="h-16 bg-white/5 rounded" />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </ScrollReveal>

          {/* Content */}
          <div className="order-1 lg:order-2">
            <ScrollReveal>
              <div className="flex items-center space-x-3 mb-6">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-500/20 to-blue-500/20 flex items-center justify-center">
                  <Cloud className="h-6 w-6 text-purple-400" />
                </div>
                <span className="text-purple-400 text-sm font-medium uppercase tracking-wider">
                  SaaS
                </span>
              </div>
            </ScrollReveal>

            <ScrollReveal delay={0.1}>
              <h2 className="text-3xl sm:text-4xl font-bold text-white mb-6">
                {t('saas.title')}
              </h2>
            </ScrollReveal>

            <ScrollReveal delay={0.2}>
              <p className="text-lg text-white/70 mb-8">
                {t('saas.desc')}
              </p>
            </ScrollReveal>

            <ScrollReveal delay={0.3}>
              <div className="flex flex-wrap gap-3 mb-8">
                {features.map((feature) => (
                  <div
                    key={feature.label}
                    className="flex items-center space-x-2 px-4 py-2 bg-white/5 border border-white/10 rounded-full"
                  >
                    <feature.icon className="h-4 w-4 text-purple-400" />
                    <span className="text-white/80 text-sm">{feature.label}</span>
                  </div>
                ))}
              </div>
            </ScrollReveal>

            <ScrollReveal delay={0.4}>
              <Button variant="outline" className="border-white/20 text-white hover:bg-white/10">
                Try Platform
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            </ScrollReveal>
          </div>
        </div>
      </div>
    </section>
  );
}
