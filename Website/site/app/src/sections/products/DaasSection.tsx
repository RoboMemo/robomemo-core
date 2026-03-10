import { useLanguage } from '@/context/LanguageContext';
import { ScrollReveal } from '@/components/ScrollReveal';
import { Database, Package, Check, Download } from 'lucide-react';
import { Button } from '@/components/ui/button';

export function DaasSection() {
  const { t } = useLanguage();

  const features = [
    t('daas.feature1'),
    t('daas.feature2'),
    t('daas.feature3'),
  ];

  return (
    <section className="py-24 bg-black">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Content */}
          <div>
            <ScrollReveal>
              <div className="flex items-center space-x-3 mb-6">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500/20 to-purple-500/20 flex items-center justify-center">
                  <Database className="h-6 w-6 text-blue-400" />
                </div>
                <span className="text-blue-400 text-sm font-medium uppercase tracking-wider">
                  DaaS
                </span>
              </div>
            </ScrollReveal>

            <ScrollReveal delay={0.1}>
              <h2 className="text-3xl sm:text-4xl font-bold text-white mb-6">
                {t('daas.title')}
              </h2>
            </ScrollReveal>

            <ScrollReveal delay={0.2}>
              <p className="text-lg text-white/70 mb-8">
                {t('daas.desc')}
              </p>
            </ScrollReveal>

            <ScrollReveal delay={0.3}>
              <ul className="space-y-4 mb-8">
                {features.map((feature, idx) => (
                  <li key={idx} className="flex items-center space-x-3">
                    <div className="w-6 h-6 rounded-full bg-green-500/20 flex items-center justify-center">
                      <Check className="h-4 w-4 text-green-400" />
                    </div>
                    <span className="text-white/80">{feature}</span>
                  </li>
                ))}
              </ul>
            </ScrollReveal>

            <ScrollReveal delay={0.4}>
              <Button className="bg-white text-black hover:bg-white/90">
                <Package className="mr-2 h-5 w-5" />
                Browse Datasets
              </Button>
            </ScrollReveal>
          </div>

          {/* Visual */}
          <ScrollReveal delay={0.2} direction="right">
            <div className="relative">
              <div className="bg-white/5 border border-white/10 rounded-2xl p-8">
                <div className="grid grid-cols-2 gap-4">
                  {[
                    { name: 'Grasping', size: '2.3 GB' },
                    { name: 'Placement', size: '1.8 GB' },
                    { name: 'Tool Use', size: '3.1 GB' },
                    { name: 'Assembly', size: '4.2 GB' },
                  ].map((item) => (
                    <div
                      key={item.name}
                      className="bg-white/5 border border-white/10 rounded-xl p-4 hover:border-white/20 transition-all duration-300"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <Package className="h-5 w-5 text-blue-400" />
                        <Download className="h-4 w-4 text-white/40" />
                      </div>
                      <div className="text-white font-medium text-sm">{item.name}</div>
                      <div className="text-white/40 text-xs">{item.size}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </ScrollReveal>
        </div>
      </div>
    </section>
  );
}
