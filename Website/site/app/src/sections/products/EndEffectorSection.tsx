import { useLanguage } from '@/context/LanguageContext';
import { ScrollReveal } from '@/components/ScrollReveal';
import { Wrench, Zap, Check, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';

export function EndEffectorSection() {
  const { t } = useLanguage();

  const features = [
    'Integrated torque sensor',
    'Precision control',
    'Plug-and-play installation',
    'Instant skill acquisition',
  ];

  return (
    <section className="py-24 bg-black relative overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-to-b from-orange-900/5 via-transparent to-transparent" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Product Image */}
          <ScrollReveal direction="left">
            <div className="relative">
              <div className="aspect-square rounded-2xl overflow-hidden border border-white/10 bg-white/5">
                <img
                  src="/images/end-effector.jpg"
                  alt="End Effector"
                  className="w-full h-full object-cover hover:scale-105 transition-transform duration-500"
                />
              </div>
              {/* Decorative */}
              <div className="absolute -top-4 -right-4 w-24 h-24 bg-orange-500/20 rounded-full blur-2xl" />
              <div className="absolute -bottom-4 -left-4 w-32 h-32 bg-blue-500/20 rounded-full blur-2xl" />
            </div>
          </ScrollReveal>

          {/* Content */}
          <div>
            <ScrollReveal>
              <div className="flex items-center space-x-3 mb-6">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-orange-500/20 to-yellow-500/20 flex items-center justify-center">
                  <Wrench className="h-6 w-6 text-orange-400" />
                </div>
                <span className="text-orange-400 text-sm font-medium uppercase tracking-wider">
                  Hardware
                </span>
              </div>
            </ScrollReveal>

            <ScrollReveal delay={0.1}>
              <h2 className="text-3xl sm:text-4xl font-bold text-white mb-6">
                {t('ee.title')}
              </h2>
            </ScrollReveal>

            <ScrollReveal delay={0.2}>
              <p className="text-lg text-white/70 mb-8">
                {t('ee.desc')}
              </p>
            </ScrollReveal>

            <ScrollReveal delay={0.3}>
              <div className="flex items-center space-x-2 mb-6">
                <Zap className="h-5 w-5 text-yellow-400" />
                <span className="text-white font-medium">{t('ee.skill')}</span>
              </div>
            </ScrollReveal>

            <ScrollReveal delay={0.4}>
              <ul className="space-y-3 mb-8">
                {features.map((feature, idx) => (
                  <li key={idx} className="flex items-center space-x-3">
                    <div className="w-5 h-5 rounded-full bg-green-500/20 flex items-center justify-center">
                      <Check className="h-3 w-3 text-green-400" />
                    </div>
                    <span className="text-white/70">{feature}</span>
                  </li>
                ))}
              </ul>
            </ScrollReveal>

            <ScrollReveal delay={0.5}>
              <Button variant="outline" className="border-orange-500/30 text-orange-400 hover:bg-orange-500/10">
                Learn More
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            </ScrollReveal>
          </div>
        </div>
      </div>
    </section>
  );
}
