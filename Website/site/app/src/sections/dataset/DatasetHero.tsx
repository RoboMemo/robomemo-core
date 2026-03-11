import { useInView } from 'framer-motion';
import { useRef, useEffect, useState } from 'react';
import { useLanguage } from '@/context/LanguageContext';
import { ScrollReveal } from '@/components/ScrollReveal';
import { Layers, Clock, Database, Cpu } from 'lucide-react';

function AnimatedCounter({ value, suffix = '' }: { value: number; suffix?: string }) {
  const [count, setCount] = useState(0);
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true });

  useEffect(() => {
    if (isInView) {
      const duration = 2000;
      const steps = 60;
      const increment = value / steps;
      let current = 0;
      
      const timer = setInterval(() => {
        current += increment;
        if (current >= value) {
          setCount(value);
          clearInterval(timer);
        } else {
          setCount(Math.floor(current));
        }
      }, duration / steps);

      return () => clearInterval(timer);
    }
  }, [isInView, value]);

  return (
    <span ref={ref}>
      {count}{suffix}
    </span>
  );
}

export function DatasetHero() {
  const { t } = useLanguage();

  const stats = [
    { icon: Layers, value: 36, suffix: '', label: t('dataset.stats.tasks') },
    { icon: Clock, value: 595, suffix: '+', label: t('dataset.stats.hours') },
    { icon: Database, value: 10, suffix: 'M+', label: t('dataset.stats.samples') },
    { icon: Cpu, value: 1, suffix: '', label: t('dataset.stats.type') },
  ];

  return (
    <section id="dataset" className="py-24 bg-black relative overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0">
        <img
          src="/images/dataset-samples.jpg"
          alt="Dataset Background"
          className="w-full h-full object-cover opacity-20"
        />
        <div className="absolute inset-0 bg-gradient-to-b from-black via-black/90 to-black" />
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <ScrollReveal>
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold text-white mb-6">
              {t('dataset.hero.title')}
            </h2>
            <p className="text-lg text-white/70 max-w-2xl mx-auto">
              {t('dataset.hero.desc')}
            </p>
          </div>
        </ScrollReveal>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
          {stats.map((stat, index) => (
            <ScrollReveal key={stat.label} delay={index * 0.1}>
              <div className="bg-white/5 border border-white/10 rounded-2xl p-6 text-center hover:border-white/20 transition-all duration-300">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500/20 to-purple-500/20 flex items-center justify-center mx-auto mb-4">
                  <stat.icon className="h-6 w-6 text-blue-400" />
                </div>
                <div className="text-3xl sm:text-4xl font-bold text-white mb-2">
                  <AnimatedCounter value={stat.value} suffix={stat.suffix} />
                </div>
                <div className="text-white/60 text-sm">
                  {stat.label}
                </div>
              </div>
            </ScrollReveal>
          ))}
        </div>
      </div>
    </section>
  );
}
