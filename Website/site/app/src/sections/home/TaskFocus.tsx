import { useLanguage } from '@/context/LanguageContext';
import { ScrollReveal } from '@/components/ScrollReveal';
import { Wrench } from 'lucide-react';

export function TaskFocus() {
  const { t } = useLanguage();

  return (
    <section className="py-24 bg-black relative overflow-hidden">
      {/* Background Gradient */}
      <div className="absolute inset-0 bg-gradient-to-r from-blue-900/10 via-transparent to-purple-900/10" />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Text Content */}
          <div>
            <ScrollReveal>
              <div className="flex items-center space-x-3 mb-6">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500/20 to-purple-500/20 flex items-center justify-center">
                  <Wrench className="h-6 w-6 text-blue-400" />
                </div>
                <span className="text-blue-400 text-sm font-medium uppercase tracking-wider">
                  Action Level
                </span>
              </div>
            </ScrollReveal>

            <ScrollReveal delay={0.1}>
              <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold text-white mb-6">
                {t('task.title')}
              </h2>
            </ScrollReveal>

            <ScrollReveal delay={0.2}>
              <p className="text-lg text-white/70 leading-relaxed">
                {t('task.desc')}
              </p>
            </ScrollReveal>

            <ScrollReveal delay={0.3}>
              <div className="mt-8 flex flex-wrap gap-3">
                <span className="px-4 py-2 bg-white/5 border border-white/10 rounded-full text-white/70 text-sm">
                  Screws Securing
                </span>
                <span className="px-4 py-2 bg-white/5 border border-white/10 rounded-full text-white/70 text-sm">
                  Nut Fastening
                </span>
                <span className="px-4 py-2 bg-white/5 border border-white/10 rounded-full text-white/70 text-sm">
                  Assembly
                </span>
              </div>
            </ScrollReveal>
          </div>

          {/* Image */}
          <ScrollReveal delay={0.2} direction="right">
            <div className="relative">
              <div className="aspect-video rounded-2xl overflow-hidden border border-white/10">
                <img
                  src="/images/task-screws.jpg"
                  alt="Robot Screwing Task"
                  className="w-full h-full object-cover hover:scale-105 transition-transform duration-500"
                />
              </div>
              {/* Decorative Elements */}
              <div className="absolute -top-4 -right-4 w-24 h-24 bg-blue-500/20 rounded-full blur-2xl" />
              <div className="absolute -bottom-4 -left-4 w-32 h-32 bg-purple-500/20 rounded-full blur-2xl" />
            </div>
          </ScrollReveal>
        </div>
      </div>
    </section>
  );
}
