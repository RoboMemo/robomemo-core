import { useLanguage } from '@/context/LanguageContext';
import { ScrollReveal } from '@/components/ScrollReveal';
import { GraduationCap, Database, Brain, Rocket, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';

export function RobotSchoolSection() {
  const { t } = useLanguage();

  const steps = [
    { icon: Database, title: t('school.step1'), description: 'Collect task-specific data' },
    { icon: Brain, title: t('school.step2'), description: 'Train and fine-tune models' },
    { icon: Rocket, title: t('school.step3'), description: 'Deploy and validate' },
  ];

  return (
    <section className="py-24 bg-black">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Content */}
          <div>
            <ScrollReveal>
              <div className="flex items-center space-x-3 mb-6">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-green-500/20 to-blue-500/20 flex items-center justify-center">
                  <GraduationCap className="h-6 w-6 text-green-400" />
                </div>
                <span className="text-green-400 text-sm font-medium uppercase tracking-wider">
                  Managed Service
                </span>
              </div>
            </ScrollReveal>

            <ScrollReveal delay={0.1}>
              <h2 className="text-3xl sm:text-4xl font-bold text-white mb-6">
                {t('school.title')}
              </h2>
            </ScrollReveal>

            <ScrollReveal delay={0.2}>
              <p className="text-lg text-white/70 mb-8">
                {t('school.desc')}
              </p>
            </ScrollReveal>

            <ScrollReveal delay={0.3}>
              <Button className="bg-white text-black hover:bg-white/90 mb-8">
                Get Started
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            </ScrollReveal>
          </div>

          {/* Steps */}
          <div className="space-y-4">
            {steps.map((step, index) => (
              <ScrollReveal key={step.title} delay={index * 0.15} direction="right">
                <div className="bg-white/5 border border-white/10 rounded-xl p-6 flex items-center space-x-4 hover:border-white/20 hover:bg-white/[0.07] transition-all duration-300">
                  <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-green-500/20 to-blue-500/20 flex items-center justify-center flex-shrink-0">
                    <step.icon className="h-7 w-7 text-green-400" />
                  </div>
                  <div>
                    <div className="flex items-center space-x-2 mb-1">
                      <span className="text-green-400 text-xs font-medium">Step {index + 1}</span>
                    </div>
                    <h3 className="text-lg font-semibold text-white">{step.title}</h3>
                    <p className="text-white/50 text-sm">{step.description}</p>
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
