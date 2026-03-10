import { useLanguage } from '@/context/LanguageContext';
import { ScrollReveal } from '@/components/ScrollReveal';
import { Card, CardContent } from '@/components/ui/card';
import { Bot, Database, Layers } from 'lucide-react';

export function ValueProposition() {
  const { t } = useLanguage();

  const values = [
    {
      icon: Bot,
      title: t('value.fullbody.title'),
      description: t('value.fullbody.desc'),
    },
    {
      icon: Database,
      title: t('value.data.title'),
      description: t('value.data.desc'),
    },
    {
      icon: Layers,
      title: t('value.cross.title'),
      description: t('value.cross.desc'),
    },
  ];

  return (
    <section className="py-24 bg-black">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <ScrollReveal>
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold text-white text-center mb-16">
            {t('value.title')}
          </h2>
        </ScrollReveal>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {values.map((value, index) => (
            <ScrollReveal key={value.title} delay={index * 0.1}>
              <Card className="bg-white/5 border-white/10 hover:border-white/20 hover:bg-white/[0.07] transition-all duration-300 h-full group">
                <CardContent className="p-8">
                  <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-blue-500/20 to-purple-500/20 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                    <value.icon className="h-7 w-7 text-blue-400" />
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-4">
                    {value.title}
                  </h3>
                  <p className="text-white/60 leading-relaxed">
                    {value.description}
                  </p>
                </CardContent>
              </Card>
            </ScrollReveal>
          ))}
        </div>
      </div>
    </section>
  );
}
