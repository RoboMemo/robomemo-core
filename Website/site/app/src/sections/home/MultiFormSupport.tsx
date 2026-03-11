import { useLanguage } from '@/context/LanguageContext';
import { ScrollReveal } from '@/components/ScrollReveal';

export function MultiFormSupport() {
  const { t } = useLanguage();

  const forms = [
    {
      name: t('form.single'),
      image: '/images/robot-single-arm.png',
      description: 'Single-arm manipulation tasks',
    },
    {
      name: t('form.dual'),
      image: '/images/robot-dual-arm.png',
      description: 'Dual-arm coordination tasks',
    },
    {
      name: t('form.wheeled'),
      image: '/images/robot-wheeled.png',
      description: 'Mobile manipulation platforms',
    },
    {
      name: t('form.quadruped'),
      image: '/images/robot-quadruped.png',
      description: 'Legged locomotion systems',
    },
  ];

  return (
    <section className="py-24 bg-black">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <ScrollReveal>
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold text-white mb-6">
              {t('form.title')}
            </h2>
            <p className="text-lg text-white/70 max-w-2xl mx-auto">
              {t('form.desc')}
            </p>
          </div>
        </ScrollReveal>

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
          {forms.map((form, index) => (
            <ScrollReveal key={form.name} delay={index * 0.1}>
              <div className="bg-white/5 border border-white/10 rounded-2xl p-6 hover:border-white/20 hover:bg-white/[0.07] transition-all duration-300 text-center group cursor-pointer">
                {/* Icon */}
                <div className="w-24 h-24 mx-auto mb-4 flex items-center justify-center">
                  <img
                    src={form.image}
                    alt={form.name}
                    className="w-full h-full object-contain filter brightness-0 invert opacity-80 group-hover:opacity-100 transition-opacity duration-300"
                  />
                </div>
                
                <h3 className="text-lg font-semibold text-white mb-2">
                  {form.name}
                </h3>
                <p className="text-white/50 text-sm">
                  {form.description}
                </p>
              </div>
            </ScrollReveal>
          ))}
        </div>
      </div>
    </section>
  );
}
