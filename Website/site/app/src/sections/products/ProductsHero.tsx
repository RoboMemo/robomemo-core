import { useLanguage } from '@/context/LanguageContext';
import { ScrollReveal } from '@/components/ScrollReveal';

export function ProductsHero() {
  const { t } = useLanguage();

  return (
    <section id="products" className="py-24 bg-black relative overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-gradient-to-b from-blue-900/10 via-transparent to-purple-900/10" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-gradient-to-r from-blue-500/5 to-purple-500/5 rounded-full blur-3xl" />
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <ScrollReveal>
          <div className="text-center">
            <h2 className="text-4xl sm:text-5xl md:text-6xl font-bold text-white mb-6">
              {t('products.hero.title')}
            </h2>
            <p className="text-lg text-white/70 max-w-2xl mx-auto">
              Comprehensive embodied AI solutions from data to deployment
            </p>
          </div>
        </ScrollReveal>
      </div>
    </section>
  );
}
