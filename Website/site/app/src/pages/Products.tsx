import { ProductsHero } from '@/sections/products/ProductsHero';
import { DaasSection } from '@/sections/products/DaasSection';
import { SaasSection } from '@/sections/products/SaasSection';
import { RobotSchoolSection } from '@/sections/products/RobotSchoolSection';
import { EndEffectorSection } from '@/sections/products/EndEffectorSection';

export function Products() {
  return (
    <>
      <div className="pt-16">
        <ProductsHero />
        <DaasSection />
        <SaasSection />
        <RobotSchoolSection />
        <EndEffectorSection />
      </div>
    </>
  );
}
