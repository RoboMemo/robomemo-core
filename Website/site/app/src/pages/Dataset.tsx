import { DatasetHero } from '@/sections/dataset/DatasetHero';
import { DatasetFeatures } from '@/sections/dataset/DatasetFeatures';

export function Dataset() {
  return (
    <>
      <div className="pt-16">
        <DatasetHero />
        <DatasetFeatures />
      </div>
    </>
  );
}
