import { Hero } from '@/sections/home/Hero';
import { ValueProposition } from '@/sections/home/ValueProposition';
import { TaskFocus } from '@/sections/home/TaskFocus';
import { GrowthCurve } from '@/sections/home/GrowthCurve';
import { MultiFormSupport } from '@/sections/home/MultiFormSupport';
import { ScaleAIModel } from '@/sections/home/ScaleAIModel';
import { LanxiangAnalogy } from '@/sections/home/LanxiangAnalogy';

export function Home() {
  return (
    <>
      <Hero />
      <ValueProposition />
      <TaskFocus />
      <GrowthCurve />
      <MultiFormSupport />
      <ScaleAIModel />
      <LanxiangAnalogy />
    </>
  );
}
