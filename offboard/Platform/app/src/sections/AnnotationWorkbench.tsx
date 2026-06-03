import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import DataAnnotation from '@/sections/DataAnnotation';
import AutoAnnotation from '@/sections/AutoAnnotation';
import StructuredVQA from '@/sections/StructuredVQA';

export default function AnnotationWorkbench() {
  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-2xl font-bold">标注工作台</h2>
        <p className="text-muted-foreground">手工标注、VLM 自动标注与结构化 VQA</p>
      </div>

      <Tabs defaultValue="manual">
        <TabsList className="grid w-full max-w-md grid-cols-3">
          <TabsTrigger value="manual">手工标注</TabsTrigger>
          <TabsTrigger value="vlm">VLM 自动标注</TabsTrigger>
          <TabsTrigger value="vqa">结构化 VQA</TabsTrigger>
        </TabsList>

        <TabsContent value="manual">
          <DataAnnotation />
        </TabsContent>

        <TabsContent value="vlm">
          <AutoAnnotation />
        </TabsContent>

        <TabsContent value="vqa">
          <StructuredVQA />
        </TabsContent>
      </Tabs>
    </div>
  );
}
