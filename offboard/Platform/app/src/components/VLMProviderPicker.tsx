import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

const VLM_PROVIDERS = [
  { id: 'gemini', name: 'Google Gemini', defaultModel: 'gemini-1.5-pro' },
  { id: 'gpt4v', name: 'OpenAI GPT-4V', defaultModel: 'gpt-4o' },
  { id: 'claude', name: 'Anthropic Claude', defaultModel: 'claude-opus-4-6' },
  { id: 'qwen2vl', name: 'Qwen2-VL', defaultModel: 'qwen2-vl-72b' },
  { id: 'internvl', name: 'InternVL', defaultModel: 'internvl2-40b' },
  { id: 'local-smolvlm', name: 'Local SmolVLM', defaultModel: 'smolvlm' },
];

interface VLMProviderPickerProps {
  provider: string;
  apiKey: string;
  model?: string;
  onChange: (provider: string, apiKey: string, model?: string) => void;
}

export default function VLMProviderPicker({ provider, apiKey, model, onChange }: VLMProviderPickerProps) {
  const selectedProvider = VLM_PROVIDERS.find(p => p.id === provider);

  const handleProviderChange = (id: string) => {
    const p = VLM_PROVIDERS.find(x => x.id === id);
    onChange(id, apiKey, p?.defaultModel);
  };

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label>VLM 提供商</Label>
        <Select value={provider} onValueChange={handleProviderChange}>
          <SelectTrigger>
            <SelectValue placeholder="选择 VLM 提供商..." />
          </SelectTrigger>
          <SelectContent>
            {VLM_PROVIDERS.map(p => (
              <SelectItem key={p.id} value={p.id}>{p.name}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      {provider && provider !== 'local-smolvlm' && (
        <div className="space-y-2">
          <Label>API Key</Label>
          <Input
            type="password"
            placeholder="sk-..."
            value={apiKey}
            onChange={e => onChange(provider, e.target.value, model)}
          />
        </div>
      )}
      {selectedProvider && (
        <div className="space-y-2">
          <Label>模型</Label>
          <Input
            placeholder={selectedProvider.defaultModel}
            value={model ?? selectedProvider.defaultModel}
            onChange={e => onChange(provider, apiKey, e.target.value)}
          />
        </div>
      )}
    </div>
  );
}
