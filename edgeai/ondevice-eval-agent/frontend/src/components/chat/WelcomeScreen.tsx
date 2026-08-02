import { Boxes, Compass, Hammer, Workflow } from 'lucide-react';

const SUGGESTIONS: Array<{
  text: string;
  icon: typeof Boxes;
  tone: 'green' | 'blue' | 'purple' | 'red';
}> = [
  { text: 'List the available models', icon: Boxes, tone: 'blue' },
  { text: 'What inputs does the first model need?', icon: Workflow, tone: 'green' },
  { text: 'How do I read its output?', icon: Compass, tone: 'purple' },
  { text: 'Give me a frontend integration snippet', icon: Hammer, tone: 'red' },
];

const TONE_BORDER: Record<string, string> = {
  green: 'rgba(16, 185, 129, 0.3)',
  blue: 'rgba(59, 130, 246, 0.3)',
  purple: 'rgba(168, 85, 247, 0.3)',
  red: 'rgba(239, 68, 68, 0.3)',
};
const TONE_COLOR: Record<string, string> = {
  green: '#10B981',
  blue: '#3B82F6',
  purple: '#A855F7',
  red: '#EF4444',
};

export function WelcomeScreen({ onPick }: { onPick: (text: string) => void }) {
  return (
    <div className="mx-auto flex max-w-xl flex-col items-center justify-center px-6 py-12 text-center">
      <div className="mb-6 flex items-center gap-3">
        <div
          className="flex h-11 w-11 items-center justify-center rounded-lg border"
          style={{
            background: 'var(--gray-100)',
            borderColor: 'var(--primary-20)',
            color: 'var(--zededa-cyan)',
          }}
        >
          <Compass className="h-5 w-5" />
        </div>
        <h2
          className="text-[1.35rem] font-semibold whitespace-nowrap"
          style={{ color: 'var(--gray-900)' }}
        >
          Explore on-device models
        </h2>
      </div>
      <p
        className="mb-6 max-w-md text-[15px] leading-6"
        style={{ color: 'var(--gray-500)' }}
      >
        Ask about available models, inputs and outputs, or how to wire them into
        your app. The agent will call tools and stream its answer back.
      </p>

      <div className="flex flex-wrap justify-center gap-2">
        {SUGGESTIONS.map((s) => {
          const Icon = s.icon;
          return (
            <button
              key={s.text}
              type="button"
              onClick={() => onPick(s.text)}
              className="inline-flex items-center gap-2 rounded-full border bg-transparent px-4 py-2 text-sm font-medium transition-all hover:-translate-y-px hover:shadow-sm active:scale-[0.97]"
              style={{
                borderColor: TONE_BORDER[s.tone],
                color: 'var(--gray-700)',
              }}
            >
              <Icon className="h-3.5 w-3.5" style={{ color: TONE_COLOR[s.tone] }} />
              <span>{s.text}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
