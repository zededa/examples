/**
 * Tailwind config — mirrors the ZEDEDA EPI design tokens from
 * webapp/static/css/variables.css so the React frontend uses
 * the same palette, radii, shadows, and typography as the legacy UI.
 */
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: ['class', '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        cyan: {
          DEFAULT: '#00B8C3',
          hover: '#00A5AF',
        },
        tag: {
          blue: '#5B8DEF',
          green: '#10B981',
          teal: '#14B8A6',
          amber: '#F59E0B',
          pink: '#EC4899',
          red: '#EF4444',
          cyan: '#06B6D4',
          purple: '#A855F7',
          indigo: '#6366F1',
        },
        success: '#10B981',
        warning: '#F59E0B',
        error: '#EF4444',
        info: '#3B82F6',
      },
      fontFamily: {
        sans: [
          'Inter',
          'ui-sans-serif',
          'system-ui',
          '-apple-system',
          'sans-serif',
        ],
        mono: [
          'Fira Code',
          'ui-monospace',
          'SFMono-Regular',
          'Menlo',
          'Monaco',
          'monospace',
        ],
      },
      borderRadius: {
        btn: '20px',
        card: '12px',
        bubble: '16px',
      },
      boxShadow: {
        floating:
          '0 25px 50px -12px rgba(0, 0, 0, 0.15), 0 0 0 1px rgba(0, 0, 0, 0.05)',
        'floating-focus':
          '0 25px 50px -12px rgba(0, 0, 0, 0.15), 0 0 30px -5px rgba(0, 184, 195, 0.15)',
        hover: '0 4px 12px rgba(0, 0, 0, 0.08)',
        'glow-cyan': '0 0 20px rgba(0, 184, 195, 0.3)',
      },
      keyframes: {
        messageSlideIn: {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        typingBounce: {
          '0%,60%,100%': { transform: 'translateY(0)', opacity: '0.4' },
          '30%': { transform: 'translateY(-6px)', opacity: '1' },
        },
        cursorBlink: {
          '0%,50%': { opacity: '1' },
          '50.01%,100%': { opacity: '0' },
        },
      },
      animation: {
        'message-in': 'messageSlideIn 0.25s ease-out',
        'typing-bounce': 'typingBounce 1.4s infinite',
        'cursor-blink': 'cursorBlink 1s step-end infinite',
      },
    },
  },
  plugins: [],
};
