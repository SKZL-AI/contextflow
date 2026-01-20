/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        slate: {
          850: '#1a1f2e',
          950: '#0a0e17',
        },
        violet: {
          450: '#9061f9',
          550: '#7c3aed',
        },
      },
      backgroundColor: {
        'dark-primary': '#0a0e17',
        'dark-secondary': '#1a1f2e',
        'dark-tertiary': '#242938',
      },
      borderColor: {
        'dark-border': '#2d3548',
      },
      boxShadow: {
        'glow-violet': '0 0 20px rgba(139, 92, 246, 0.3)',
        'glow-slate': '0 0 20px rgba(100, 116, 139, 0.2)',
      },
    },
  },
  plugins: [],
}
