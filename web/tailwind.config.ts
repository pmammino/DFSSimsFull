import type { Config } from "tailwindcss";

// RotoWire full-dark brand tokens, ported from the Streamlit app
// (.streamlit/config.toml + app.py :root variables).
const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        rw: {
          red: "#f22e45",
          "red-400": "#f5566a",
          "red-700": "#c21e31",
          navy: "#002248",
          ink: "#000d1a",
          surface: "#002248",
          raised: "#083363",
          line: "#1c4a7a",
          mut: "#8ba0ba",
          turf: "#00e657",
          ketchup: "#ff4537",
        },
      },
      fontFamily: {
        body: ["Cosmica", "ui-sans-serif", "system-ui", "Segoe UI", "sans-serif"],
        display: ["'Integral CF'", "Impact", "system-ui", "sans-serif"],
        mono: ["'Cosmica Mono'", "ui-monospace", "Menlo", "Consolas", "monospace"],
      },
      borderRadius: {
        card: "12px",
      },
    },
  },
  plugins: [],
};

export default config;
