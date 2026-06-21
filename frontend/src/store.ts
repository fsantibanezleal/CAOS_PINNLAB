// Minimal global UI state (theme + language), persisted to localStorage. Case selection is route/local state.
import { create } from "zustand";

export type Theme = "dark" | "light";
export type Lang = "en" | "es";

interface UIState {
  theme: Theme;
  lang: Lang;
  setTheme: (t: Theme) => void;
  toggleTheme: () => void;
  setLang: (l: Lang) => void;
}

function initialTheme(): Theme {
  const saved = localStorage.getItem("pinnlab.theme");
  return saved === "light" || saved === "dark" ? saved : "dark";
}
function initialLang(): Lang {
  const saved = localStorage.getItem("pinnlab.lang");
  if (saved === "en" || saved === "es") return saved;
  return navigator.language?.startsWith("es") ? "es" : "en";
}

export const useUI = create<UIState>((set, get) => ({
  theme: initialTheme(),
  lang: initialLang(),
  setTheme: (theme) => {
    localStorage.setItem("pinnlab.theme", theme);
    document.documentElement.setAttribute("data-theme", theme);
    set({ theme });
  },
  toggleTheme: () => get().setTheme(get().theme === "dark" ? "light" : "dark"),
  setLang: (lang) => {
    localStorage.setItem("pinnlab.lang", lang);
    set({ lang });
  },
}));
