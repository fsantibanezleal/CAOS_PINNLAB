import { useTranslation } from "react-i18next";
import { NavLink, Outlet } from "react-router-dom";

import i18n from "./i18n";
import { useUI, type Lang } from "./store";

const NAV: Array<[string, string]> = [
  ["", "app"],
  ["introduction", "introduction"],
  ["methodology", "methodology"],
  ["implementation", "implementation"],
  ["experiments", "experiments"],
  ["benchmark", "benchmark"],
];

export function App() {
  const { t } = useTranslation();
  const { theme, toggleTheme, lang, setLang } = useUI();

  const switchLang = (l: Lang) => {
    setLang(l);
    void i18n.changeLanguage(l);
  };

  return (
    <div className="shell">
      <header className="top">
        <div className="brand">
          {t("brand")}
          <span className="dot">●</span>
          <small>{t("tagline")}</small>
        </div>
        <nav className="main">
          {NAV.map(([path, key]) => (
            <NavLink key={key} to={`/${path}`} end={path === ""} className={({ isActive }) => (isActive ? "active" : "")}>
              {t(`nav.${key}`)}
            </NavLink>
          ))}
        </nav>
        <div className="spacer" />
        <button className="iconbtn" onClick={() => switchLang(lang === "en" ? "es" : "en")} title="Language">
          {lang.toUpperCase()}
        </button>
        <button className="iconbtn" onClick={toggleTheme} title="Theme">
          {theme === "dark" ? "☀" : "☾"}
        </button>
        <a className="iconbtn" href="https://github.com/fsantibanezleal/CAOS_PINNLAB" target="_blank" rel="noreferrer">
          GitHub
        </a>
      </header>
      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
