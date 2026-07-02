import { Briefcase, Github, Globe, Info, Languages, Moon, Sun, Waves } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { NavLink, Outlet } from "react-router-dom";

import { ArchitectureModal } from "./components/ArchitectureModal";
import { EXTERNAL_LINKS } from "./lib/links";
import { APP_VERSION } from "./lib/version";
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
  const [archOpen, setArchOpen] = useState(false);

  const switchLang = (l: Lang) => {
    setLang(l);
    void i18n.changeLanguage(l);
  };

  return (
    <div className="shell">
      <header className="top">
        <NavLink to="/" className="brand" aria-label={t("brand")}>
          <Waves size={18} aria-hidden="true" className="brand-mark" />
          <span>{t("brand")}</span>
          <small>{t("tagline")}</small>
        </NavLink>
        <nav className="main">
          {NAV.map(([path, key]) => (
            <NavLink key={key} to={`/${path}`} end={path === ""} className={({ isActive }) => (isActive ? "active" : "")}>
              {t(`nav.${key}`)}
            </NavLink>
          ))}
        </nav>
        <div className="spacer" />
        <div className="header-actions">
          <a className="iconbtn" href={EXTERNAL_LINKS.github} target="_blank" rel="noreferrer noopener" title={t("header.github")} aria-label={t("header.github")}>
            <Github size={17} aria-hidden="true" />
          </a>
          <a className="iconbtn" href={EXTERNAL_LINKS.personal} target="_blank" rel="noreferrer noopener" title={t("header.personal")} aria-label={t("header.personal")}>
            <Globe size={17} aria-hidden="true" />
          </a>
          <a className="iconbtn" href={EXTERNAL_LINKS.portfolio} target="_blank" rel="noreferrer noopener" title={t("header.portfolio")} aria-label={t("header.portfolio")}>
            <Briefcase size={17} aria-hidden="true" />
          </a>
          <span className="header-sep" aria-hidden="true" />
          <button className="iconbtn" onClick={() => setArchOpen(true)} title={t("arch.open")} aria-label={t("arch.open")}>
            <Info size={17} aria-hidden="true" />
          </button>
          <button className="iconbtn lang" onClick={() => switchLang(lang === "en" ? "es" : "en")} title={t("settings.lang", { defaultValue: "Language" })} aria-label="Language">
            <Languages size={17} aria-hidden="true" />
            <span>{lang.toUpperCase()}</span>
          </button>
          <button className="iconbtn" onClick={toggleTheme} title={t("settings.theme", { defaultValue: "Theme" })} aria-label="Theme">
            {theme === "dark" ? <Sun size={17} aria-hidden="true" /> : <Moon size={17} aria-hidden="true" />}
          </button>
        </div>
      </header>

      <main className="content">
        <Outlet />
      </main>

      <footer className="site-footer">
        {/* Compact, to-the-point footer (ADR-0016 §2): brand + lab + version + author + single GitHub + license.
            The personal/portfolio links already live in the header, so they are NOT repeated here (focus error). */}
        <div className="footer-meta">
          <span className="footer-brand">{t("brand")}</span>
          <span aria-hidden="true">·</span>
          <span>{t("footer.complement")}</span>
          <span aria-hidden="true">·</span>
          <span className="faint">{t("footer.version")} v{APP_VERSION}</span>
          <span aria-hidden="true">·</span>
          <span>{t("footer.attribution")}</span>
          <span aria-hidden="true">·</span>
          <a href={EXTERNAL_LINKS.github} target="_blank" rel="noreferrer noopener">{t("header.github")}</a>
          <span aria-hidden="true">·</span>
          <span className="faint">{t("footer.license")}</span>
        </div>
      </footer>

      {archOpen && <ArchitectureModal onClose={() => setArchOpen(false)} />}
    </div>
  );
}
