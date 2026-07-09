// Canonical external links (ADR-0016). Only public github.com / public-site URLs: never local machine paths,
// never private vault references. Mirrors the CAOS_SEISMIC reference shell.

export const EXTERNAL_LINKS = {
  /** The project's public source repository. */
  github: "https://github.com/fsantibanezleal/CAOS_PINNLAB",
  /** Felipe's personal site. */
  personal: "https://fsantibanezleal.github.io",
  /** Felipe's portfolio. */
  portfolio: "https://fasl-work.com",
} as const;
