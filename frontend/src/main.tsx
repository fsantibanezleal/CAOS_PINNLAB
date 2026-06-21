import "katex/dist/katex.min.css";
import "./styles.css";
import "./i18n";

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { RouterProvider } from "react-router-dom";

import { router } from "./router";
import { useUI } from "./store";

// apply the persisted theme before first paint
document.documentElement.setAttribute("data-theme", useUI.getState().theme);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
