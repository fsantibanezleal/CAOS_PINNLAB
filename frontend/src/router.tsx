import { createHashRouter } from "react-router-dom";

import { App } from "./App";
import { AppPage } from "./pages/AppPage";
import { Benchmark } from "./pages/Benchmark";
import { Experiments } from "./pages/Experiments";
import { Implementation } from "./pages/Implementation";
import { Introduction } from "./pages/Introduction";
import { Methodology } from "./pages/Methodology";

// HashRouter: works on GitHub Pages with no server rewrites (refresh-safe), incl. a custom domain.
export const router = createHashRouter([
  {
    path: "/",
    element: <App />,
    children: [
      { index: true, element: <AppPage /> },
      { path: "introduction", element: <Introduction /> },
      { path: "methodology", element: <Methodology /> },
      { path: "implementation", element: <Implementation /> },
      { path: "experiments", element: <Experiments /> },
      { path: "benchmark", element: <Benchmark /> },
    ],
  },
]);
