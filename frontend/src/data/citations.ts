/** Real, link-verified references (ADR-0016 §7.5). Every entry carries a real DOI (preferred) or URL: a bare
 *  author-year with no link is a FAIL. Referenced inline via <Cite id> and per-section via <Refs ids={[...]}>;
 *  there is NO bottom-of-page bibliography dump (the banned ReferenceList pattern). */
export interface Citation {
  id: string;
  label: string; // short in-text label, e.g. "Raissi 2019"
  citation: string; // full citation string
  doi?: string;
  url?: string;
}

export const CITATIONS: Record<string, Citation> = {
  "wu-2023": { id: "wu-2023", label: "Wu 2023", citation: "Wu, Zhu, Tan, Kartha & Lu (2023). A comprehensive study of non-adaptive and residual-based adaptive sampling for PINNs. CMAME 403:115671.", doi: "10.1016/j.cma.2022.115671" },
  "wang-causality-2024": { id: "wang-causality-2024", label: "Wang 2024 (causality)", citation: "Wang, Sankaran & Perdikaris (2024). Respecting causality for training physics-informed neural networks. CMAME 421:116813.", doi: "10.1016/j.cma.2024.116813" },
  "wang-ntk-2022": { id: "wang-ntk-2022", label: "Wang 2022 (NTK)", citation: "Wang, Yu & Perdikaris (2022). When and why PINNs fail to train: a neural tangent kernel perspective. JCP 449:110768.", doi: "10.1016/j.jcp.2021.110768" },
  "tancik-2020": { id: "tancik-2020", label: "Tancik 2020", citation: "Tancik et al. (2020). Fourier features let networks learn high-frequency functions in low-dimensional domains. NeurIPS 33.", doi: "10.48550/arXiv.2006.10739" },
  "moseley-2023": { id: "moseley-2023", label: "Moseley 2023", citation: "Moseley, Markham & Nissen-Meyer (2023). Finite basis PINNs (FBPINNs): scalable domain decomposition. Adv. Comput. Math. 49:62.", doi: "10.1007/s10444-023-10065-9" },
  "kharazmi-2021": { id: "kharazmi-2021", label: "Kharazmi 2021", citation: "Kharazmi, Zhang & Karniadakis (2021). hp-VPINNs: variational physics-informed neural networks. CMAME 374:113547.", doi: "10.1016/j.cma.2020.113547" },
  "lu-deepxde-2021": { id: "lu-deepxde-2021", label: "Lu 2021 (DeepXDE)", citation: "Lu, Meng, Mao & Karniadakis (2021). DeepXDE: a deep learning library for solving differential equations. SIAM Review 63(1):208-228.", doi: "10.1137/19M1274067" },
  "li-fno-2021": { id: "li-fno-2021", label: "Li 2021 (FNO)", citation: "Li et al. (2021). Fourier Neural Operator for parametric partial differential equations. ICLR 2021.", doi: "10.48550/arXiv.2010.08895" },
  "raissi-2019": { id: "raissi-2019", label: "Raissi 2019", citation: "Raissi, Perdikaris & Karniadakis (2019). Physics-informed neural networks. JCP 378:686-707.", doi: "10.1016/j.jcp.2018.10.045" },
  "anagnostopoulos-2024": { id: "anagnostopoulos-2024", label: "Anagnostopoulos 2024", citation: "Anagnostopoulos, Toscano, Stergiopulos & Karniadakis (2024). Residual-based attention in physics-informed neural networks. CMAME 421:116805.", doi: "10.48550/arXiv.2307.00379" },
  "wang-piratenets-2024": { id: "wang-piratenets-2024", label: "Wang 2024 (PirateNets)", citation: "Wang, Li, Chen & Perdikaris (2024). PirateNets: physics-informed deep learning with residual adaptive networks. JMLR 25.", doi: "10.48550/arXiv.2402.00326" },
  "toscano-2024": { id: "toscano-2024", label: "Toscano 2024", citation: "Toscano et al. (2024). From PINNs to PIKANs: recent advances in physics-informed machine learning (PIKANs).", doi: "10.48550/arXiv.2410.13228" },
  "cho-spinn-2023": { id: "cho-spinn-2023", label: "Cho 2023 (SPINN)", citation: "Cho, Nam, Yang, Yun, Hong & Park (2023). Separable physics-informed neural networks (SPINN). NeurIPS 36.", doi: "10.48550/arXiv.2306.15969" },
  "krishnapriyan-2021": { id: "krishnapriyan-2021", label: "Krishnapriyan 2021", citation: "Krishnapriyan, Gholami, Zhe, Kirby & Mahoney (2021). Characterizing possible failure modes in physics-informed neural networks. NeurIPS 34.", doi: "10.48550/arXiv.2109.01050" },
  "raissi-hfm-2020": { id: "raissi-hfm-2020", label: "Raissi 2020 (HFM)", citation: "Raissi, Yazdani & Karniadakis (2020). Hidden fluid mechanics: learning velocity and pressure fields from flow visualizations. Science 367(6481):1026-1030.", doi: "10.1126/science.aaw4741" },
  "cai-2021": { id: "cai-2021", label: "Cai 2021", citation: "Cai, Mao, Wang, Yin & Karniadakis (2021). Physics-informed neural networks (PINNs) for fluid mechanics: a review. Acta Mech. Sinica 37:1727-1738.", doi: "10.1007/s10409-021-01148-1" },
  "grossmann-2024": { id: "grossmann-2024", label: "Grossmann 2024", citation: "Grossmann, Komorowska, Latz & Schonlieb (2024). Can physics-informed neural networks beat the finite element method? IMA J. Appl. Math.", doi: "10.48550/arXiv.2302.04107" },
  "wang-deeponet-2021": { id: "wang-deeponet-2021", label: "Wang 2021 (DeepONet)", citation: "Wang, Wang & Perdikaris (2021). Learning the solution operator of parametric PDEs with physics-informed DeepONets. Science Advances 7(40):eabi8605.", doi: "10.1126/sciadv.abi8605" },
  "karniadakis-2021": { id: "karniadakis-2021", label: "Karniadakis 2021", citation: "Karniadakis, Kevrekidis, Lu, Perdikaris, Wang & Yang (2021). Physics-informed machine learning. Nature Reviews Physics 3:422-440.", doi: "10.1038/s42254-021-00314-5" },
  "tartakovsky-2020": { id: "tartakovsky-2020", label: "Tartakovsky 2020", citation: "Tartakovsky et al. (2020). Physics-informed deep neural networks for learning parameters and constitutive relationships in subsurface flow problems. Water Resources Research 56:e2019WR026731.", doi: "10.1029/2019WR026731" },
  "rasht-behesht-2022": { id: "rasht-behesht-2022", label: "Rasht-Behesht 2022", citation: "Rasht-Behesht, Huber, Shukla & Karniadakis (2022). Physics-informed neural networks (PINNs) for wave propagation and full waveform inversions. JGR Solid Earth 127:e2021JB023120.", doi: "10.1029/2021JB023120" },
  "cuomo-2022": { id: "cuomo-2022", label: "Cuomo 2022", citation: "Cuomo, Schiano Di Cola, Giampaolo, Rozza, Raissi & Piccialli (2022). Scientific machine learning through PINNs: where we are and what's next. J. Sci. Comput. 92:88.", doi: "10.1007/s10915-022-01939-z" },
  "ghia-1982": { id: "ghia-1982", label: "Ghia 1982", citation: "Ghia, Ghia & Shin (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. JCP 48(3):387-411.", doi: "10.1016/0021-9991(82)90058-4" },
};

export function cite(id: string): Citation | undefined {
  return CITATIONS[id];
}
