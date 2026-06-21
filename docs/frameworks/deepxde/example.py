"""Minimal runnable DeepXDE example — the PINN-Lab recipe in ~60 lines.

Solves the 1D heat equation u_t = alpha*u_xx on (0,1)x(0,1] with a HARD-CONSTRAINT output transform that bakes the
IC u(x,0)=sin(pi x) and the Dirichlet BCs u(0,t)=u(1,t)=0 EXACTLY (no IC/BC loss terms), trains Adam -> L-BFGS,
exports the trained net to ONNX, and checks ONNX-vs-model parity. This is exactly the bridge the web app relies on:
the hard constraint is pure tensor algebra, so it is captured inside the ONNX graph and the browser evaluates the
same function that was validated here.

Run inside the offline venv:
    .venv-pipeline/Scripts/python.exe docs/frameworks/deepxde/example.py     # Windows
    .venv-pipeline/bin/python        docs/frameworks/deepxde/example.py      # Linux/macOS

Exact solution (the validation anchor): u*(x,t) = exp(-alpha*pi^2*t) * sin(pi x).
"""
from __future__ import annotations

import os

os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np


ALPHA = 0.5


def analytic(xt: np.ndarray) -> np.ndarray:
    x, t = xt[:, 0:1], xt[:, 1:2]
    return np.exp(-ALPHA * np.pi**2 * t) * np.sin(np.pi * x)


def main() -> None:
    import deepxde as dde
    import onnxruntime as ort
    import torch

    dde.config.set_random_seed(42)

    # geometry: x in [0,1], t in [0,1]  (here t is just another coordinate, not a TimeDomain, because the IC is hard)
    geom = dde.geometry.Hypercube([0.0, 0.0], [1.0, 1.0])

    def pde(x, u):
        u_t = dde.grad.jacobian(u, x, i=0, j=1)
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        return u_t - ALPHA * u_xx

    data = dde.data.PDE(geom, pde, [], num_domain=4000, num_boundary=0, solution=analytic, num_test=4000)
    net = dde.nn.FNN([2, 48, 48, 48, 1], "tanh", "Glorot normal")

    # HARD CONSTRAINT: u_hat = t*x*(1-x)*N + sin(pi x).  At t=0 -> sin(pi x) (IC); at x=0,1 -> 0 (BC). Pure tensor ops.
    net.apply_output_transform(lambda x, u: x[:, 1:2] * x[:, 0:1] * (1.0 - x[:, 0:1]) * u + torch.sin(np.pi * x[:, 0:1]))

    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
    model.train(iterations=12000, display_every=2000)
    model.compile("L-BFGS")
    model.train()

    # export the trained net to ONNX (dynamic batch axis -> any number of query coordinates)
    net.eval()
    dummy = torch.zeros(1, 2, dtype=torch.float32)
    torch.onnx.export(
        net, (dummy,), "heat1d_example.onnx",
        input_names=["coords"], output_names=["u"],
        dynamic_axes={"coords": {0: "n"}, "u": {0: "n"}},
        opset_version=18, dynamo=True, external_data=False,
    )

    # parity: the ONNX must match model.predict on a random in-domain sample (proves the train -> web bridge)
    rng = np.random.default_rng(43)
    sample = rng.random((512, 2)).astype(np.float64)
    pred_dde = np.asarray(model.predict(sample)).reshape(-1, 1)
    sess = ort.InferenceSession("heat1d_example.onnx", providers=["CPUExecutionProvider"])
    pred_onnx = np.asarray(sess.run(["u"], {"coords": sample.astype(np.float32)})[0]).reshape(-1, 1)
    parity = float(np.max(np.abs(pred_dde - pred_onnx)))

    # report the true error vs the analytic anchor on a grid
    xs = np.linspace(0, 1, 161)
    ts = np.linspace(0, 1, 101)
    xx, tt = np.meshgrid(xs, ts, indexing="ij")
    grid = np.stack([xx.ravel(), tt.ravel()], axis=1)
    u_pred = np.asarray(model.predict(grid)).ravel()
    u_true = analytic(grid).ravel()
    rel_l2 = float(np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true))

    print(f"relative-L2 vs analytic: {rel_l2:.2e}   (expect < 1e-2)")
    print(f"ONNX parity (max abs):   {parity:.2e}   (expect < 1e-4)")
    print("wrote heat1d_example.onnx")


if __name__ == "__main__":
    main()
