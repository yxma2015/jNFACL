import torch
import scanpy as sc
from utils import torch_soft,compute_da,adam,to_numpy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def FALCON(parameters, config):
    eps = 1e-6
    wg, ws = parameters['wg'], parameters['ws']
    b = parameters['bg']
    fg, fs = parameters['fg'], parameters['fs']
    eg = parameters['eg']
    a, lap_ws = parameters['a'], parameters['lap_ws']
    alpha, beta, gamma, eta = parameters['alpha'], parameters['beta'], parameters['gamma'], parameters['eta']

    epochs = parameters["epochs"]
    err = []

    for i in range(epochs):
        # update B
        b = b * (wg @ (fg + eg).T + ws @ fs.T) / (b @ ((fg + eg) @ (fg + eg).T + fs @ fs.T) + eps)
        b = torch.clamp(b, min=eps, max=1)

        # update Fg
        fg = fg * (2 * b.T @ b @ (fg + eg) + 2 * beta * fg @ a.T) / (
                    2 * b.T @ wg + alpha * fg @ lap_ws.T + 2 * beta * fg + 2 * beta * fg @ a @ a.T + eps)
        fg = torch.clamp(fg, min=eps, max=1)

        # update Fs
        fs = fs * (b.T @ ws) / (b.T @ b @ fs + eps)
        fs = torch.clamp(fs, min=eps, max=1)

        # update Eg
        eg = eg * (2 * b.T @ wg) / (2 * b.T @ b @ (fg + eg) + 2 * eg + eps)
        eg = torch.clamp(eg, min=eps, max=1)
        eg = torch_soft(x=eg, th=1e-2)

        # update A (Adam)
        da = compute_da(a=a, neig_indicator=ws, beta=beta, eta=eta, fg=fg, epoch=i + 1)
        a, config = adam(w=a, dw=da, config=config)
        a = torch.clamp(a, min=eps)
        a = torch_soft(x=a, th=1e-3)
        a = 0.5 * (a + a.T)

        # computing components of the objective function sa needed
        err1 = torch.norm(wg - b @ (fg + eg), p="fro") + torch.norm(ws - b @ fs, p="fro") + torch.norm(eg, p="fro")
        err2 = alpha * torch.trace(fg @ lap_ws @ fg.T)
        err3 = beta * (torch.norm(fg - fg @ a, p="fro") + torch.norm(a, p="fro"))

        print(f"## Epoch {i} | err1={err1.item():.6f}, err2={err2.item():.6f}, err3={err3.item():.6f}")
        err.append((err1 + err2 + err3).item())

    return to_numpy(fg), to_numpy(fs), to_numpy(a), to_numpy(eg), to_numpy(err),to_numpy(b)