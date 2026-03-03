import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import h5py
import numpy as np
import math
import sys
from collections import defaultdict

EPSILON = 1e-6


class VaeSpatialLocalizer(nn.Module):

    def __init__(
        self, waveforms_dim, exps, ch_locs, exp_ids, capacity1=500, capacity2=200, dropout1=0, dropout2=0, prior_var=1.0, device="cuda"
    ):
        super(VaeSpatialLocalizer, self).__init__()

        self.ch_locs = torch.from_numpy(ch_locs).float().to(device)
        self.exp_ids = torch.from_numpy(exp_ids).long().to(device)

        self.capacity1 = capacity1
        self.capacity2 = capacity2
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.prior_varx = prior_var
        self.prior_vary = prior_var
        self.prior_varz = prior_var

        self.fc1 = nn.Linear(waveforms_dim, self.capacity1)
        self.fc2 = nn.Linear(self.capacity1, self.capacity2)
        self.fc_mean = nn.Linear(self.capacity2, 3)
        self.fc_var = nn.Linear(self.capacity2, 3)

        self.batchnorm1 = nn.BatchNorm1d(self.capacity1)
        self.batchnorm2 = nn.BatchNorm1d(self.capacity2)

        self.exps_0 = torch.from_numpy(exps).float().to(device)
        self.exps = torch.from_numpy(exps).float().to(device)
        self.b_s = torch.from_numpy(exps[:,1]).float().to(device)

    def encode(self, waveforms):
        x = self.batchnorm1(self.fc1(torch.flatten(waveforms, start_dim=1)))
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout1)
        x = self.batchnorm2(self.fc2(x))
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout2)
        xyz_mu = self.fc_mean(x)
        xyz_var = F.softplus(self.fc_var(x)) + EPSILON
        return xyz_mu, xyz_var

    def decode(self, sampled_n_loc):
        distances = getTensorDistances(sampled_n_loc, self.ch_locs)
        a_exps = torch.index_select(self.exps, 0, self.exp_ids)
        b_exps = torch.index_select(self.b_s, 0, self.exp_ids)
        a_exps = a_exps.view(a_exps.shape[0], 1)
        b_exps = b_exps.view(b_exps.shape[0], 1)
        recon_amps = -torch.exp(distances * b_exps + a_exps)
        return recon_amps

    def forward(self, amps, waveforms):
        xyz_mu, xyz_var = self.encode(amps, waveforms)
        if self.training:
            x_sample = self.reparameterize_normal(xyz_mu[:, 0], xyz_var[:, 0])
            y_sample = self.reparameterize_normal(xyz_mu[:, 1], xyz_var[:, 1])
            z_sample = self.reparameterize_normal(xyz_mu[:, 2], xyz_var[:, 2])
        else:
            x_sample = xyz_mu[:, 0]
            y_sample = xyz_mu[:, 1]
            z_sample = xyz_mu[:, 2]
        sampled_n_loc = torch.cat((x_sample, y_sample, z_sample), 1)
        recon_amps = self.decode(sampled_n_loc)
        return recon_amps, xyz_mu, xyz_var


def getTensorDistances(n_loc, ch_locs):
    n_loc = n_loc.view(n_loc.shape[0], 1, n_loc.shape[1])
    subtract = (n_loc - ch_locs) ** 2
    summed = torch.sum(subtract, dim=2)
    return torch.sqrt(summed)
    

def reparameterize_normal(mu, var):
    std = torch.sqrt(var)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)


def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def loss_function(
    recon_amps,
    t_amps,
    x_mu,
    x_var,
    y_mu,
    y_var,
    z_mu,
    z_var,
    epochs,
    prior_varx,
    prior_vary,
    prior_varz,
    device,
):
    batch_size = recon_amps.shape[0]
    MSE = (
        torch.sum(
            F.mse_loss(recon_amps, t_amps[:, :, 0], reduction="none") * t_amps[:, :, 1]
        )
        / batch_size
    )

    m_qx = x_mu
    m_px = torch.zeros(m_qx.shape).to(device)
    var_qx = x_var
    var_px = (torch.zeros(var_qx.shape) + prior_varx**2).to(device)
    KLD_x = kl_divergence_normal(m_qx, var_qx, m_px, var_px) / batch_size

    m_qy = y_mu
    m_py = torch.zeros(m_qx.shape).to(device)
    var_qy = y_var
    var_py = (torch.zeros(var_qx.shape) + prior_vary**2).to(device)
    KLD_y = kl_divergence_normal(m_qy, var_qy, m_py, var_py) / batch_size

    m_qz = z_mu
    m_pz = torch.zeros(m_qx.shape).to(device)
    var_qz = z_var
    var_pz = (torch.zeros(var_qx.shape) + prior_varz**2).to(device)
    KLD_z = kl_divergence_normal(m_qz, var_qz, m_pz, var_pz) / batch_size

    return MSE + KLD_x + KLD_y + KLD_z


def kl_divergence_normal(mu_q, var_q, mu_p, var_p):
    kld = torch.sum(
        0.5 * (torch.log(var_p) - torch.log(var_q))
        + torch.div(var_q + (mu_q - mu_p) ** 2, 2 * var_p)
        - 0.5
    )
    return kld


def train(model, device, args, optimizer, train_loader, epoch, train_losses):
    model.train()
    train_loss = 0
    num_loops = 0
    for batch_idx, (
        t_amps,
        t_waveforms,
        t_ch_locs,
        center_loc,
        _,
        exp_ids,
        _,
        _,
    ) in enumerate(train_loader):
        t_amps = t_amps.to(device)
        t_waveforms = t_waveforms.to(device)
        t_ch_locs = t_ch_locs.to(device)
        exp_ids = exp_ids.to(device)
        optimizer.zero_grad()
        recon_amps, x_mu, x_var, y_mu, y_var, z_mu, z_var = model(
            t_amps, t_waveforms, t_ch_locs, exp_ids
        )
        loss = loss_function(
            recon_amps,
            t_amps,
            x_mu,
            x_var,
            y_mu,
            y_var,
            z_mu,
            z_var,
            epoch,
            model.prior_varx,
            model.prior_vary,
            model.prior_varz,
            device,
        )
        loss.backward()
        train_loss += loss.item() / t_amps.shape[0]
        num_loops += 1
        optimizer.step()
    train_losses.append(train_loss / num_loops)
    return model
