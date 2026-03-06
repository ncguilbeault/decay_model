import torch
from torch import nn
from torch.nn import functional as F


class SpikeLocalizationEncoder(nn.Module):

    def __init__(
        self,
        waveforms_dim: int,
        capacity1: int = 500,
        capacity2: int = 200,
        dropout1: float = 0.0,
        dropout2: float = 0.0,
        epsilon: float = 1e-6,
        device: str = "cuda",
    ):
        super(SpikeLocalizationEncoder, self).__init__()

        self.capacity1 = capacity1
        self.capacity2 = capacity2
        self.epsilon = epsilon

        self.fc1 = nn.Linear(waveforms_dim, self.capacity1, device=device)
        self.fc2 = nn.Linear(self.capacity1, self.capacity2, device=device)
        self.fc_mean = nn.Linear(self.capacity2, 3, device=device)
        self.fc_var = nn.Linear(self.capacity2, 3, device=device)

        self.batchnorm1 = nn.BatchNorm1d(self.capacity1, device=device)
        self.batchnorm2 = nn.BatchNorm1d(self.capacity2, device=device)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)

        self.softplus = nn.Softplus()

        self.apply(_weight_init)
        self.to(device)

    def forward(self, waveforms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.flatten(waveforms, start_dim=1)
        x = self.fc1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        xyz_mu = self.fc_mean(x)
        x = self.fc_var(x)
        xyz_var = self.softplus(x) + self.epsilon
        return xyz_mu, xyz_var


class SpikeLocalizationDecoder(nn.Module):

    def __init__(self, exps: torch.Tensor, ch_locs: torch.Tensor, device: str = "cuda"):
        super(SpikeLocalizationDecoder, self).__init__()

        self.ch_locs = ch_locs
        self.exps = exps

        self.to(device)

    def forward(
        self, sampled_n_loc: torch.Tensor, exp_ids: torch.Tensor
    ) -> torch.Tensor:
        n_loc = sampled_n_loc.view(sampled_n_loc.shape[0], 1, sampled_n_loc.shape[1])
        subtract = (n_loc - self.ch_locs[exp_ids]) ** 2
        summed = torch.sum(subtract, dim=2)
        distances = torch.sqrt(summed)
        recon_amps = -torch.exp(
            distances * self.exps[exp_ids, 1].unsqueeze(-1)
            + self.exps[exp_ids, 0].unsqueeze(-1)
        )
        return recon_amps


class SpikeLocalizationVAE(nn.Module):

    def __init__(
        self,
        waveforms_dim: int,
        exps: torch.Tensor,
        ch_locs: torch.Tensor,
        capacity1: int = 500,
        capacity2: int = 200,
        dropout1: float = 0.0,
        dropout2: float = 0.0,
        epsilon: float = 1e-6,
        device: str = "cuda",
    ):
        super(SpikeLocalizationVAE, self).__init__()

        self.encoder = SpikeLocalizationEncoder(
            waveforms_dim, capacity1, capacity2, dropout1, dropout2, epsilon, device
        )
        self.decoder = SpikeLocalizationDecoder(exps, ch_locs, device)

    def reparameterize_normal(
        self, mu: torch.Tensor, var: torch.Tensor
    ) -> torch.Tensor:
        std = var.sqrt()
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(
        self, waveforms: torch.Tensor, exp_ids: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xyz_mu, xyz_var = self.encoder(waveforms)
        if training:
            xyz_mu = self.reparameterize_normal(xyz_mu, xyz_var)
        recon_amps = self.decoder(xyz_mu, exp_ids)
        return recon_amps, xyz_mu, xyz_var


class SpikeLocalizationLossFunc(nn.Module):

    def __init__(self, prior_var: float, device="cuda"):
        super(SpikeLocalizationLossFunc, self).__init__()
        self.prior_var = torch.ones(3, device=device) * (prior_var**2)
        self.to(device)

    def kl_divergence_normal(
        self,
        mu_q: torch.Tensor,
        var_q: torch.Tensor,
        mu_p: torch.Tensor,
        var_p: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sum(
            0.5 * (torch.log(var_p) - torch.log(var_q))
            + torch.div(var_q + (mu_q - mu_p) ** 2, 2 * var_p)
            - 0.5
        )

    def forward(
        self,
        recon_amps: torch.Tensor,
        amps: torch.Tensor,
        xyz_mu: torch.Tensor,
        xyz_var: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = recon_amps.shape[0]
        xyz_muq = torch.zeros_like(xyz_mu)
        xyz_varq = torch.ones_like(xyz_var) * self.prior_var
        loss = (
            torch.sum(
                F.mse_loss(recon_amps, amps[:, :, 0], reduction="none") * amps[:, :, 1]
            )
            + self.kl_divergence_normal(xyz_mu, xyz_var, xyz_muq, xyz_varq)
        ) / batch_size
        return loss


def _weight_init(module: nn.Module) -> None:
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
