import lightning
import torch
from torch import nn
from torch.nn import functional as F

from .components.variational_auto_encoder import VAENet


class VAE(lightning.pytorch.LightningModule):
    def __init__(self, input_height=32, num_classses=10):
        super().__init__()
        self.net = VAENet()
        self.num_classes = num_classses
        self.input_height = input_height
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x, y):
        one_hot_labels = F.one_hot(y, num_classes=self.num_classes)
        # encode x to get the mu and variance parameters
        x_encoded = self.net.encoder(x, one_hot_labels)
        mu, log_var = self.net.fc_mu(x_encoded), self.net.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decode
        x_hat = self.net.decoder(z, one_hot_labels)
        return x_hat, z, mu, std

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = log_qzx - log_pz
        kl = kl.sum(-1)
        return kl

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat, z, mu, std = self.forward(x, y)
        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)
        # kl
        kl = self.kl_divergence(z, mu, std)
        # elbo
        elbo = kl - recon_loss
        elbo = elbo.mean()

        self.log_dict(
            {
                "elbo": elbo,
                "kl": kl.mean(),
                "recon_loss": recon_loss.mean(),
                "reconstruction": recon_loss.mean(),
            }
        )
        return elbo
