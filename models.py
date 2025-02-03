# @title VRNNAnomalyQuality

import torch
from torch import nn
from torch.distributions import Normal, Distribution
import math
from torch import Tensor
# Define Reparameterized Diagonal Gaussian Distribution
class ReparameterizedDiagonalGaussian(Distribution):
    def __init__(self, mu, log_sigma):
        assert mu.shape == log_sigma.shape, "Mu and log_sigma shapes must match."
        self.mu = mu
        self.sigma = log_sigma.exp()

    def sample_epsilon(self):
        return torch.empty_like(self.mu).normal_()

    def rsample(self):
        return self.mu + self.sigma * self.sample_epsilon()

    def log_prob(self, z):
        dist = Normal(self.mu, self.sigma)
        return dist.log_prob(z)


def kl_divergence_diag_gaussians(p: ReparameterizedDiagonalGaussian, q: ReparameterizedDiagonalGaussian) -> Tensor:
    log_var_p = p.sigma.log() * 2
    log_var_q = q.sigma.log() * 2

    kl_div = 0.5 * (
        (log_var_q - log_var_p).sum(dim=-1)
        + ((p.sigma ** 2 + (p.mu - q.mu) ** 2) / (q.sigma ** 2)).sum(dim=-1)
        - p.mu.size(-1)
    )
    return kl_div

# Define Encoder, Prior, and Decoder classes as before
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.phi_x = nn.Sequential(nn.Linear(input_dim, latent_dim), nn.ReLU())
        self.encoder_net = nn.Linear(latent_dim * 2, 2 * latent_dim)

    def forward(self, x_enc, h):
        x_enc = self.phi_x(x_enc)
        enc = self.encoder_net(torch.cat([x_enc, h[0]], dim=-1))
        mu, log_sigma = torch.chunk(enc, 2, dim=-1)
        return ReparameterizedDiagonalGaussian(mu, log_sigma), x_enc


class Prior(nn.Module):
    def __init__(self, latent_dim):
        super(Prior, self).__init__()
        self.prior_net = nn.Linear(latent_dim, 2 * latent_dim)

    def forward(self, h):
        hidden = self.prior_net(h[0])
        mu, log_sigma = torch.chunk(hidden, 2, dim=-1)
        return ReparameterizedDiagonalGaussian(mu, log_sigma)


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.phi_z = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU())
        self.decoder_net = nn.Linear(latent_dim * 2, output_dim)

    def forward(self, z, h):
        z_enc = self.phi_z(z)
        dec_input = torch.cat([z_enc, h[0]], dim=-1)
        logits = self.decoder_net(dec_input)
        return torch.sigmoid(logits), z_enc


# Define GRUState and LSTMState classes
class GRUState(nn.Module):
    def __init__(self, latent_dim):
        super(GRUState, self).__init__()
        self.gru = nn.GRU(latent_dim * 2, latent_dim, batch_first=True)

    def forward(self, x_enc, z_enc, h):
        gru_input = torch.cat([x_enc, z_enc], dim=-1).unsqueeze(1)
        _, h_next = self.gru(gru_input, h.unsqueeze(0))
        return h_next.squeeze(0), None  # No cell state needed


class LSTMState(nn.Module):
    def __init__(self, latent_dim):
        super(LSTMState, self).__init__()
        self.lstm = nn.LSTMCell(latent_dim * 2, latent_dim)

    def forward(self, x_enc, z_enc, h, c):
        lstm_input = torch.cat([x_enc, z_enc], dim=-1)
        h_next, c_next = self.lstm(lstm_input, (h, c))
        return h_next, c_next

class HybridStateUpdate(nn.Module):
    def __init__(self, latent_dim):
        super(HybridStateUpdate, self).__init__()
        self.gru = GRUState(latent_dim)
        self.lstm = LSTMState(latent_dim)
        self.alpha = nn.Parameter(torch.tensor(1.0))  # وزن المزيج بين GRU و LSTM

    def forward(self, x_enc, z_enc, h, c=None):
        h_gru, _ = self.gru(x_enc, z_enc, h)
        h_lstm, c_new = self.lstm(x_enc, z_enc, h, c)
        h_new = self.alpha * h_gru + (self.alpha) * h_lstm

        if c is not None:
            return h_new, c_new
        else:
            return h_new, None
