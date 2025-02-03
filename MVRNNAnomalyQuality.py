from models import *

class MVRNNAnomalyQuality(nn.Module):
    def __init__(self, input_dim, latent_dim, state_type="Hybrid"):
        super(VRNNAnomalyQuality, self).__init__()

        self.encoder = Encoder(input_dim, latent_dim)
        self.prior = Prior(latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

        # Choose the state mechanism
        if state_type == "GRU":
            self.state_update = GRUState(latent_dim)
        elif state_type == "LSTM":
            self.state_update = LSTMState(latent_dim)
        elif state_type == "Hybrid":
            self.state_update = HybridStateUpdate(latent_dim)
        else:
            raise ValueError("state_type must be 'GRU', 'LSTM', or 'Hybrid'")

        self.state_type = state_type
        self.h_0 = torch.zeros(1, latent_dim)
        self.c_0 = torch.zeros(1, latent_dim) if state_type in ["LSTM", "Hybrid"] else None
        self.threshold = 0.0001

    def forward(self, x, beta=1.0):
        batch_size, seq_len, _ = x.size()
        h = self.h_0.expand(batch_size, -1).to(x.device)
        c = self.c_0.expand(batch_size, -1).to(x.device) if self.c_0 is not None else None
        listlogis = []

        total_loss, kl_divergence, recon_loss = 0, 0, 0

        for t in range(seq_len):
            x_t = x[:, t, :]

            prior_dist = self.prior((h, c) if self.c_0 is not None else (h, None))
            posterior_dist, x_enc = self.encoder(x_t, (h, c) if self.c_0 is not None else (h, None))

            z_t = posterior_dist.rsample()
            x_recon, z_enc = self.decoder(z_t, (h, c) if self.c_0 is not None else (h, None))

            if self.state_type == "Hybrid":
                h, c = self.state_update(x_enc, z_enc, h, c)
            elif self.state_type == "LSTM":
                h, c = self.state_update(x_enc, z_enc, h, c)
            else:
                h, _ = self.state_update(x_enc, z_enc, h)

            kl_div = kl_divergence_diag_gaussians(posterior_dist, prior_dist).sum(dim=-1)
            kl_divergence += kl_div.mean()
            recon_loss += nn.functional.binary_cross_entropy(x_recon, x_t, reduction='sum')
            listlogis.append(x_recon)

        recon_loss /= seq_len*10
        kl_divergence /= seq_len*10
        total_loss = recon_loss + beta * kl_divergence
        return total_loss, recon_loss, kl_divergence

    def get_logis(self, x):
        batch_size, seq_len, _ = x.size()
        h = self.h_0.expand(batch_size, -1).to(x.device)
        c = self.c_0.expand(batch_size, -1).to(x.device) if self.c_0 is not None else None
        listlogis = []

        for t in range(seq_len):
            x_t = x[:, t, :]

            prior_dist = self.prior((h, c) if self.c_0 is not None else (h, None))
            posterior_dist, x_enc = self.encoder(x_t, (h, c) if self.c_0 is not None else (h, None))
            z_t = posterior_dist.rsample()
            x_recon, z_enc = self.decoder(z_t, (h, c) if self.c_0 is not None else (h, None))
            listlogis.append(x_recon)

            if self.state_type == "Hybrid":
                h, c = self.state_update(x_enc, z_enc, h, c)
            elif self.state_type == "LSTM":
                h, c = self.state_update(x_enc, z_enc, h, c)
            else:
                h, _ = self.state_update(x_enc, z_enc, h)

        x_rect = torch.stack(listlogis, dim=0)
        x_rect = x_rect.permute(1, 0, 2)
        return x_rect

    def calculate_anomaly_rate(self, inputs):
        batch_size = inputs.size(0)
        h = self.h_0.expand(batch_size, -1).contiguous().to(inputs.device)
        c = self.c_0.expand(batch_size, -1).contiguous().to(inputs.device) if self.c_0 is not None else None

        total_anomalies = 0
        total_points = 0

        for t in range(inputs.size(1)):
            x = inputs[:, t, :]
            posterior_dist, x_enc = self.encoder(x, (h, c) if self.c_0 is not None else (h, None))
            z = posterior_dist.rsample()
            x_recon, z_enc = self.decoder(z, (h, c) if self.c_0 is not None else (h, None))

            diff = torch.abs(x - x_recon)
            anomalies = (diff > self.threshold).sum(dim=-1)

            total_anomalies += anomalies.sum().item()
            total_points += x.numel()

            if self.state_type == "Hybrid":
                h, c = self.state_update(x_enc, z_enc, h, c)
            elif self.state_type == "LSTM":
                h, c = self.state_update(x_enc, z_enc, h, c)
            else:
                h, _ = self.state_update(x_enc, z_enc, h)

        anomaly_rate = total_anomalies / total_points
        return anomaly_rate

    def calc_mi(self, inputs):
        batch_size = inputs.size(0)
        h = self.h_0.expand(batch_size, -1).contiguous().to(inputs.device)
        c = self.c_0.expand(batch_size, -1).contiguous().to(inputs.device) if self.c_0 is not None else None

        neg_entropy = 0
        log_qz = 0

        for t in range(inputs.size(1)):
            x = inputs[:, t, :]
            posterior_dist, x_enc = self.encoder(x, (h, c) if self.c_0 is not None else (h, None))
            pz = self.prior((h, c) if self.c_0 is not None else h)

            mu = posterior_dist.mu
            logsigma = torch.log(posterior_dist.sigma)
            z = posterior_dist.rsample()
            _, z_enc = self.decoder(z, (h, c) if self.c_0 is not None else (h, None))

            if self.state_type == "Hybrid":
                h, c = self.state_update(x_enc, z_enc, h, c)
            elif self.state_type == "LSTM":
                h, c = self.state_update(x_enc, z_enc, h, c)
            else:
                h, _ = self.state_update(x_enc, z_enc, h)

            neg_entropy += (-0.5 * self.encoder.encoder_net.out_features // 2 * math.log(2 * math.pi)
                            - 0.5 * (1 + 2 * logsigma).sum(-1)).mean()

            var = logsigma.exp() ** 2
            z = z.unsqueeze(1)
            mu = mu.unsqueeze(0)
            logsigma = logsigma.unsqueeze(0)
            dev = z - mu
            log_density = -0.5 * (dev ** 2 / var).sum(dim=-1) - 0.5 * (
                self.encoder.encoder_net.out_features // 2 * math.log(2 * math.pi) + (2 * logsigma).sum(dim=-1))
            log_qz1 = torch.logsumexp(log_density, dim=1) - math.log(batch_size)
            log_qz += log_qz1.mean(-1)

        mi = (neg_entropy / inputs.size(1)) - (log_qz / inputs.size(1))
        return mi
