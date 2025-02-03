# @title VRNNVotingModel
from models import *

class VRNNVotingModel(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VRNNVotingModel, self).__init__()
        self.vrnn_gru = VRNNAnomalyQuality(input_dim, latent_dim, state_type="GRU")
        self.vrnn_lstm = VRNNAnomalyQuality(input_dim, latent_dim, state_type="LSTM")
        self.state_type = "Voting"

    def forward(self, x, beta=1.0):
        # Train both models and get their losses
        loss_gru, recon_loss_gru, kl_div_gru = self.vrnn_gru(x, beta)
        loss_lstm, recon_loss_lstm, kl_div_lstm = self.vrnn_lstm(x, beta)

        # Take the average of the losses as the final loss (voting mechanism)
        total_loss = (loss_gru + loss_lstm) / 2
        recon_loss = (recon_loss_gru + recon_loss_lstm) / 2
        kl_divergence = (kl_div_gru + kl_div_lstm) / 2

        return total_loss, recon_loss, kl_divergence

    def calculate_anomaly_rate(self, inputs):
        # Calculate anomaly rate for both GRU and LSTM models
        anomaly_rate_gru = self.vrnn_gru.calculate_anomaly_rate(inputs)
        anomaly_rate_lstm = self.vrnn_lstm.calculate_anomaly_rate(inputs)

        # Take the average anomaly rate (voting mechanism)
        anomaly_rate = (anomaly_rate_gru + anomaly_rate_lstm) / 2
        return anomaly_rate

    def calc_mi(self, inputs):
        # Calculate mutual information for both GRU and LSTM models
        mi_gru = self.vrnn_gru.calc_mi(inputs)
        mi_lstm = self.vrnn_lstm.calc_mi(inputs)

        # Take the average mutual information (voting mechanism)
        mi = (mi_gru + mi_lstm) / 2
        return mi

    def get_logis(self, x):

       logis_gru=self.vrnn_gru.get_logis(x)
       logis_lstm=self.vrnn_lstm.get_logis(x)
       logis=(logis_gru+logis_lstm)/2
       return logis
