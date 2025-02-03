# @title AISDataset

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

LAT, LON, SOG, COG, HEADING, TIMESTAMP = list(range(6))

class AISDataset(Dataset):
    def __init__(self, tracks, lat_bins, lon_bins, sog_bins, cog_bins, mean, divm=360):
        self.tracks = tracks
        self.lat_bins = lat_bins
        self.lon_bins = lon_bins
        self.sog_bins = sog_bins
        self.cog_bins = cog_bins
        self.total_bins = lat_bins + lon_bins + sog_bins + cog_bins
        self.mean = mean
        self.dataFourHot = []
        self.dataTrackdFourHot = []
        self.processdata(divm)

    def processdata(self, divm=360):
        outdata = self.get_four_hot_all_data(divm=divm)
        self.dataTrackdFourHot = outdata
        print('Finished get_four_hot_all_data', len(outdata))
        self.dataFourHot = []
        for i in range(len(outdata)):
            for j in range(len(outdata[i])):
                self.dataFourHot.append(outdata[i][j, :])
        print('Finished processdata')

    def get_four_hot_all_data(self, divm=360):
        results = []
        for i in range(len(self.tracks)):
            results.append(self.sparse_AIS_to_dense(self.tracks[i]))
        return results

    def sparse_AIS_to_dense(self, msg_):
        dense_vect = np.zeros((len(msg_), self.total_bins))
        for i in range(len(msg_)):
            msg = msg_[i]
            dense_vect[i, np.int16(msg[0] * self.lat_bins)] = 1.0
            dense_vect[i, np.int16(msg[1] * self.lon_bins) + self.lat_bins] = 1.0
            dense_vect[i, np.int16(msg[2] * self.sog_bins) + self.lat_bins + self.lon_bins] = 1.0
            dense_vect[i, np.int16(msg[3] * self.cog_bins) + self.lat_bins + self.lon_bins + self.sog_bins] = 1.0
        return dense_vect

    def dense_to_sparse(self, dense_msg, divm=360):
        sparse_msgs = []
        for msg in dense_msg:
            lat_bin = np.argmax(msg[:self.lat_bins])
            lon_bin = np.argmax(msg[self.lat_bins:self.lat_bins + self.lon_bins]) + self.lat_bins
            sog_bin = np.argmax(msg[self.lat_bins + self.lon_bins:self.lat_bins + self.lon_bins + self.sog_bins]) + self.lat_bins + self.lon_bins
            cog_bin = np.argmax(msg[self.lat_bins + self.lon_bins + self.sog_bins:]) + self.lat_bins + self.lon_bins + self.sog_bins
            sparse_msgs.append([
                lat_bin / self.lat_bins,
                (lon_bin - self.lat_bins) / self.lon_bins,
                (sog_bin - self.lat_bins - self.lon_bins) / self.sog_bins,
                (cog_bin - self.lat_bins - self.lon_bins - self.sog_bins) / self.cog_bins
            ])
        return np.array(sparse_msgs)

    def __len__(self):
        return len(self.dataTrackdFourHot)

    def __getitem__(self, idx):
        if idx >= len(self):
            idx = -1
        inputs = self.dataTrackdFourHot[idx]
        return torch.FloatTensor(inputs)


def create_dataloader(tracks, batch_size, lat_bins, lon_bins, sog_bins, cog_bins, shuffle=True):
    dataset = AISDataset(tracks, lat_bins, lon_bins, sog_bins, cog_bins)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
