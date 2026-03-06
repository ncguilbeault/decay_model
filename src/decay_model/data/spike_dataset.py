import torch
from torch.utils import data


class SpikeDataset(data.Dataset):
    def __init__(
        self,
        amps,
        waveforms,
        ch_locs,
        center_locs,
        spike_ids,
        exp_ids,
        min_amps,
        min_waveforms,
    ):
        assert (
            amps.shape[0]
            == waveforms.shape[0]
            == ch_locs.shape[0]
            == center_locs.shape[0]
            == spike_ids.shape[0]
            == exp_ids.shape[0]
            == min_amps.shape[0]
            == min_waveforms.shape[0]
        ), "All input arrays must have the same number of samples"

        self.amps = amps
        self.waveforms = waveforms
        self.ch_locs = ch_locs
        self.center_locs = center_locs
        self.spike_ids = spike_ids
        self.exp_ids = exp_ids
        self.min_amps = min_amps
        self.min_waveforms = min_waveforms

        self.n_spikes = amps.shape[0]

    def __len__(self):
        return self.n_spikes

    def __getitem__(self, index):
        return (
            self.amps[index],
            self.waveforms[index],
            self.ch_locs[index],
            self.center_locs[index],
            self.spike_ids[index],
            self.exp_ids[index],
            self.min_amps[index],
            self.min_waveforms[index],
        )
