import torch
from torch.utils import data

class SpikeDataset(data.Dataset):
    def __init__(self, amps, waveforms, ch_locs, center_locs, spike_ids, exp_ids, min_amps, min_waveforms):
        assert amps.shape[0] == waveforms.shape[0] == ch_locs.shape[0] == center_locs.shape[0] == spike_ids.shape[0] == exp_ids.shape[0] == min_amps.shape[0] == min_waveforms.shape[0], "All input arrays must have the same number of samples"
        
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
        return self.amps[index], self.waveforms[index], self.ch_locs[index], self.center_locs[index], self.spike_ids[index], self.exp_ids[index], self.min_amps[index], self.min_waveforms[index]
    
def create_dataset(waveforms: torch.Tensor, amps: torch.Tensor, channel_locations: torch.Tensor, center_locations: torch.Tensor, spike_cluster_ids: torch.Tensor, batch_size: int, device: str = "cuda") -> data.DataLoader:
    n_events = waveforms.shape[0]

    waveforms_tensor = torch.concat((waveforms, amps[:,:,1].unsqueeze(-1)), axis=2)

    exp_ids = torch.arange(n_events, device=device)
    amps_argsorted = torch.argsort(amps, dim=1)
    min_waveforms_tensor = waveforms_tensor[torch.arange(waveforms_tensor.shape[0]).unsqueeze(1), amps_argsorted[:,:,0], :][:,0,:-1]
    min_amps_tensor = torch.min(min_waveforms_tensor, axis = 1, keepdim = True).values

    training_set = SpikeDataset(amps, waveforms_tensor, channel_locations, center_locations, spike_cluster_ids, exp_ids, min_amps_tensor, min_waveforms_tensor)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)

    return train_loader