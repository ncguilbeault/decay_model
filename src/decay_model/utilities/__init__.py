import xml.etree.ElementTree as ET
import torch

def load_channel_positions_from_xml(xml_path: str, subtract_mean: bool = True, use_3d: bool = True, device: torch.device = "cpu") -> torch.Tensor:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    xpos = root[3][0][-1][-1][-2].attrib
    ypos = root[3][0][-1][-1][-1].attrib

    assert len(xpos) == len(ypos), "The number of x and y channel positions must be the same"
    assert all(key in ypos for key in xpos) and all(key in xpos for key in ypos), "The channel keys in x and y positions must match"

    channel_ids = []
    channel_keys = xpos.keys()
    for channel in channel_keys:
        chan_id = int(channel.split("CH")[1])
        channel_ids.append(chan_id)

    max_channels = max(channel_ids) + 1
    channel_positions = torch.zeros((max_channels, 3), device=device)

    for channel_id in channel_ids:
        x = float(xpos[f"CH{channel_id}"])
        y = float(ypos[f"CH{channel_id}"])
        if use_3d:
            pos = torch.tensor([0, x, y], device=device)
        else:
            pos = torch.tensor([x, y], device=device)
        channel_positions[channel_id] = pos

    idx_to_keep = torch.where(torch.any(channel_positions!=0, dim=1))[0]
    channel_positions = channel_positions[idx_to_keep]

    if subtract_mean:
        channel_positions -= torch.mean(channel_positions, dim=0)
    
    return channel_positions

def pad_channel_positions(channel_positions: torch.Tensor, x_spacing: float | None = None, y_spacing: float | None = None) -> torch.Tensor:

    if channel_positions.shape[1] == 2:
        x_coords = torch.unique(channel_positions[:, 0])
        y_coords = torch.unique(channel_positions[:, 1])
    else:
        x_coords = torch.unique(channel_positions[:, 1])
        y_coords = torch.unique(channel_positions[:, 2])
    
    assert x_spacing is not None or x_coords.shape[0] > 1, "If x_spacing is not provided, there must be more than one unique x coordinate to infer spacing"

    assert y_spacing is not None or y_coords.shape[0] > 1, "If y_spacing is not provided, there must be more than one unique y coordinate to infer spacing"

    if x_spacing is None:
        x_spacing = x_coords[-1] - x_coords[0] + (x_coords[1] - x_coords[0])
    
    if y_spacing is None:
        y_spacing = y_coords[-1] - y_coords[0] + (y_coords[1] - y_coords[0])

    offsets = torch.tensor([[0, i * x_spacing, j * y_spacing] for i in range(-1, 2) for j in range(-1, 2) if (i, j) != (0, 0)], device=channel_positions.device)

    shifted_positions = channel_positions.unsqueeze(1) + offsets.unsqueeze(0)
    return torch.vstack((channel_positions, shifted_positions.view(-1, channel_positions.shape[1])))

    