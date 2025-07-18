import torch
import torch.nn.functional as F


def fast_bbc_mask(mel2ph, mask_length=3, min_segment_length=5, mask_prob=1.):

    batch_size, seq_len = mel2ph.shape
    device = mel2ph.device
    masked_mel2ph = torch.where(mel2ph > 0, mel2ph + 1, mel2ph)

    padded = F.pad(masked_mel2ph, [1, 1], value=-1)  # [B, L+2]

    diff_mask = padded[:, 1:] != padded[:, :-1]  # [B, L+1]
    result = masked_mel2ph.clone()
    for batch_idx in range(batch_size):
        seq = masked_mel2ph[batch_idx]
        boundaries = torch.where(diff_mask[batch_idx])[0]

        if len(boundaries) < 2:
            continue

        starts = boundaries[:-1]
        ends = boundaries[1:]
        lengths = ends - starts
        values = seq[starts]

        valid_mask = (values != 0) & (lengths >= min_segment_length)

        if not valid_mask.any():
            continue

        valid_indices = torch.where(valid_mask)[0]
        random_mask = torch.rand(len(valid_indices), device=device) < mask_prob
        selected_segments = valid_indices[random_mask]

        for seg_idx in selected_segments:
            start = starts[seg_idx]
            end = ends[seg_idx]
            mask_start = max(start, end - mask_length)
            result[batch_idx, mask_start:end] = 1

    return result


def fast_fast_bbc_mask(mel2ph, mask_length=3, min_segment_length=5, mask_prob=0.3):

    batch_size, seq_len = mel2ph.shape
    device = mel2ph.device

    masked_mel2ph = torch.where(mel2ph > 0, mel2ph + 1, mel2ph)


    padded = F.pad(masked_mel2ph, [1, 1], value=-1)  # [B, L+2]


    boundaries = (padded[:, 1:] != padded[:, :-1]).float()  # [B, L+1]

    segment_ids = torch.cumsum(boundaries, dim=1) - 1  # [B, L+1]
    segment_ids = segment_ids[:, :-1]
    max_segments = int(segment_ids.max().item()) + 1

    segment_starts, segment_lengths, segment_values = compute_segment_info_parallel(
        masked_mel2ph, segment_ids, max_segments, seq_len, device
    )


    valid_segments = (segment_values != 0) & (segment_lengths >= min_segment_length)

    random_vals = torch.rand_like(segment_values.float())
    mask_decisions = valid_segments & (random_vals < mask_prob)

    result = apply_masks_parallel(
        masked_mel2ph, segment_starts, segment_lengths, mask_decisions,
        mask_length, seq_len, device
    )

    return result
def compute_segment_info_parallel(masked_mel2ph, segment_ids, max_segments, seq_len, device):

    pos_idx = torch.arange(seq_len, device=device)[None, None, :]  # [1, 1, L]
    seg_idx = torch.arange(max_segments, device=device)[None, :, None]  # [1, S, 1]


    segment_mask = (segment_ids[:, None, :] == seg_idx)  # [B, S, L]


    pos_masked = torch.where(segment_mask, pos_idx, seq_len)
    segment_starts = pos_masked.min(dim=2)[0]  # [B, S]


    segment_lengths = segment_mask.sum(dim=2)  # [B, S]

    first_pos_mask = (pos_masked == segment_starts[:, :, None])
    values_masked = torch.where(first_pos_mask, masked_mel2ph[:, None, :], 0)
    segment_values = values_masked.sum(dim=2)  # [B, S]

    return segment_starts, segment_lengths, segment_values



def apply_masks_parallel(masked_mel2ph, segment_starts, segment_lengths, mask_decisions,
                         mask_length, seq_len, device):

    result = masked_mel2ph.clone()


    pos_indices = torch.arange(seq_len, device=device)[None, None, :]  # [1, 1, L]

    segment_ends = segment_starts + segment_lengths  # [B, S]
    mask_starts = torch.clamp(segment_ends - mask_length, min=segment_starts)  # [B, S]
    mask_ends = segment_ends  # [B, S]


    mask_matrix = (
            (pos_indices >= mask_starts[:, :, None]) &
            (pos_indices < mask_ends[:, :, None]) &
            mask_decisions[:, :, None]
    )  # [B, S, L]


    final_mask = mask_matrix.any(dim=1)  # [B, L]


    result = torch.where(final_mask, torch.ones_like(result), result)

    return result
