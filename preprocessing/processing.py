import numpy as np

def _process_video(frames):
    frames = np.asarray(frames, dtype=np.float64)

    if frames.size == 0:
        return frames.reshape((0, 3))

    # Already extracted per-frame RGB values: (N, 3)
    if frames.ndim == 2 and frames.shape[1] == 3:
        return frames

    # Single RGB vector: (3,)
    if frames.ndim == 1 and frames.shape[0] == 3:
        return frames.reshape((1, 3))

    # Raw video frames: (N, H, W, 3) or (H, W, 3)
    if frames.ndim == 3 and frames.shape[2] == 3:
        return np.array([frame.mean(axis=(0, 1)) for frame in frames], dtype=np.float64)

    if frames.ndim == 4 and frames.shape[-1] == 3:
        return np.array([frame.mean(axis=(0, 1)) for frame in frames], dtype=np.float64)

    raise ValueError(
        "_process_video expected RGB frames shaped (H, W, 3), (N, H, W, 3), "
        "or per-frame RGB values shaped (N, 3). "
        f"Got shape {frames.shape}."
    )