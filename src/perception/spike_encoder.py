import cupy as cp

def encode_spatial_location(
    img_gpu,
    pop_info,
    pop_map,
    target_color,
    color_threshold=0.8,
    detection_threshold=10
):
    if img_gpu is None or img_gpu.shape[0] == 0:
        return None
    height, width, _ = img_gpu.shape
    col_width = width // 3
    # define columns
    regions = {
        "Vision_Left": (0, col_width),
        "Vision_Center": (col_width, 2 * col_width),
        "Vision_Right": (2 * col_width, width)
    }
    all_spikes = []
    # process each region
    for pop_name, (start_col, end_col) in regions.items():
        if pop_name not in pop_map:
            continue
        region_img = img_gpu[:, start_col:end_col, :]
        distances = cp.linalg.norm(region_img - target_color, axis=2)
        detected_pixels = cp.sum(distances < color_threshold)
        if detected_pixels > detection_threshold:
            pop = pop_info[pop_map[pop_name]]
            # fire a proportion of neurons based on how many pixels are detected
            proportion_seen = min(1.0, detected_pixels.item() / (height * col_width * 0.1))
            num_firing = int(pop['count'] * proportion_seen)
            if num_firing > 0:
                indices = cp.random.choice(
                    cp.arange(pop['start'], pop['end']),
                    size=num_firing,
                    replace=False
                )
                all_spikes.append(indices)
    if not all_spikes:
        return None
    return cp.concatenate(all_spikes).astype(cp.int32)