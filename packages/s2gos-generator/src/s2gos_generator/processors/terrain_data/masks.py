import numpy as np
from PIL import Image
from s2gos_utils.io.paths import mkdir, write_image
from upath import UPath

from ...core.grid import SceneGrid

MAX_PIXEL_VALUE = 255


def generate_buffer_mask(
    buffer_grid: SceneGrid, aoi_size_m: float, output_path: UPath
) -> UPath:
    """Generate the buffer mask: opaque everywhere except over the target area.

    Args:
        buffer_grid: Grid the buffer texture lives on.
        aoi_size_m: Side length of the target area, in metres.
        output_path: UPath where the mask will be saved.

    Returns:
        Path to the generated mask file.
    """

    inside = np.abs(buffer_grid.cell_centres()) < aoi_size_m / 2.0

    mask = np.full((buffer_grid.n, buffer_grid.n), MAX_PIXEL_VALUE, dtype=np.uint8)
    mask[np.outer(inside, inside)] = 0

    mkdir(output_path.parent)
    image = Image.fromarray(mask, mode="L")
    write_image(image, output_path)

    return output_path
