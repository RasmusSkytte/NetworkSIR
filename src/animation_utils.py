import numpy as np
from pathlib import Path


def get_animation_filenames():
    filenames = Path("Data/network").glob(f"v__*.hdf5")
    return [str(file) for file in sorted(filenames)]
