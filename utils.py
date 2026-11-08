from typing import Dict, Optional
import numpy as np

def denormalize_wind(array: np.ndarray, stats: Optional[Dict[str, float]]) -> np.ndarray:
    if not stats:
        return array
    return array * float(stats.get("wind_std", 1.0)) + float(stats.get("wind_mean", 0.0))

