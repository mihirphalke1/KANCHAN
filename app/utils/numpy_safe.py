"""
JSON serialisation helper that converts numpy scalars/arrays to Python
primitives so FastAPI's JSONResponse can serialise them.

Kept in its own module so routers can import it directly without creating a
circular dependency on app.main (which imports the routers themselves).
"""
import json

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)):    return bool(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return super().default(obj)


def numpy_safe(data):
    """Round-trip through JSON to convert all numpy scalars to Python primitives."""
    return json.loads(json.dumps(data, cls=_NumpyEncoder))
