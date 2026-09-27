from functools import wraps

import torch


def auto_cast(func):
    @wraps(func)
    def wrapper(model_self, batch, *args, **kwargs):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                batch[key] = value.to(torch.bfloat16)
        return func(model_self, batch, *args, **kwargs)

    return wrapper
