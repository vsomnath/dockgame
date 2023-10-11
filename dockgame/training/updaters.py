import torch
import torch.optim as optim

# Type aliases
Model = torch.nn.Module


class ExponentialMovingAverage:
    """ from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ema.py
    Maintains (exponential) moving average of a set of parameters. """

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates,
                    shadow_params=self.shadow_params)

    def load_state_dict(self, state_dict, device):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = [tensor.to(device) for tensor in state_dict['shadow_params']]


def get_optimizer(
    model: Model, 
    optim_name: str = 'adam', 
    lr: float = 0.001, 
    weight_decay: float = 0.001
) -> torch.optim.Optimizer:
    if optim_name == "adamw":
        optimizer = optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()), 
                              lr=lr, weight_decay=weight_decay, amsgrad=False)
    elif optim_name == "adam":
        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                              lr=lr, weight_decay=weight_decay, amsgrad=False)
    else:
        raise ValueError(f"Optimizer of type {optim_name} is not supported.")

    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer, 
    scheduler_name: str = 'plateau', 
    scheduler_mode: str = 'min', 
    factor: float = 0.5,
    patience: int = 10, 
    min_lr: float = 0
) -> optim.lr_scheduler.ReduceLROnPlateau:
    if scheduler_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, 
                                                         mode=scheduler_mode, factor=factor,
                                                         patience=patience, min_lr=min_lr)
    else:
        print("Setting scheduler to None", flush=True)
        scheduler = None

    return scheduler


def get_ema(model: Model, decay_rate: float) -> ExponentialMovingAverage:
    if decay_rate is None:
        print("Setting EMA to None", flush=True)
        return None

    ema = ExponentialMovingAverage(
        parameters=model.parameters(), decay=decay_rate)
    return ema
