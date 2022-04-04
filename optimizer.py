from transformers.optimization import AdamW, get_scheduler
from torch import nn
import torch
import math
def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

def create_optimizer(model, args):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        optimizer_kwargs = {"lr": args.learning_rate}
        adam_kwargs = {
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
        }
        decay_parameters = get_parameter_names(model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if n in decay_parameters],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]
        optimizer_kwargs.update(adam_kwargs)
        optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

        return optimizer

def get_warmup_steps(args, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            args.warmup_steps if args.warmup_steps > 0 else math.ceil(num_training_steps * args.warmup_ratio)
        )
        return warmup_steps

def create_scheduler(args, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.
        Args:
            num_training_steps (int): The number of training steps to do.
        """
        
        lr_scheduler = get_scheduler(
            args.lr_scheduler_type,
            optimizer= optimizer if optimizer is None else optimizer,
            num_warmup_steps= get_warmup_steps(args,num_training_steps),
            num_training_steps=num_training_steps,
        )
        return lr_scheduler 