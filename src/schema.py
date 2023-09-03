from dataclasses import dataclass
from omegaconf import OmegaConf


@dataclass
class Cfg:
    learning_rate: float = 0.001
    dropout_rate: float = 0.3
    lambda_for_cl: float = 0.1
    substitution_rate: float = 0.3
    masking_rate: float = 0.3
    cropping_rate: float = 0.3
    batch_size: int = 256
    sequence_len: int = 30
    embedding_dim: int = 64
    early_stop_patience: int = 40
    warmup_steps: int = 1500
    epochs: int = 150
    clip_norm: float = 5.0
    log_interval: int = 10
    temperature: float = 0.1
    try_num: str = '1'
    run_mode: str = 'train'
    data_type: str = 'RentTheRunway'
    train_type: str = 'CL'
    aug_mode: str = 'substitute'
    model_trainable: bool = True
    rec_num_dense_layer: int = 2
    
    k: int = 10
    sparsity: float = 0.1


# Duck typing for accessing dataclass variable
def load_config(path: str) -> Cfg:
    schema = OmegaConf.structured(Cfg())

    # duck typing
    config: Cfg = OmegaConf.merge(schema, OmegaConf.load(path))

    return config