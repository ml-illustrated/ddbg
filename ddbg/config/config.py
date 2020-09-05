
import logging
import psutil
from fvcore.common.config import CfgNode

def load_yaml_config( filename: str ) -> CfgNode:
    cfg = get_default_cfg()
    cfg.merge_from_file( filename )

    if type( cfg.trainer.num_workers ) == str and cfg.trainer.num_workers == 'auto':
        cfg.trainer.num_workers = psutil.cpu_count(logical=False)
    
    return cfg

def get_default_cfg() -> CfgNode:
    from .defaults import _DefC as DefaultConfig

    return DefaultConfig.clone()

