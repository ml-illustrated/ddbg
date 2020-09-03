
import logging
from fvcore.common.config import CfgNode

def load_yaml_config( filename: str ) -> CfgNode:
    cfg = get_default_cfg()
    cfg.merge_from_file( filename )
    
    return cfg

def get_default_cfg() -> CfgNode:
    from .defaults import _DefC as DefaultConfig

    return DefaultConfig.clone()

