import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

_CONFIG_CACHE: Optional[Dict[str, Any]] = None

def get_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads config.yaml. Uses caching after first load unless an explicit path is given.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and config_path is None:
        return _CONFIG_CACHE

    if config_path:
        path = Path(config_path)
    else:
        project_root = Path(__file__).resolve().parent.parent.parent
        path = project_root / 'config' / 'config.yaml'

    try:
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                loaded = yaml.safe_load(f) or {}
        else:
            logger.warning(f"Config file not found at {path}. Using defaults.")
            loaded = {}
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        loaded = {}

    if config_path is None:
        _CONFIG_CACHE = loaded
    return loaded or {}

def get_param(section: str, key: str, default: Any = None) -> Any:
    """
    Helper function to get a parameter from the config.
    
    Parameters
    ----------
    section : str
        The section key in the config.
    key : str
        The parameter key within the section.
    default : Any, optional
        Default value if the key or section is not found.
        
    Returns
    -------
    Any
        The parameter value.
    """
    config = get_config()
    return config.get(section, {}).get(key, default)
