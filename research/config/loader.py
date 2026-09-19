"""
YAML configuration loader for the research subsystem.
"""

import os
import yaml
import logging
from typing import Optional, Dict, List, Any

# Configure logging
logger = logging.getLogger(__name__)

# Global cache for the configuration
_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def load_protocol(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads the research protocol configuration from a YAML file.
    Caches the result after the first load unless a new path is provided.
    
    Args:
        path (Optional[str]): Path to the YAML configuration file. If None,
            defaults to `protocol.yaml` in the same directory as this file.
            
    Returns:
        Dict[str, Any]: The parsed configuration dictionary.
        
    Raises:
        ValueError: If the file does not exist or fails to parse.
    """
    global _CONFIG_CACHE
    
    if path is None:
        # Default to protocol.yaml in the same directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, "protocol.yaml")
        
    if _CONFIG_CACHE is not None and path == os.path.join(os.path.dirname(os.path.abspath(__file__)), "protocol.yaml"):
        logger.debug("Returning cached configuration.")
        return _CONFIG_CACHE
        
    if not os.path.isfile(path):
        raise ValueError(f"Configuration file not found: {path}")
        
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        if not isinstance(config, dict):
            raise ValueError(f"Invalid configuration format in {path}. Expected a dictionary.")
            
        logger.info(f"Successfully loaded configuration from {path}")
        
        # Cache if we loaded the default
        if path == os.path.join(os.path.dirname(os.path.abspath(__file__)), "protocol.yaml"):
            _CONFIG_CACHE = config
            
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file at {path}: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error loading configuration from {path}: {e}")


def get_section(config: Dict[str, Any], section: str) -> Dict[str, Any]:
    """
    Retrieves a specific section from the configuration dictionary.
    
    Args:
        config (Dict[str, Any]): The full configuration dictionary.
        section (str): The name of the section to retrieve.
        
    Returns:
        Dict[str, Any]: The configuration section.
        
    Raises:
        ValueError: If the section does not exist in the configuration.
    """
    if section not in config:
        raise ValueError(f"Required section '{section}' missing from configuration.")
        
    section_data = config[section]
    if not isinstance(section_data, dict):
        raise ValueError(f"Section '{section}' must be a dictionary.")
        
    return section_data


def get_normalizations(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extracts the normalization specifications from the configuration.
    Expects them under the root `normalizations` key.
    
    Args:
        config (Dict[str, Any]): The full configuration dictionary.
        
    Returns:
        List[Dict[str, Any]]: A list of normalization specification dictionaries.
        
    Raises:
        ValueError: If the normalizations section is missing or invalid.
    """
    if "normalizations" not in config:
        raise ValueError("Missing 'normalizations' list in configuration.")
        
    normalizations = config["normalizations"]
    if not isinstance(normalizations, list):
        raise ValueError("'normalizations' must be a list of dictionaries.")
        
    return normalizations


def get_characteristics(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Extracts the factor to characteristics mapping from the configuration.
    Expects them under the root `characteristics` key.
    
    Args:
        config (Dict[str, Any]): The full configuration dictionary.
        
    Returns:
        Dict[str, List[str]]: A dictionary mapping factor names to lists of characteristic names.
        
    Raises:
        ValueError: If the characteristics section is missing or invalid.
    """
    if "characteristics" not in config:
        raise ValueError("Missing 'characteristics' mapping in configuration.")
        
    characteristics = config["characteristics"]
    if not isinstance(characteristics, dict):
        raise ValueError("'characteristics' must be a dictionary mapping factors to lists of features.")
        
    return characteristics
