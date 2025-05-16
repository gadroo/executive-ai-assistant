import yaml
from pathlib import Path
import asyncio
import functools

_ROOT = Path(__file__).absolute().parent


def _read_yaml_file():
    with open(_ROOT.joinpath("config.yaml")) as stream:
        return yaml.safe_load(stream)


async def get_config_async(config: dict):
    # This loads things either ALL from configurable, or
    # all from the config.yaml
    # This is done intentionally to enforce an "all or nothing" configuration
    if "email" in config["configurable"]:
        return config["configurable"]
    else:
        # Use asyncio.to_thread to avoid blocking the event loop
        return await asyncio.to_thread(_read_yaml_file)


async def get_config(config: dict):
    # This loads things either ALL from configurable, or
    # all from the config.yaml
    # This is done intentionally to enforce an "all or nothing" configuration
    if "email" in config["configurable"]:
        return config["configurable"]
    else:
        # Use asyncio.to_thread to avoid blocking the event loop
        return await asyncio.to_thread(_read_yaml_file)


# For backward compatibility - sync version that should be replaced in all callsites
def get_config_sync(config: dict):
    """Synchronous version of get_config - should be replaced with async version."""
    if "email" in config["configurable"]:
        return config["configurable"]
    else:
        with open(_ROOT.joinpath("config.yaml")) as stream:
            return yaml.safe_load(stream)
