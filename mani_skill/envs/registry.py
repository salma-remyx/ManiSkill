from mani_skill.envs import BaseEnv

REGISTERED_ENVS: dict[str, type[BaseEnv]] = {}


def register_env(env_id: str):
    """Decorator to register an environment class with a given ID."""

    def decorator(env_cls: type[BaseEnv]) -> type[BaseEnv]:
        REGISTERED_ENVS[env_id] = env_cls
        return env_cls

    return decorator
