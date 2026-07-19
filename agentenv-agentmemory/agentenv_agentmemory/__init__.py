from .memoryarena_webshop_env import MemoryArenaWebShopEnv


def launch() -> None:
    from .launch import launch as run_launch

    run_launch()


__all__ = ["MemoryArenaWebShopEnv", "launch"]
