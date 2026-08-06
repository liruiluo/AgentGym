"""Native SWE-smith environment package."""

from .dataset import SwesmithDataset
from .environment import SwesmithEpisodeManager
from .grader import SwesmithHiddenGrader
from .image_manifest import SwesmithImageManifest
from .sandbox import LinuxNamespaceEpisodeSandbox
from .workspace import SwesmithWorkspaceMaterializer

__all__ = [
    "LinuxNamespaceEpisodeSandbox",
    "SwesmithDataset",
    "SwesmithEpisodeManager",
    "SwesmithHiddenGrader",
    "SwesmithImageManifest",
    "SwesmithWorkspaceMaterializer",
]


def launch() -> None:
    from .launch import launch as _launch

    _launch()
