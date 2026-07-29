import os
from pathlib import Path

__version__ = "4.0.0"

# ---------------------------------------------------------------------------- #
# Setup paths
# ---------------------------------------------------------------------------- #
PACKAGE_DIR = Path(__file__).parent.resolve()
PACKAGE_ASSET_DIR = PACKAGE_DIR / "assets"
# Non-package data
ASSET_DIR = Path(
    os.path.join(
        os.getenv("MS_ASSET_DIR", os.path.join(os.path.expanduser("~"), ".maniskill")),
        "data",
    )
)
DEMO_DIR = Path(
    os.path.join(
        os.getenv("MS_ASSET_DIR", os.path.join(os.path.expanduser("~"), ".maniskill")),
        "demos",
    )
)
