"""Read-only compatibility layer for LeRobot Dataset v2.1.

The v3 dataset implementation intentionally rejects v2.1 metadata.  X-trainer
uses this narrow adapter instead of weakening that global compatibility check or
rewriting recorded demonstrations.
"""

from .dataset import LeRobotDatasetV21
from .metadata import LeRobotDatasetMetadataV21

__all__ = ["LeRobotDatasetMetadataV21", "LeRobotDatasetV21"]
