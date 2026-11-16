import os

# Set project root as two levels above this file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

DATA_DIR = os.path.join(PROJECT_ROOT, 'datasets')
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'configs')
CKPT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')


def dataset_trajectory_dir(data_root: str, trajectory_subdir: str) -> str:
    """Resolve datasets/<data_root>/<trajectory_subdir> (paths come from YAML config)."""
    return os.path.join(DATA_DIR, data_root, trajectory_subdir)
