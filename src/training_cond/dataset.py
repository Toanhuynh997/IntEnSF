import os
import warnings

import torch
import zipfile
import io
from torch.utils.data import Dataset
from project_utils.paths import DATA_DIR, dataset_trajectory_dir
from project_utils.normalization import (
    load_norm_stats,
    normalize_tensor,
    resolve_norm_stats_path,
)

class DatasetCond(Dataset):
    def __init__(self, data_path_full, img_res_h, img_res_w, img_input_channels, img_cond_channels, name,
                 xflip=False, yflip=False, cache=False, max_mismatches=0, **kwargs):
        """
        Args:
            path (str): Folder or .zip file with .pt files.
            img_res_h, img_res_w (int): Height and width of image/condition arrays.
            img_channels (int): Channels in 'image' array.
            cond_channels (int): Channels in 'conditions' array.
            xflip, yflip (bool): Enable horizontal/vertical deterministic flips.
            cache (bool): Cache data in RAM.
            max_mismatches (int): Max shape mismatches to show before stopping.
        """
        self.data_path_full     = data_path_full
        self.img_res_h          = img_res_h
        self.img_res_w          = img_res_w
        self.img_input_channels = img_input_channels
        self.img_cond_channels  = img_cond_channels
        self.image_shape        = (img_input_channels, img_res_h, img_res_w)
        self.condition_shape    = (img_cond_channels, img_res_h, img_res_w) if img_cond_channels > 0 else None
        self.name               = name
        self._xflip             = xflip
        self._yflip             = yflip
        self._cache             = cache
        self._cached_data       = {}
        self._zipfile           = None

        # File list
        if os.path.isdir(data_path_full):
            self._type = 'dir'
            self._file_list = sorted(f for f in os.listdir(data_path_full) if f.endswith('.pt'))
        elif os.path.isfile(data_path_full) and data_path_full.endswith('.zip'):
            self._type = 'zip'
            self._file_list = sorted(f for f in zipfile.ZipFile(data_path_full, 'r').namelist() if f.endswith('.pt'))
        else:
            raise ValueError(f"Unsupported path: {data_path_full}")

        if len(self._file_list) == 0:
            raise RuntimeError(f"No .pt files found in {data_path_full}")

        # Flipping variants
        self.variants = [(False, False)]
        if xflip and yflip:
            self.variants = [(False, False), (True, False), (False, True), (True, True)]
        elif xflip:
            self.variants = [(False, False), (True, False)]
        elif yflip:
            self.variants = [(False, False), (False, True)]

        self.num_variants = len(self.variants)
        self.dataset_len = len(self._file_list) * self.num_variants

        # Validate file shapes
        self._check_shapes(max_mismatches)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        base_idx = index // self.num_variants
        variant_idx = index % self.num_variants
        apply_xflip, apply_yflip = self.variants[variant_idx]

        fname = self._file_list[base_idx]
        data = self._load_file(fname)

        image = data['targets']
        # cond = data['conditions'][:self.img_cond_channels,...] if self.img_cond_channels > 0 else torch.empty(0)
        cond = data['conditions'][:self.img_cond_channels,...]

        if apply_xflip:
            image = torch.flip(image, dims=[2])
            cond = torch.flip(cond, dims=[2]) if cond.numel() > 0 else cond
        if apply_yflip:
            image = torch.flip(image, dims=[1])
            cond = torch.flip(cond, dims=[1]) if cond.numel() > 0 else cond

        return image, cond

    def _load_file(self, fname):
        if self._cache and fname in self._cached_data:
            return self._cached_data[fname]

        if self._type == 'zip':
            if self._zipfile is None:
                self._zipfile = zipfile.ZipFile(self.data_path_full, 'r')
            with self._zipfile.open(fname, 'r') as f:
                buffer = io.BytesIO(f.read())
                data = torch.load(buffer, weights_only=False)
        else:
            data = torch.load(os.path.join(self.data_path_full, fname), weights_only=False)

        if self._cache:
            self._cached_data[fname] = data
        return data

    def _check_shapes(self, max_mismatches=0):
        print("Checking file shapes...")
        mismatch_count = 0
        has_conditions = self.condition_shape is not None
        
        for fname in self._file_list:
            try:
                data = self._load_file(fname)
                target, condition = data['targets'], data['conditions'][:self.img_cond_channels,...] if has_conditions else None
                
                
                # Always check image shape
                img_shape = tuple(target.shape)
                if img_shape != self.image_shape:
                    print(f"Mismatch in {fname}: read image {img_shape}, expected {self.image_shape}")
                    mismatch_count += 1
                else:
                    # Only check condition shape if conditions exist
                    if has_conditions:
                        cond_shape = tuple(condition.shape)
                        if cond_shape != self.condition_shape:
                            print(f"Mismatch in {fname}: read condition {cond_shape}, expected {self.condition_shape}")
                            mismatch_count += 1
                            
                if mismatch_count >= max_mismatches:
                    print("Max mismatches reached.")
                    break
                
            except Exception as e:
                print(f"Failed to read {fname}: {e}")
                mismatch_count += 1
                if mismatch_count >= max_mismatches:
                    print("Max mismatches reached.")
                    break

        if mismatch_count == 0:
            print("All files match expected shapes.")
        else:
            raise ValueError(f"{mismatch_count} file(s) had shape mismatches.")

    def close(self):
        if self._zipfile is not None:
            self._zipfile.close()
            self._zipfile = None

    # Needed to make it work with multiprocessing (e.g., num_workers > 0)
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_zipfile'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._zipfile = None


def _config_value_is_set(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and value.strip().lower() in ("", "null", "none"):
        return False
    return True


class DatasetPathCond(Dataset):
    """
    Dataset that reads trajectory tensors from `path_*.pt` files and emits
    consecutive prediction pairs:

      conditions = [x_{t-history}, ..., x_{t-1}]
      target     = x_t

    Each `path_*.pt` is expected to contain a float tensor shaped (T, C, H, W).
    """

    def __init__(
        self,
        name,
        path_dir=None,
        data_path_full=None,
        data_root=None,
        trajectory_subdir=None,
        img_res_h=None,
        img_res_w=None,
        img_input_channels=None,
        img_cond_channels=None,
        history=1,
        var_select_target=None,
        var_select_cond=None,
        xflip=False,
        yflip=False,
        cache_paths=False,
        ignore_burn_in_files=True,
        apply_normalization=True,
        norm_stats_path=None,
        max_size=None,
        **kwargs,
    ):
        self.name = name
        self.history = int(history)
        if self.history < 0:
            raise ValueError("history must be >= 0")

        if _config_value_is_set(data_root):
            self.data_root = str(data_root)
        elif _config_value_is_set(kwargs.get("sqg_root")):
            warnings.warn(
                "sqg_root is deprecated; use data_root in dataset kwargs.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.data_root = str(kwargs["sqg_root"])
        else:
            self.data_root = None
        self.trajectory_subdir = (
            str(trajectory_subdir) if _config_value_is_set(trajectory_subdir) else None
        )

        if _config_value_is_set(data_path_full):
            self.data_path_full = str(data_path_full)
        elif _config_value_is_set(path_dir):
            if os.path.isabs(path_dir):
                self.data_path_full = path_dir
            else:
                self.data_path_full = os.path.join(DATA_DIR, path_dir)
        elif self.data_root is not None and self.trajectory_subdir is not None:
            self.data_path_full = dataset_trajectory_dir(self.data_root, self.trajectory_subdir)
        else:
            raise ValueError(
                "DatasetPathCond requires path_dir, data_path_full, or data_root + trajectory_subdir"
            )
        if not os.path.isdir(self.data_path_full):
            raise ValueError(f"path_dir does not exist or is not a directory: {self.data_path_full}")

        fnames = sorted(f for f in os.listdir(self.data_path_full) if f.startswith("path_") and f.endswith(".pt"))
        if ignore_burn_in_files:
            fnames = [f for f in fnames if not f.endswith("_burn_in.pt")]
        if len(fnames) == 0:
            raise RuntimeError(f"No path_*.pt files found in {self.data_path_full}")
        self._file_list = fnames

        probe = torch.load(
            os.path.join(self.data_path_full, self._file_list[0]),
            weights_only=False,
        )
        if not isinstance(probe, torch.Tensor) or probe.ndim != 4:
            raise TypeError(
                f"{self._file_list[0]}: expected a 4D torch.Tensor (T,C,H,W), got {type(probe)}"
            )
        _T, stored_C, stored_H, stored_W = probe.shape

        inferred_res_h = int(stored_H)
        inferred_res_w = int(stored_W)

        if var_select_target is not None:
            self.var_select_target = [int(x) for x in var_select_target]
        else:
            self.var_select_target = list(range(stored_C))
        if var_select_cond is not None:
            self.var_select_cond = [int(x) for x in var_select_cond]
        else:
            self.var_select_cond = list(self.var_select_target)

        if _config_value_is_set(img_res_h):
            self.img_res_h = int(img_res_h)
            if self.img_res_h != inferred_res_h:
                raise ValueError(
                    f"img_res_h ({self.img_res_h}) does not match trajectory H ({inferred_res_h})"
                )
        else:
            self.img_res_h = inferred_res_h
        if _config_value_is_set(img_res_w):
            self.img_res_w = int(img_res_w)
            if self.img_res_w != inferred_res_w:
                raise ValueError(
                    f"img_res_w ({self.img_res_w}) does not match trajectory W ({inferred_res_w})"
                )
        else:
            self.img_res_w = inferred_res_w

        inferred_input_channels = len(self.var_select_target)
        inferred_cond_channels = self.history * len(self.var_select_cond)
        if _config_value_is_set(img_input_channels):
            self.img_input_channels = int(img_input_channels)
            if self.img_input_channels != inferred_input_channels:
                raise ValueError(
                    f"img_input_channels ({self.img_input_channels}) does not match "
                    f"len(var_select_target) ({inferred_input_channels})"
                )
        else:
            self.img_input_channels = inferred_input_channels
        if _config_value_is_set(img_cond_channels):
            self.img_cond_channels = int(img_cond_channels)
            if self.img_cond_channels != inferred_cond_channels:
                raise ValueError(
                    f"img_cond_channels ({self.img_cond_channels}) does not match "
                    f"history*len(var_select_cond) ({inferred_cond_channels})"
                )
        else:
            self.img_cond_channels = inferred_cond_channels

        expected_cond_channels = self.history * len(self.var_select_cond)
        if expected_cond_channels != self.img_cond_channels:
            raise ValueError(
                f"img_cond_channels ({self.img_cond_channels}) must equal history*len(var_select_cond) "
                f"({self.history}*{len(self.var_select_cond)}={expected_cond_channels})"
            )
        if len(self.var_select_target) != self.img_input_channels:
            raise ValueError(
                f"img_input_channels ({self.img_input_channels}) must equal len(var_select_target) "
                f"({len(self.var_select_target)})"
            )

        self._xflip = bool(xflip)
        self._yflip = bool(yflip)
        self._cache_paths = bool(cache_paths)
        self._ignore_burn_in_files = bool(ignore_burn_in_files)
        self._max_size = None if max_size is None else int(max_size)
        self._apply_normalization = bool(apply_normalization)

        self._norm_stats = None
        if self._apply_normalization:
            stats_path = resolve_norm_stats_path(self.data_path_full, norm_stats_path)
            self._norm_stats = load_norm_stats(stats_path)

        # Flipping variants.
        self.variants = [(False, False)]
        if self._xflip and self._yflip:
            self.variants = [(False, False), (True, False), (False, True), (True, True)]
        elif self._xflip:
            self.variants = [(False, False), (True, False)]
        elif self._yflip:
            self.variants = [(False, False), (False, True)]
        self.num_variants = len(self.variants)

        # Read each file once to get lengths (and validate shapes).
        self._traj_lens = []
        self._traj_cache = {}  # file -> tensor
        for fname in self._file_list:
            traj = self._load_traj(fname)
            if traj.ndim != 4:
                raise ValueError(f"{fname}: expected traj.ndim=4 (T,C,H,W), got {traj.shape}")
            T, C, H, W = traj.shape
            if H != self.img_res_h or W != self.img_res_w:
                raise ValueError(f"{fname}: expected H,W=({self.img_res_h},{self.img_res_w}), got ({H},{W})")
            # Need at least history+1 frames to form one (cond,target) pair.
            if T <= self.history:
                self._traj_lens.append(0)
            else:
                self._traj_lens.append(T - self.history)

        # Build cumulative index for fast mapping global index -> (path_idx, t_idx).
        self._cum = [0]
        for n in self._traj_lens:
            self._cum.append(self._cum[-1] + int(n))
        self._num_base = self._cum[-1]
        if self._num_base == 0:
            raise RuntimeError(f"No usable samples found (check history={self.history} vs path lengths).")

        # Apply max_size if requested.
        if self._max_size is not None and self._num_base > self._max_size:
            self._num_base = self._max_size

        self.dataset_len = self._num_base * self.num_variants

    def __len__(self):
        return self.dataset_len

    def _load_traj(self, fname):
        if self._cache_paths and fname in self._traj_cache:
            return self._traj_cache[fname]
        traj = torch.load(os.path.join(self.data_path_full, fname), weights_only=False)
        if not isinstance(traj, torch.Tensor):
            raise TypeError(f"{fname}: expected a torch.Tensor, got {type(traj)}")
        # Keep on CPU; training loop moves to GPU.
        traj = traj.to(torch.float32)
        if self._cache_paths:
            self._traj_cache[fname] = traj
        return traj

    def _find_path_and_offset(self, base_idx):
        # base_idx in [0, _num_base)
        # Find smallest i such that cum[i+1] > base_idx.
        lo, hi = 0, len(self._cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cum[mid + 1] <= base_idx:
                lo = mid + 1
            else:
                hi = mid
        path_idx = lo
        offset = base_idx - self._cum[path_idx]
        return path_idx, offset

    def __getitem__(self, index):
        base_idx = index // self.num_variants
        variant_idx = index % self.num_variants
        apply_xflip, apply_yflip = self.variants[variant_idx]

        if base_idx >= self._num_base:
            # Should not happen, but keep error message explicit.
            raise IndexError(f"Index out of range: {index}")

        path_idx, offset = self._find_path_and_offset(base_idx)
        fname = self._file_list[path_idx]
        traj = self._load_traj(fname)  # (T,C,H,W)

        t = self.history + offset
        # target snapshot
        target = traj[t, self.var_select_target, :, :]  # (C_target,H,W)

        # condition snapshots (history frames)
        if self.history == 0:
            cond = torch.empty((0, self.img_res_h, self.img_res_w), dtype=traj.dtype)
        else:
            block = traj[t - self.history : t, self.var_select_cond, :, :]  # (history,C_cond,H,W)
            cond = block.reshape(self.history * len(self.var_select_cond), self.img_res_h, self.img_res_w)

        if apply_xflip:
            target = torch.flip(target, dims=[2])
            cond = torch.flip(cond, dims=[2]) if cond.numel() > 0 else cond
        if apply_yflip:
            target = torch.flip(target, dims=[1])
            cond = torch.flip(cond, dims=[1]) if cond.numel() > 0 else cond

        if self._norm_stats is not None:
            target = normalize_tensor(target, self._norm_stats)
            if cond.numel() > 0:
                cond = normalize_tensor(cond, self._norm_stats)

        return target, cond
