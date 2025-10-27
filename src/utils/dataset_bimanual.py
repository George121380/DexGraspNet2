import os
import numpy as np
import torch
from torch.utils.data import Dataset
import MinkowskiEngine as ME

# Joint order aligned with visualization sample (100015_merged_1031.npy)
JOINT_ORDER = [
    'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
    'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
    'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
    'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
    'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0',
    'WRJRx', 'WRJRy', 'WRJRz', 'WRJTx', 'WRJTy', 'WRJTz',
]


def qpos_dict_to_vector(qpos_dict: dict) -> np.ndarray:
    return np.array([float(qpos_dict[name]) for name in JOINT_ORDER], dtype=np.float32)


class BimanualDataset(Dataset):
    """
    Minimal dataset for bimanual diffusion training using pot_data.
    Expects two paths in config.data:
      - pc_path: path to obj_points.npy (N,3)
      - pairs_path: path to grasp_pairs.npy {object_name:str, pairs:dict}
    """

    def __init__(self, config: dict, split: str, is_train: bool = True):
        super().__init__()
        self.full_config = config
        self.config = config.data
        self.is_train = is_train

        self.voxel_size = float(self.config.voxel_size)
        pc_path = self.config.pc_path
        pairs_path = self.config.pairs_path

        if not os.path.exists(pc_path):
            raise FileNotFoundError(f"pc_path not found: {pc_path}")
        if not os.path.exists(pairs_path):
            raise FileNotFoundError(f"pairs_path not found: {pairs_path}")

        self.point_clouds = np.load(pc_path).astype(np.float32)  # (N,3)
        pairs_obj = np.load(pairs_path, allow_pickle=True).item()
        pairs = pairs_obj['pairs']
        # flatten into list for indexing
        self.items = []
        for idx in sorted(pairs.keys()):
            v = pairs[idx]
            left = v['left']
            right = v['right']
            self.items.append(dict(
                left_qpos=qpos_dict_to_vector(left['qpos']),
                right_qpos=qpos_dict_to_vector(right['qpos']),
                left_kp=np.array(left['center_point'], dtype=np.float32),
                right_kp=np.array(right['center_point'], dtype=np.float32),
            ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        pc = self.point_clouds  # (N,3)
        coors = pc / self.voxel_size
        feats = np.ones_like(pc, dtype=np.float32)

        return {
            'point_clouds': pc,                 # (N,3)
            'coors': coors,                     # (N,3)
            'feats': feats,                     # (N,3)
            'keypoint_left': it['left_kp'],     # (3,)
            'keypoint_right': it['right_kp'],   # (3,)
            'qpos_left': it['left_qpos'],       # (28,)
            'qpos_right': it['right_qpos'],     # (28,)
        }


def minkowski_collate_fn_biman(list_data):
    coordinates_batch, features_batch = ME.utils.sparse_collate([d['coors'] for d in list_data], [d['feats'] for d in list_data])
    coordinates_batch, features_batch, original2quantize, quantize2original = ME.utils.sparse_quantize(
        coordinates_batch, features_batch, return_index=True, return_inverse=True)
    res = {
        'coors': coordinates_batch,
        'feats': features_batch,
        'original2quantize': original2quantize,
        'quantize2original': quantize2original,
    }

    # merge the rest
    def collate_fn_(batch):
        first = batch[0]
        out = {}
        for k in first:
            if k in ['coors', 'feats']:
                continue
            arrs = [torch.from_numpy(d[k]) if isinstance(d[k], np.ndarray) else d[k] for d in batch]
            out[k] = torch.stack([a if isinstance(a, torch.Tensor) else torch.from_numpy(a) for a in arrs], 0)
        return out

    res.update(collate_fn_(list_data))
    return res


