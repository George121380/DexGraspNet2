### Bimanual Diffusion - Implementation Progress

Date: (auto)

1) Created `src/network_biman/` package with `BimanualDiffusion` skeleton
   - Backbone reuse via `get_backbone`
   - Diffusion head via `MLPWrapper` + `GaussianDiffusion1D`
   - Conditioning builder supports global feature; if `use_keypoints` and point-wise features exist, fuses averaged left/right kp features (gathered by nearest neighbor)

2) Wired model entry
   - Updated `src/network/model.py`: added `'bimanual_diff'` → `BimanualDiffusion`

3) Forward/Sample API
   - `forward(data)`: builds target by concatenating `qpos_left` and `qpos_right` (expects J+6 each or pre-packed)
   - `sample(data, sample_num)`: returns `(left, right, log_prob)` with shape `(B, K, J+6)` each

Next steps
- Make `use_keypoints` gating explicit in configs
- Added dataset loader `src/utils/dataset_bimanual.py` with `JOINT_ORDER`, vector mapping, kp+qpos fields, and Minkowski collate
- Integrated dataset will require a minimal config stub:
  - data.pc_path: path to obj_points.npy
  - data.pairs_path: path to grasp_pairs.npy
  - model: {type: bimanual_diff, backbone: sparseconv, feature_dim: 256, joint_num: 22, use_keypoints: true, diffusion: {...}}
  - Then swap DataLoader to use `BimanualDataset` + `minkowski_collate_fn_biman`
  - Run a quick training smoke test via `src/train.py`


