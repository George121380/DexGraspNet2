## 双手模式 vs 单手模式：关键差异概览

- **模型类型与入口**
  - 单手：`model/type` 取值 `graspness_diffusion`、`graspness_isa`、`graspness_cvae`（实现于 `src/network/graspness_sample.py`），或 `glob_diff`（实现于 `src/network/diffusion_sample.py`）。由 `src/network/model.py:get_model` 分发。
  - 双手：`model/type` 为 `bimanual_diff`（实现于 `src/network_biman/bimanual_diffusion.py`）。同样由 `src/network/model.py:get_model` 分发。

- **数据集与输入格式**
  - 单手数据：`src/utils/dataset.py:GraspNetDataset`
    - 字段包含：`point_clouds`、`coors`、`feats`、`seg`、`objectness`、`graspness`、`rot(K,3,3)`、`trans(K,3)`、`centers(K)`、`qpos(K,J)`、可选 `edge`。
    - 依赖场景渲染数据与抓取标注，训练时按最近点选择 K 个 GT 抓取；支持数据增强（绕 z 轴随机旋转）。
    - collate：`src/utils/dataset.py:minkowski_collate_fn`，产出 MinkowskiEngine 稀疏张量所需的 `coors/feats` 及 `quantize2original` 等。
  - 双手数据：`src/utils/dataset_bimanual.py:BimanualDataset`
    - 期望 pot_data：单个对象点云 `obj_points.npy` 与配对抓取 `grasp_pairs.npy`。
    - 字段包含：`point_clouds`、`coors`、`feats`、`keypoint_left/right(3,)`、`qpos_left/right(28,)`。
    - 不包含单手中的 `seg/objectness/graspness`；当前未实现数据增强。
    - collate：`src/utils/dataset_bimanual.py:minkowski_collate_fn_biman`。

- **特征提取与条件构建**（两者共用 Minkowski 稀疏 UNet 背骨）
  - 单手：提取逐点特征 `(B,N,C)`，用以逐点评估抓取性与作为条件生成头的局部条件。
  - 双手：先做全局池化得到 `(B,C)` 条件；若配置 `use_keypoints=true`，再从逐点特征处按最近邻聚合 `keypoint_left/right` 的特征并与全局特征融合（`global + 0.5*(kp_l+kp_r)`）。

- **训练目标与损失**
  - 单手（`src/network/graspness_sample.py`）：
    - 物体/抓取性：`objectness` 交叉熵 + `graspness` SmoothL1（仅在物体点上计入）。
    - 位姿/关节：
      - 若 `graspness_diffusion`：扩散损失（旋转表示 + 欧氏量），可选 `dist_joint` 将关节并入扩散向量；否则关节由额外 MLP 回归并以 SmoothL1 监督。
      - 若 `graspness_isa`：MLP 直接回归四元数与欧氏量，监督为欧氏 L1 + SO(3) 角度差。
      - 若 `graspness_cvae`：CVAE 重构损失（代码中聚合到 `cvae_loss`）。
  - 双手（`src/network_biman/bimanual_diffusion.py`）：
    - 目标为左右手连接的向量 `concat(left, right)`，每个长度为 `(J + 6)`，其中 `6` 为手腕刚体 DOF（`WRJRx/WRJRy/WRJRz/WRJTx/WRJTy/WRJTz`）。
    - 仅使用扩散损失（`GaussianDiffusion1D.calculate_loss`）。不包含单手中的 `objectness/graspness/joint` 等额外损失项。

- **推理与采样流程**
  - 单手：
    - 逐点得到 `objectness/graspness`，按抓取性阈值或比例选择“抓取点”，再用 FPS/邻域策略采样种子点。
    - 条件生成头（扩散/MLP）基于种子点特征输出旋转、平移增量与关节；平移为体素中心上的偏移，经缩放还原到相机坐标系。
    - 打分：`score = log_prob + graspness * graspness_scale`，用于排序筛选。
    - 输出：`rot(B,K,3,3)`、`trans(B,K,3)`、`joints(B,K,J)`、`score(B,K)`。
  - 双手：
    - 构建全局条件（可融合关键点特征），复制 `sample_num` 次作为条件输入扩散采样。
    - 扩散输出 `(B, sample_num, 2*(J+6))`，按中间位置切分为左右手向量；当前未与抓取性分数融合打分。
    - 输出：`left(B,K,J+6)`、`right(B,K,J+6)`、`log_prob(B,K)`。

- **输出语义差异**
  - 单手：直接给出显式位姿（旋转矩阵 R 与平移 t）及手部关节；更易与点云/相机系对齐与可视化。
  - 双手：给出左右手的关节+手腕 6 DoF 的参数向量，手腕位姿解算/可视化通常在下游（见可视化与能量评估工具 `visualization/utils/bimanual_*`）。

- **配置与训练脚本差异**
  - 单手：多种 `model/type` 与损失权重；数据段落包含 `graspness_data`、`k`、`sample_total`、`scene_fraction`、`fraction` 等控制采样与标注装载的键。
  - 双手：`configs/network/train_bimanual.yaml` 与 `train_bimanual_smoke.yaml`，核心键：`model.type=bimanual_diff`、`joint_num`（如 22）、`use_keypoints`、pot_data 路径（`pc_path`/`pairs_path`）。
  - `src/train.py` 中的数据加载与评估：当 `model.type == 'bimanual_diff'` 走 `BimanualDataset` 与 `minkowski_collate_fn_biman`，且不构建单手的验证集列表。

- **工程层面补充差异**
  - 单手依赖分割、抓取性监督与近点采样策略，强调“点-局部”条件。
  - 双手强调“全局-双臂协同”与关键点提示（可选），目标直接在联合空间中学习左右手的参数耦合。

以上差异可结合下列文件快速定位：
- 模型分发：`src/network/model.py`
- 单手核心：`src/network/graspness_sample.py`、`src/network/diffusion_sample.py`、`src/utils/dataset.py`
- 双手核心：`src/network_biman/bimanual_diffusion.py`、`src/utils/dataset_bimanual.py`
- 配置示例：`configs/network/train_bimanual.yaml`、`configs/network/train_bimanual_smoke.yaml`



