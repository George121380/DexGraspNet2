### 双手模式（bimanual_diff）当前网络结构（2025-10-27）

本文档总结当前代码中“双手模式”网络的实际实现与数据流，基于以下文件：
- 入口与类型分发：`src/network/model.py`
- 模型主体：`src/network_biman/bimanual_diffusion.py`
- 骨干网络与特征抽取：`src/network/backbones/backbones.py`
- 扩散与时间嵌入：`src/network/diffusion.py`, `src/network/mlp.py`
- 数据与打包：`src/utils/dataset_bimanual.py`
- 训练脚本：`src/train.py`
- 参考配置：`configs/network/train_bimanual.yaml`

---

### 1) 模型入口与类型
- `get_model(config)`：当 `config.type == 'bimanual_diff'` 时返回 `BimanualDiffusion(config)`。

---

### 2) 模型结构（BimanualDiffusion）
文件：`src/network_biman/bimanual_diffusion.py`

- 关键参数
  - `feature_dim`：骨干输出/条件通道维度（如 256）。
  - `use_keypoints`：是否使用左右关键点作为条件特征融合（默认 True）。
  - `joint_num`：手部关节数（不含 WRJ 6 自由度）；配置示例为 22。
  - `out_dim = 2 * (joint_num + 6)`：双手输出通道数（每只手 `J+6`，6 为 WRJRx, WRJRy, WRJRz, WRJTx, WRJTy, WRJTz）。配置示例：`2*(22+6)=56`。

- 子模块
  - 骨干：`self.backbone = get_backbone(config.backbone, feature_dim, ...)`
    - `sparseconv` → `MinkUNet14D(in=3, out=feature_dim, D=3)`（逐点特征）
    - `sparse_glob_conv` → `ResNet14D(in=3, out=feature_dim, D=3)`（偏全局表示）
  - 策略头（用于扩散预测残差/速度）：
    - `self.policy = MLPWrapper(channels=out_dim, feature_dim=feature_dim, hidden_layers_dim=[512,256], act='mish')`
    - `MLPWrapper` 将 `cond` 与时间嵌入拼接后送入 `MLP`。
  - 扩散：`self.diffusion = GaussianDiffusion1D(self.policy, config.diffusion)`
    - `scheduler_type: 'DDPMScheduler'`，`prediction_type ∈ {epsilon, v_prediction}`，`num_train_timesteps`, `num_inference_timesteps` 由配置给出。
  - 条件映射：`self.cond_project = nn.Identity()`（当前为恒等映射，未做额外投影/融合层）。

- 特征抽取与条件构建
  - `get_point_features(data)` → `get_feature(backbone_name, backbone, data)`：
    - 对 `sparseconv`：将 `ME.SparseTensor(feat, coors)` 经 `MinkUNet14D` 得到稀疏输出 `.F`（M,C），再用 `quantize2original` 还原到逐点 `(B,N,C)`。
    - 对 `sparse_glob_conv`：返回骨干输出（当前实现直接返回 `.F`，通常用于全局向量）。
  - `build_condition(data, feat)`：
    - 若 `feat` 为逐点 `(B,N,C)`：`global_feat = max_pool(feat, dim=1)` 得到 `(B,C)`。
    - 若启用 `use_keypoints` 且存在 `keypoint_left/right` 且具备逐点特征：
      - 通过最近邻从 `(point_clouds[B,N,3], point_feats[B,N,C])` 聚合关键点特征：`kp_left`, `kp_right` ∈ `(B,C)`。
      - 条件融合：`cond = global_feat + 0.5*(kp_left + kp_right)`。
    - 否则：`cond = global_feat`。
    - 最终：`cond = cond_project(cond)`（当前为 Identity）。

- 前向与采样
  - 训练 `forward(data)`：
    - 期望标签：`qpos_left/right` 各为 `(B, J+6)`（其中 6 为 WRJ 六自由度，要求在尾部）。
    - 拼接目标：`target = cat([qpos_left, qpos_right], dim=-1)` → `(B, 2*(J+6))`。
    - 条件：如上 `cond ∈ (B,C)`。
    - 损失：`loss_diff = GaussianDiffusion1D.calculate_loss(target, cond)`，返回 `{'loss_diffusion': loss_diff}`。
  - 采样 `sample(data, sample_num)`：
    - 重复条件到 `(B*sample_num, C)`，经扩散采样得到 `(B, sample_num, out_dim)`，再按 `split = J+6` 拆分为 `left`, `right`。

---

### 3) 扩散与时间嵌入
文件：`src/network/diffusion.py`

- `SinusoidalPosEmb(feature_dim)`：将标量时间步（归一化至 `[0,1]`）嵌入为 `(B, feature_dim)` 正余弦位置编码。
- `MLPWrapper(channels, feature_dim, ...)`：
  - 输入为 `x`（被噪声化的目标，形状 `(B, channels)`）、`t`（归一化时间）、`cond`（条件 `(B, feature_dim)`）。
  - 前向：`embedding(t) → emb`，再令 `x_cat = concat([x, cond + emb], -1)` 后经 `MLP`。
- `GaussianDiffusion1D(model, config)`：
  - 训练：
    - 采样 `t`，噪声 `noise`，`noised_x = scheduler.add_noise(x, noise, t)`。
    - `cond_now = cond_fn(noised_x, t/T, cond)`（当前为恒等）
    - `pred = model(noised_x, t/T, cond_now)`，`target = noise`（epsilon）或 `get_velocity(x,noise,t)`（v_prediction）。
    - 损失：`MSE(pred, target)`。
  - 采样：
    - 迭代时间步，使用 DDPM 的参数（支持 ODE/off-ODE 分支），返回 `x` 与 `log_prob`（如未启用则为 0）。

---

### 4) 骨干网络与特征抽取
文件：`src/network/backbones/backbones.py`

- `get_backbone(backbone_name, feature_dim, backbone_config)`：
  - `sparseconv` → `MinkUNet14D(in_channels=3, out_channels=feature_dim, D=3)`（推荐用于需要逐点特征的情形）。
  - `sparse_glob_conv` → `ResNet14D(in_channels=3, out_channels=feature_dim, D=3)`（偏全局表征）。
- `get_feature(backbone_name, backbone, data)`：
  - 构建 `ME.SparseTensor(feats, coordinates)`，经骨干前向得到 `.F`。
  - `sparseconv` 分支使用 `quantize2original` 将稀疏特征映射回 `(B,N,C)` 逐点特征；否则返回骨干输出（通常为全局/稀疏坐标系特征）。

---

### 5) 数据接口与 Collate（双手）
文件：`src/utils/dataset_bimanual.py`

- 键与形状（单样本）：
  - `point_clouds: (N,3)`、`coors: (N,3)/voxel_size → ME 坐标`、`feats: (N,3)=1`。
  - `keypoint_left/right: (3,)`。
  - `qpos_left/right: (J+6,)`，当前实现中 `J=22`（手指），尾部 6 维为 WRJ（Rx,Ry,Rz,Tx,Ty,Tz）。
- `minkowski_collate_fn_biman(list_data)`：
  - `ME.utils.sparse_collate` → `ME.utils.sparse_quantize` 得到 `coors, feats, original2quantize, quantize2original`。
  - 其余键（含 `point_clouds`, `keypoint_*`, `qpos_*`）按批次堆叠为张量，供骨干与条件模块使用。

---

### 6) 配置要点（示例：`configs/network/train_bimanual.yaml`）
- `model.type: bimanual_diff`
- `model.backbone: sparseconv`（或 `sparse_glob_conv`）
- `model.feature_dim: 256`
- `model.joint_num: 22`（手指关节数，不含 WRJ 6 自由度）
- `model.use_keypoints: true`
- `model.diffusion.scheduler_type: DDPMScheduler`
- `model.diffusion.scheduler.prediction_type: v_prediction`（或 `epsilon`）
- `model.diffusion.num_inference_timesteps: 50`
- 数据：`voxel_size: 0.005`，并在 `dataset_bimanual.py` 中读取 `pc_path` 与 `pairs_path`

---

### 7) 端到端数据流（训练）
1. Dataset 读入 `point_clouds`, 关键点与 `qpos_left/right`，经 `minkowski_collate_fn_biman` 得到稀疏输入与映射索引。
2. 骨干：`ME.SparseTensor(feats, coors)` → `backbone` → 稀疏输出 `.F` →（对 `sparseconv`）还原逐点 `(B,N,C)`。
3. 条件：`global_feat = max_pool(feat, dim=1)`；若启用关键点，使用最近邻从逐点特征聚合 `kp_left/right` 并与 `global_feat` 融合得到 `cond`。
4. 目标：将左/右手目标向量（`J+6`）拼接为 `target ∈ (B, 2*(J+6))`。
5. 扩散：按 DDPM 训练流程计算 `loss_diffusion`。
6. 返回：`loss` 与 `{'loss_diffusion': loss}` 供日志记录与优化。

---

### 8) 张量形状汇总（常见设置）
- `point_clouds`: `(B,N,3)`
- `coors`: `(M,4)`（ME: batch 索引 + 量化坐标）
- `feats`: `(M,3)`（常为全 1）
- `feature_dim`: `C`（如 256）
- `feat (sparseconv)`: `(B,N,C)`；`global_feat`: `(B,C)`
- `keypoint_left/right`: `(B,3)`
- `cond`: `(B,C)`
- `qpos_left/right`: `(B, J+6)`（`J=22` → `28`）
- `target`: `(B, 2*(J+6))`（`2*28=56`）

---

### 9) 重要注意事项
- `joint_num` 表示不含 WRJ 的手指关节数；数据集中 `qpos_*` 向量实际长度为 `J+6`，且 WRJ 6 维需位于尾部以与模型约定一致。
- 若使用 `sparse_glob_conv`（仅全局/非逐点特征），当前关键点分支需要逐点特征以做 NN 聚合；此时将仅使用 `global_feat` 作为条件（不聚合关键点特征）。
- `cond_project` 当前为 `Identity`，未来若需对 `global_feat` 与关键点特征作更复杂融合可在此扩展。
- 关键点与点云需处于同一坐标系；如不一致需在数据阶段对齐。

---

以上即为当前“双手模式”网络结构与数据流的实现要点（截至 2025-10-27）。

