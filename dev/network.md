### 抓取位姿预测网络（DexGraspNet2）设计综述

本文档梳理本代码库中抓取位姿预测网络的整体设计、训练与推理流程、数据接口与张量形状、各模块职责（Backbone/Diffusion/Graspness/CVAE），以及关键超参与依赖。对应源码主要位于 `src/` 目录。

---

### 总体架构与训练入口

- **训练入口**: `src/train.py`
  - 加载配置与命令行参数、设置随机种子与日志器、创建数据集与 DataLoader、构建模型与优化器/学习率调度、循环训练与验证、保存 checkpoint。
  - 关键调用链：
    - 配置: `src/utils/config.py` 的 `load_config`（YAML + CLI 覆盖，返回可点式访问的 `DotDict`）
    - 数据: `src/utils/dataset.py` 的 `GraspNetDataset`、`minkowski_collate_fn` 与简单的 `Loader` 包装器（无限迭代）
    - 模型: `src/network/model.py` 的 `get_model` 按 `config.model.type` 实例化
    - 记录: `src/utils/logger.py`（W&B + 本地目录 `experiments/<exp_name>`）

训练主循环（要点）：

```text
for it in [cur_iter, max_iter):
  data = train_loader.get(); data = {k: v.to(device)}
  loss, metrics = model(data)
  loss.backward(); clip_grad_norm_; optimizer.step(); scheduler.step()
  if it % log_every: logger.log(metrics, 'train', it)
  if (it+1) % save_every: logger.save(state_dict, it+1)
  if it % val_every: 评估各 val split，聚合均值并记录
```

---

### 配置流与命令行映射

- `load_config(yaml, arg_mapping, args)`：将 `yaml` 的层级键与 CLI 覆盖合并（优先级：CLI > 默认 > YAML），并返回 `DotDict`。
- `train.py` 提供一组参数映射（示例）：
  - `--type` → `model/type`，`--backbone` → `model/backbone`
  - `--dist_joint` → `model/dist_joint`
  - `--iter` → `max_iter`，`--camera` → `camera`
  - `--diff_pred` → `model/diffusion/scheduler/prediction_type`
- 常用配置片段：
  - `model.*`: 模型类型、骨干、特征维、缩放系数、loss 权重、diffusion 配置等
  - `data.*`: 相机、点数、体素大小 `voxel_size`、采样策略等

---

### 数据接口与张量形状

数据由 `GraspNetDataset` 生成，并通过 `minkowski_collate_fn` 组合为稀疏张量输入骨干网络（MinkowskiEngine）。关键键与形状（训练模式）：

- `point_clouds`: `(B, N, 3)`
- 稀疏张量相关：
  - `coors`: `(∑Mᵇ, 4)`，ME 需要的 [batch_id, x, y, z]
  - `feats`: `(∑Mᵇ, 3)`，通常为全 1/坐标特征
  - `original2quantize`, `quantize2original`: 稀疏量化索引映射
- 标注与监督：
  - `seg`: `(B, N)`，语义/实例掩码（0 为背景）
  - `objectness`: `(B, N)`，是否为物体点（二分类标签）
  - `graspness`: `(B, N)`，抓取可行性（对数尺度）
  - `rot`: `(B, K, 3, 3)`，候选抓取姿态旋转
  - `trans`: `(B, K, 3)`，候选抓取平移（相机帧）
  - `qpos`: `(B, K, J)` 或 `(B, K, 1)`（机械手或夹爪关节）
  - `centers`: `(B, K)`，每个候选抓取对应的点云索引
  - `has_graspness`: `(B, 1)`，指示是否存在有效 `graspness`

评估模式（`is_eval=True`）下，数据以场景/视角为主，返回 `scene`, `view`, `point_clouds`, 稀疏输入与 `seg`/`edge`（若有）。

---

### Backbone 与特征提取（MinkowskiEngine）

入口：`src/network/backbones/backbones.py`

- `get_backbone(backbone_name, feature_dim, backbone_config)`：
  - `"sparseconv"` → `MinkUNet14D(in_channels=3, out_channels=feature_dim)`（U-Net 结构）
  - `"sparse_glob_conv"` → `ResNet14D(in_channels=3, out_channels=feature_dim)`（全局池化 ResNet）
- `get_feature(backbone_name, backbone, data)`：
  - 构造 `ME.SparseTensor`(`feats`, `coors`)，前向得到输出 `mink_output.F`
  - `sparseconv`: 通过 `quantize2original` 映射回 `(B, N, C)` 的逐点特征
  - `sparse_glob_conv`: 返回 `mink_output`（为全局特征；下游使用需注意形状适配）

要点：
- 稀疏体素大小由 `config.data.voxel_size` 控制（训练入口会注入到 `config.model.voxel_size`）。
- `minkowski_collate_fn` 完成跨 batch 的稀疏拼接与量化，并保留原始-量化索引以便还原点级特征。

---

### 模型装配与类型

入口：`src/network/model.py`

- `get_model(config.model)`：
  - `type ∈ {graspness_isa, graspness_diffusion, graspness_cvae}` → `GraspnessSample`
  - `type ∈ {glob_diff}` → `DiffusionSample`

#### GraspnessSample（点级抓取质量与位姿）

文件：`src/network/graspness_sample.py`

- 组成：
  - Backbone 提取点级特征 `(B, N, C)`
  - 头部 `graspable: Linear(C → 3)`，输出：`objectness(logits:2)` 与 `graspness(1)`
  - 位姿/关节估计分三类：
    1) `graspness_diffusion`: 
       - 时间条件 MLP `policy = MLPWrapper(channels=J*dist_joint + 3 + rot_dim, feature_dim=C)`
       - `GaussianDiffusion1D(policy, config.diffusion)`（`DDPMScheduler`，`prediction_type ∈ {epsilon, v_prediction}`）
       - 旋转表示 `rot_type ∈ {svd(9), sixd(6), quat(4), aa(3), euler(3)}`
       - 若 `dist_joint=False`，额外 `joint_mlp`：`ConditionalTransform(C + 9 + 3 → J)`
    2) `graspness_isa`（直接回归）：
       - `joint_mlp: ConditionalTransform(C → J + 4 + 3)` 输出 `quat(4)` 与 `euc(3+J)`
    3) `graspness_cvae`：
       - 采用 `GraspCVAE` 条件 VAE（详见后文），显式利用机器人运动学和几何约束

- 前向训练：
  1) 点级特征 `feature = get_feature(data)`
  2) 分类与打分：`objectness, graspness = graspable(feature)`
  3) 选择中心点与特征：根据 `centers` 获取与抓取候选对齐的点特征
  4) 构造监督：
     - 平移采用与体素中心的差值并按 `trans_scale` 缩放
     - `dist_joint=True` 时，将关节拼接进 `euc` 并按 `joint_scale` 缩放
     - 旋转根据配置转为目标表示
  5) 计算损失（见“损失函数”小节）并加权汇总

- 采样推理：
  1) 基于 `objectness` 与 `graspness` 从点云中筛选 `k` 个种子点（可选 `cate`/`near` 策略）
  2) 在每个种子点的特征上调用 `sample_grasp`：
     - diffusion 采样得到 `(rot_rep, euc)`，再还原旋转矩阵与平移/关节
     - ISA 直接回归；CVAE 通过条件解码
  3) 组合得分：`score = log_prob + graspness * graspness_scale`
  4) 返回 `(rot, trans, joints, score, obj_indices[, graspness_parts, log_prob, seed_points])`

#### DiffusionSample（全局分布示例）

文件：`src/network/diffusion_sample.py`

- 与 `graspness_diffusion` 的核心相同，但针对批内重复采样（`sample_num`）和不含点级分类头。
- 输出 `(rot, trans, joints, log_prob, obj_indices)` 或附带点坐标/占位返回。

---

### Diffusion 细节

文件：`src/network/diffusion.py`

- `MLPWrapper`：将时间步 `t` 经过 `SinusoidalPosEmb` 后与条件特征 `cond` 相加再与输入拼接，送入 `MLP`。
- `GaussianDiffusion1D`：
  - 训练：随机采样时间步，对输入 `x` 加噪，预测 `epsilon` 或 `v`，MSE 损失
  - 采样：按 `num_inference_timesteps` 迭代，支持 ODE 模式与对数似然近似（`log_prob_type ∈ {accurate_cont, estimate, None}`）
  - `channels = J*dist_joint + 3 + rot_dim`

---

### CVAE 细节（GraspCVAE）

文件：`src/network/cvae.py`

- 条件 VAE：`Encoder([cond_size, ...]→latent) / Decoder([cond_size,...])`
- 条件输入：当前手部表面点云的 PointNet 全局特征（`hand_encoder`），条件为物体/上下文特征（本实现中直接以手特征作为 c，物体 encoder 注释掉）
- 运动学与几何：
  - 借助 `RobotModel` 前向运动学得到各连杆位姿；将预测的 `(trans, rot(轴角), joints)` 生成手部表面点云
  - 结合 Chamfer 距离、KLD、关节/位姿监督、穿透惩罚与接触图（cmap）一致性项进行优化
- 推理：从标准正态采样 `z`，解码得到 `(trans, aa, joints)`，旋转由轴角转矩阵

---

### 损失函数（训练前向）

以 `GraspnessSample` 为例：

- 分类与打分：
  - `loss_objectness = CrossEntropy(objectness_logits, gt_objectness)`（点级平均）
  - `loss_graspness = SmoothL1(graspness*gt_objectness, gt_graspness*gt_objectness)`（仅物体点，归一化平均；无标注样本屏蔽）
- 位姿/关节：
  - `graspness_diffusion`: `loss_diffusion = MSE(pred, target)`（`epsilon` 或 `v`）
  - `graspness_isa`: `loss_euc`（平移/关节 L1），`loss_quat`（SO(3) 相对角）
  - `graspness_cvae`: 复合 `loss_trans/rot/qpos/recon/KLD/pen/cmap` 等
  - 若 `dist_joint=False`：`loss_joint = SmoothL1(joint_mlp(...), gt_joints*joint_scale)`
- 汇总：
  - `loss = w.objectness*loss_objectness + w.graspness*loss_graspness + ...`（按 `config.weight.*` 加权）

---

### 采样与推理流程

1) 提取点级特征并计算 `objectness/graspness`
2) 选点策略：
   - `cate=True`：按实例/类别分配 `k` 并在各子集内基于 graspness 选点（含 5% 阈值）
   - `near=True`：球查询局部邻域，提升邻域内高 graspness 候选的覆盖
3) 基于所选特征调用 `sample_grasp`：
   - diffusion 迭代采样（可得 `log_prob`），或 ISA/CVAE 直接生成
4) 还原尺度：平移按 `voxel_center(seed)` 与 `trans_scale` 反缩放，关节按 `joint_scale` 反缩放
5) 打分与返回结果，用于后续评估或保存

评估示例参见 `src/eval/eval_gripper.py`：从 checkpoint 加载模型，遍历测试集，`model.sample(..., k=1024, ...)` 后将 `(score, width, depth, rot, trans, ...)` 保存为 `.npy`。

---

### 关键参数与缩放

- `voxel_size`：稀疏体素边长，影响稀疏张量坐标与体素中心计算
- `trans_scale` / `joint_scale`：训练时对平移与关节进行数值缩放，推理需相应反缩放
- `rot_type`：旋转表示影响 `rot_dim` 与监督/还原逻辑
- `num_train_timesteps` / `num_inference_timesteps`：diffusion 训练与采样步数
- `log_prob_type` / `ode`：采样时对数似然估计与确定性/随机路径

---

### 依赖与注意事项

- 稀疏卷积：MinkowskiEngine（需与 PyTorch/CUDA 版本匹配）
- 几何/渲染：PyTorch3D（KNN、Chamfer、旋转变换等）
- Diffusion：HuggingFace `diffusers` 调度器
- 记录：Weights & Biases（可通过 `exp_name='temp'` 关闭自动初始化）
- 一些模块（如 `conv.py`）要求 `open3d`，但默认 U-Net 路径不依赖它

---

### 主要文件索引

- 训练与评估
  - `src/train.py`：训练主程序
  - `src/eval/eval_gripper.py`：评估/保存抓取
- 配置与工具
  - `src/utils/config.py`：配置装载与 `DotDict`
  - `src/utils/logger.py`：日志与 checkpoint
  - `src/utils/dataset.py`：数据集与 ME collate
- 模型与模块
  - `src/network/model.py`：模型入口选择器
  - `src/network/graspness_sample.py`：点级抓取质量与位姿网络
  - `src/network/diffusion_sample.py`：全局 diffusion 示例
  - `src/network/backbones/*`：MinkUNet/ResNet 稀疏骨干
  - `src/network/diffusion.py`：1D 条件 diffusion 核心
  - `src/network/cvae.py`：条件 VAE 与运动学/几何损失

---

### 附：关键 API 的输入/输出（摘要）

- `GraspnessSample.forward(data)` → `loss: scalar, metrics: dict(B,)`
- `GraspnessSample.sample(data, k, ...)` → `(rot[B,K,3,3], trans[B,K,3], joints[B,K,J], score[B,K], obj_indices[B,K], ...)`
- `DiffusionSample.sample(data, sample_num)` → 同上但不经点筛选
- `GaussianDiffusion1D.calculate_loss(x[B,C], cond[B,F])` → `loss`
- `GaussianDiffusion1D.sample(cond[B,F])` → `(x[B,C], log_prob[B])`
- `get_feature(data)`（sparseconv）→ `(B, N, C)`

---

如需进一步拓展：
- 在 `graspness_diffusion` 中接入更多旋转参数化或混合表示；
- 对 `sparse_glob_conv` 的全局特征在点级任务中的形状/广播策略进行一致性检查；
- 在 CVAE 中引入显式物体编码器（现示例中注释掉），形成手-物双塔条件建模。



