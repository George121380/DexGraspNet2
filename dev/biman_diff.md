### 双手抓取扩展计划（在 DexGraspNet2 基础上最小改动）

目标：在保留现有训练/评估与网络骨架的前提下，扩展输入以包含两个人工/检测到的关键点（keypoints），作为条件特征参与预测；输出从单手抓取姿态扩展为左右两手的姿态（关节+末端位姿）。训练数据来自另一套包含点云、特征点与双手姿势的标注；推理输出需整理为给定样例的可视化格式。

---

### 一、样例数据与输出格式分析

- 点云：`obj_points.npy` → `np.ndarray(float32, (N,3))`
- 抓取对与关键点：`grasp_pairs.npy` → `dict(object_name: str, pairs: dict[int, {num:int, left:{qpos:dict, center_point:float[3]}, right:{qpos:dict, center_point:float[3]}}])`
- 可视化输出样例：`100015_merged_1031.npy` → `np.ndarray(object, (M,))`，每个元素为：
  - `dict` with keys: `qpos_left:dict[str,float]`, `qpos_right:dict[str,float]`, `scale:float`, `index_left:int`, `index_right:int`, `E_pen_left:float`, `E_pen_right:float`

结论：训练标签需要包含每只手的 `qpos`（28 个键，与 `qpos_left/right` 的键名一致），以及与每只手关联的关键点（本任务用 `center_point` 作为 keypoint 输入）。推理输出需按上述字典打包成 `np.ndarray(object, (K,))` 存盘。

---

### 二、总体设计（最小改动）

- 复用：
  - 稀疏骨干（`MinkUNet14D` 或 `ResNet14D`）、cfg/Logger/训练循环与优化器、diffusion 框架（`GaussianDiffusion1D`）与时间嵌入
  - 训练脚本 `src/train.py`，仅配置与数据集路径变动
- 扩展点：
  - 数据集：新增 `BimanualDataset` 以读取新数据（点云 + 两关键点 + 双手 qpos），并生成 Minkowski 所需键；兼容 collate
  - 条件：在 `DiffusionSample`/`GraspnessSample` 基础思路上，新增关键点条件编码与融合（concat 到 cond）
  - 输出：预测左右手（left/right）姿势。实现上建议“两头共用一个骨干 + 条件共享/分支”
  - 保存：新增推理/导出适配器，将结果组装为与 `100015_merged_1031.npy` 同构的 object 数组
  - 配置开关：新增 `model.use_keypoints`（bool），控制是否启用关键点作为条件输入，便于在同一代码下训练两套模型（仅点云 vs 点云+关键点）。

---

### 三、数据集与数据流

新增 `src/utils/dataset_bimanual.py`（规划）：
- 输入路径（示例测试用）：
  - 点云：`obj_points.npy`
  - 抓取对：`grasp_pairs.npy`（提供左右手 qpos 与 keypoints）
- 输出（单样本）键：
  - `point_clouds: (N,3)`、`coors: (N,3)/voxel_size`、`feats: (N,3)=1`
  - `keypoint_left: (3,)`、`keypoint_right: (3,)`
  - `qpos_left: (J,)`、`qpos_right: (J,)`（按统一顺序排列，J=28）
  - 可选：`scale: float`、`index_left/right: int`（若存在，则传递给导出阶段）
  - 训练模式下，打包为 batch 后通过 `minkowski_collate_fn` 增补 `original2quantize/quantize2original` 等

qpos 键名顺序：以样例 `qpos_left/right` 的 28 项顺序固定，构建 `joint_name_list` 映射为定序向量；训练与推理中保持一致，输出再还原为字典。

---

### 四、条件特征与融合策略

- 场景特征：
  - 复用 Backbone 获取 `(B,N,C)` 或全局 `C`（若使用 `sparseconv` 则进行 `max/mean` 池化至 `(B,C)` 作为全局特征）
- 关键点特征抽取（优先使用逐点特征）：
  - 若关键点本身在物体点集中：直接取该点对应的逐点特征 `(B,C)` 作为条件特征。
  - 若关键点不在物体点集中：在物体点集中查找与关键点最近的点（KNN=1），使用该最近邻点的逐点特征 `(B,C)` 作为条件特征。
  - 实现要点：
    - 浮点误差处理：以阈值 `eps` 判断“在集合中”（建议 `eps = min(1e-5, 0.25*voxel_size)`），或将关键点与点云都量化到 Minkowski 的体素坐标系后再比较是否同一体素。
    - 最近邻检索：可使用 `pytorch3d.knn_points`（更高效）或 `torch.cdist` 寻找最近索引，再对 `(B,N,C)` 特征做 `gather`。
    - 如选择 `sparse_glob_conv`（仅全局特征）无法得到逐点特征，建议改用 `sparseconv`
- 条件融合：
  - 默认：`cond = global_feature + kp_feat`（或 concat 后线性映射至 `feature_dim`）。
  - 送入 `GaussianDiffusion1D` 的 `cond_fn` 路径，与时间嵌入对齐（沿用 `MLPWrapper`：`cond + t_emb`）。
  - 当 `model.use_keypoints = false`：跳过关键点分支，`cond = global_feature`；其余流程保持不变。

---

### 五、模型与输出头（BimanualDiffusion）

在 `src/network/diffusion_sample.py` 的思路上新增 `BimanualDiffusion`：
- 通道设计：对于每只手输出 `J` 个关节 + 6 个末端自由度（WRJTx/WRJTy/WRJTz/WRJRx/WRJRy/WRJRz），共 `J+6`。双手合计 `2*(J+6)` 通道。
- policy 头：
  - `policy_left = MLPWrapper(channels=J+6, feature_dim=feature_dim)`
  - `policy_right = MLPWrapper(channels=J+6, feature_dim=feature_dim)`
  - 也可用单一 `policy` 输出 `2*(J+6)` 后拆分（简单、参数共享更强）
- diffusion：
  - 复用 `GaussianDiffusion1D`，目标 `x` 为拼接的 `[left, right]` 向量
  - 监督目标：由 GT `qpos_left/right` 字典按统一顺序映射到向量，并将 6D 末端位姿（WRJ*）按顺序拼在末尾
- 损失：
  - diffusion MSE（`epsilon`/`v_prediction`）
  - 可选正则：双手相对约束（如两 keypoints 与末端位姿距离/方向先验）后续增量实现

---

### 六、训练与验证

- 配置（新增样例）：
  - `model.type: 'bimanual_diff'`
  - `model.backbone: 'sparseconv'`、`model.feature_dim: 256/512`
  - `model.diffusion`: 复用原配置（`num_train_timesteps`, `prediction_type`, `num_inference_timesteps` 等）
  - `model.use_keypoints`: true/false（开关是否使用关键点作为条件）
  - `data.yaml`: 指向新的数据路径与 batch/voxel_size 等
- 训练脚本：沿用 `src/train.py`
  - `get_model` 中新增分支 `bimanual_diff → BimanualDiffusion`
  - logger/ckpt/验证流程保持不变；验证时统计左右手 diffusion loss 的均值

两组模型训练建议：
- 仅点云（baseline）：`model.use_keypoints=false`，`exp_name=bimanual_pc_only_*`
- 点云+关键点：`model.use_keypoints=true`，`exp_name=bimanual_pc_kp_*`
其余配置尽量一致，便于公平对比。

---

### 七、推理与结果导出（适配可视化格式）

新增导出工具 `src/eval/export_bimanual.py`（规划）：
- 输入：批量点云与关键点（与训练相同的数据键），模型 ckpt
- 推理：输出 `pred_left (J+6)` 与 `pred_right (J+6)`
- 还原为可视化格式：
  - 依据 `joint_name_list` 将 `J` 维关节向量映射回字典
  - 将最后 6 维分别映射到 `WRJRx/WRJRy/WRJRz/WRJTx/WRJTy/WRJTz`（注意顺序一致）
  - 组装条目：
    ```
    dict(
      qpos_left={name: float for name in joint_name_list+WRJ*},
      qpos_right={...},
      scale=float(1.0),
      index_left=int(0),
      index_right=int(0),
      E_pen_left=float(0.0),
      E_pen_right=float(0.0),
    )
    ```
  - 聚合为 `np.ndarray(object, (K,))` 并 `np.save` 到用户指定路径

---

### 八、实现步骤（里程碑）

1) 数据集与映射
   - 实现 `BimanualDataset`：加载 `obj_points.npy` 与 `grasp_pairs.npy`，构造 batch 键
   - 补齐 Minkowski collate，保证与现有 `minkowski_collate_fn` 兼容
   - 统一 `joint_name_list` 顺序映射/逆映射工具
2) 模型扩展
   - 新增 `BimanualDiffusion`（或在 `DiffusionSample` 中通过 `type` 分支）
   - 条件编码 MLP 与融合；policy 输出通道翻倍
3) 训练接入
   - `get_model` 增加分支；配置新增 `bimanual_diff` 用例
   - 训练数据路径与 batch 参数调试
4) 推理与导出
   - 新增导出脚本，将预测转为 `100015_merged_1031.npy` 同构
   - 用提供的测试数据跑通 end-to-end

---

### 九、测试用最小示例（基于你提供的数据）

输入：
- 点云：`/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/data/preprocess_data/pot_data/train/100015/obj_points.npy`
- 抓取与关键点：`/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/data/preprocess_data/pot_data/train/100015/grasp_pairs.npy`

流程：
- 读取点云→稀疏张量；从 `pairs[k]` 中取一个样本的 `left/right.center_point` 作为 `(keypoint_left/right)` 与 `qpos_left/right` 作为监督
- 前向得到 `pred_left/right`；训练完成后推理 K 组结果
- 导出至 `.../bimanual-pose-data/<scene>/<scene>_pred.npy`（结构对齐 `100015_merged_1031.npy`）

---

### 十、风险与缓解

- 关键点坐标系不一致：需确认与点云同一坐标系；如有变换，统一在数据集阶段处理
- 关节顺序对齐：始终通过固定 `joint_name_list` 与映射函数进行矢量↔字典的转换
- 输出 6D 末端位姿的顺序：严格沿用 `WRJRx/WRJRy/WRJRz/WRJTx/WRJTy/WRJTz`，并在导出前断言
- ME/torch3d 版本兼容性：与原项目一致的 Python/torch/cu 版本
- 训练稳定性：沿用现有 grad clip、cosine 调度、权重配置；必要时增大学习率 warmup 或减小 `channels`

---

### 十一、配置草案（片段）

```yaml
model:
  type: bimanual_diff
  backbone: sparseconv
  feature_dim: 256
  use_keypoints: true  # 开关：true 使用 keypoints 作为条件；false 仅使用点云
  diffusion:
    scheduler_type: DDPMScheduler
    scheduler:
      num_train_timesteps: 1000
      prediction_type: v_prediction
    num_inference_timesteps: 50
  weight:
    diffusion: 1.0

data:
  voxel_size: 0.005
  batch_size: 4
  num_points: 8192
  # 自定义 bimanual 数据路径在 dataset_bimanual.py 中读取
```

---

结语：以上方案在不改动训练主循环与基础模块的前提下，最小代价扩展了输入条件（双关键点）与输出（双手姿态），并提供了与现有可视化工具兼容的结果导出路径。


