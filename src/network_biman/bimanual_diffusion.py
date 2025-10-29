import torch
from torch import nn
from einops import rearrange, repeat
from typing import Dict, Tuple

from src.network.backbones.backbones import get_backbone, get_feature
from src.network.diffusion import MLPWrapper, GaussianDiffusion1D
from src.network.condition import ConditionalTransform


class BimanualDiffusion(nn.Module):
    """
    Bimanual pose diffusion model.
    - Inputs: point clouds (+ optional keypoints)
    - Outputs: concatenated left/right vectors of size 2*(J+6), where 6 corresponds to WRJRx, WRJRy, WRJRz, WRJTx, WRJTy, WRJTz
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.feature_dim = config.feature_dim
        self.use_keypoints = getattr(config, 'use_keypoints', True)
        self.joint_num = config.joint_num if hasattr(config, 'joint_num') else 28
        self.wrist_dim_per_hand = 6
        self.wrist_first = bool(getattr(config, 'wrist_first', False))
        # diffusion output dim
        if self.wrist_first:
            # only wrists: 2 * 6
            self.out_dim = 2 * self.wrist_dim_per_hand
            self.finger_dim = 2 * self.joint_num
        else:
            # full vector: 2 * (J + 6)
            self.out_dim = 2 * (self.joint_num + self.wrist_dim_per_hand)

        # backbone
        self.backbone = get_backbone(
            backbone_name=config.backbone,
            feature_dim=self.feature_dim,
            backbone_config=(getattr(config, 'backbone_parameters', {}) or {}).get(config.backbone, {}),
        )

        # policy/diffusion
        policy_params = dict(hidden_layers_dim=[512, 256], output_dim=self.out_dim, act='mish')
        self.policy = MLPWrapper(channels=self.out_dim, feature_dim=self.feature_dim, **policy_params)
        self.diffusion = GaussianDiffusion1D(self.policy, config.diffusion)

        # If wrist-first, add a joint head to regress fingers from (cond, wrists)
        if self.wrist_first:
            # Input = cond(feature_dim) + wrists(12)
            self.joint_head = ConditionalTransform(self.feature_dim + 2 * self.wrist_dim_per_hand, self.finger_dim)

        # optional extra MLP if we want to map concatenated cond
        self.cond_project = nn.Identity()

        # joint regression loss (for auxiliary comparisons if needed)
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')

    def get_point_features(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return point-wise features (B, N, C) if backbone supports it, otherwise a global (B, C).
        """
        feat = get_feature(
            backbone_name=self.config.backbone,
            backbone=self.backbone,
            data=data,
        )
        return feat

    @staticmethod
    def gather_kp_feature(point_clouds: torch.Tensor, point_feats: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        """
        Gather keypoint feature using nearest neighbor from point_clouds.
        Args:
            point_clouds: (B, N, 3)
            point_feats: (B, N, C)
            keypoints: (B, 3)
        Returns:
            kp_feat: (B, C)
        """
        # brute-force nearest neighbor (small batches); can be replaced by pytorch3d.knn_points
        with torch.no_grad():
            diffs = point_clouds - keypoints[:, None, :]
            dists = (diffs * diffs).sum(-1)  # (B, N)
            nn_idx = dists.argmin(dim=-1)  # (B,)
        kp_feat = point_feats[torch.arange(point_feats.shape[0], device=point_feats.device), nn_idx]
        return kp_feat

    def build_condition(self, data: Dict[str, torch.Tensor], feat: torch.Tensor) -> torch.Tensor:
        """
        Build conditioning vector of shape (B, C).
        Use global pooled feature; if use_keypoints, fuse with keypoint features gathered from per-point features.
        """
        if feat.dim() == 3:
            # point-wise -> global
            global_feat = feat.max(dim=1)[0]
        else:
            global_feat = feat

        if self.use_keypoints and 'keypoint_left' in data and 'keypoint_right' in data and feat.dim() == 3:
            kp_left = self.gather_kp_feature(data['point_clouds'], feat, data['keypoint_left'])
            kp_right = self.gather_kp_feature(data['point_clouds'], feat, data['keypoint_right'])
            cond = global_feat + 0.5 * (kp_left + kp_right)
        else:
            cond = global_feat
        return self.cond_project(cond)

    def forward(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Training forward: compute diffusion loss on concatenated target.
        Expected data:
            point_clouds: (B, N, 3)
            keypoint_left/right: (B, 3) if use_keypoints
            qpos_left/right: (B, J)
            wrj_left/right: (B, 6) optional; if not provided, include in qpos vector order externally
        """
        feat = self.get_point_features(data)
        cond = self.build_condition(data, feat)

        # targets: concatenate left/right
        # Expect caller packs WRJ(6) at tail of qpos_{side}
        gt_left = data['qpos_left']  # (B, J+6)
        gt_right = data['qpos_right']

        if self.wrist_first:
            # Split finger and wrists
            left_finger, left_wrist = gt_left[..., :-self.wrist_dim_per_hand], gt_left[..., -self.wrist_dim_per_hand:]
            right_finger, right_wrist = gt_right[..., :-self.wrist_dim_per_hand], gt_right[..., -self.wrist_dim_per_hand:]
            wrist_target = torch.cat([left_wrist, right_wrist], dim=-1)  # (B, 12)
            finger_target = torch.cat([left_finger, right_finger], dim=-1)  # (B, 2J)

            # diffusion only on wrists
            loss_diff = self.diffusion.calculate_loss(wrist_target, cond)

            # train joint head with teacher forcing (gt wrists) for stability
            joint_in = torch.cat([cond, wrist_target], dim=-1)
            finger_pred = self.joint_head(joint_in)
            loss_finger = self.smooth_l1(finger_pred, finger_target).mean()

            # weights
            weight_diff = getattr(getattr(self.config, 'weight', object()), 'diffusion', 1.0) if hasattr(self.config, 'weight') else 1.0
            weight_finger = getattr(getattr(self.config, 'weight', object()), 'finger', 1.0) if hasattr(self.config, 'weight') else 1.0
            loss_total = weight_diff * loss_diff + weight_finger * loss_finger

            return loss_total, dict(loss_diffusion=loss_diff, loss_finger=loss_finger, loss_total=loss_total)
        else:
            # full target: wrists + fingers
            target = torch.cat([gt_left, gt_right], dim=-1)
            loss_diff = self.diffusion.calculate_loss(target, cond)
            return loss_diff, dict(loss_diffusion=loss_diff)

    @torch.no_grad()
    def sample(self, data: Dict[str, torch.Tensor], sample_num: int = 1, **kwargs):
        B = data['point_clouds'].shape[0]
        feat = self.get_point_features(data)
        cond = self.build_condition(data, feat)
        cond = repeat(cond, 'b c -> (b k) c', k=sample_num)
        samples, log_prob = self.diffusion.sample(cond=cond)

        if self.wrist_first:
            # samples are wrists only: (B*k, 12)
            wrists = samples  # (B*k, 12)
            # finger prediction from sampled wrists
            joint_in = torch.cat([cond, wrists], dim=-1)
            finger_pred = self.joint_head(joint_in)  # (B*k, 2J)

            # reshape
            wrists = wrists.view(B, sample_num, 2 * self.wrist_dim_per_hand)
            finger_pred = finger_pred.view(B, sample_num, 2 * self.joint_num)
            log_prob = log_prob.view(B, sample_num)

            # split per hand then concat fingers + wrists to match (J+6)
            lw, rw = wrists[..., :self.wrist_dim_per_hand], wrists[..., self.wrist_dim_per_hand:]
            lf, rf = finger_pred[..., :self.joint_num], finger_pred[..., self.joint_num:]
            left = torch.cat([lf, lw], dim=-1)
            right = torch.cat([rf, rw], dim=-1)
            return left, right, log_prob
        else:
            # reshape
            samples = samples.view(B, sample_num, self.out_dim)
            log_prob = log_prob.view(B, sample_num)
            # split
            split = self.joint_num + self.wrist_dim_per_hand
            left = samples[..., :split]
            right = samples[..., split:split * 2]
            return left, right, log_prob


