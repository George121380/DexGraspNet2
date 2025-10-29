import torch
import torch.nn as nn
from typing import Optional, Tuple, List


try:
    # PointNet++ ops (should be installed from third_party/Pointnet2_PyTorch)
    from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
except Exception as e:  # pragma: no cover - import error raised only when used
    raise ImportError(
        "Failed to import pointnet2_ops. Please install third_party/Pointnet2_PyTorch/pointnet2_ops_lib."
    ) from e


class PointNet2SSGEncoder(nn.Module):
    """
    PointNet++ SSG-based encoder that produces per-point features.

    Input:
      - points_xyz: (B, N, 3) only xyz coordinates

    Output:
      - features: (B, N, C) per-point features with channel size = out_channels
    """

    def __init__(self, out_channels: int, use_xyz: bool = True) -> None:
        super().__init__()
        self.use_xyz = bool(use_xyz)

        # Encoder (Set Abstraction) blocks
        self.SA_modules = nn.ModuleList()
        # No extra per-point features beyond xyz -> first MLP starts from 0 channels
        sa1_in_channels = 0
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[sa1_in_channels, 32, 32, 64],
                use_xyz=self.use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=self.use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=self.use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=self.use_xyz,
            )
        )

        # Decoder (Feature Propagation) blocks
        self.FP_modules = nn.ModuleList()
        # Skip from original input has 0 channels (no extra features)
        self.FP_modules.append(PointnetFPModule(mlp=[128 + 0, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))

        # Projection to desired output channels
        self.point_feature_channels = 128
        if out_channels != self.point_feature_channels:
            self.proj = nn.Conv1d(self.point_feature_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.proj = nn.Identity()

        self.out_channels = out_channels

    @staticmethod
    def _break_up_pc(pc: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Split a (B, N, 3 + C) pointcloud into xyz and features.
        Here we expect (B, N, 3) so features will be None.
        """
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].contiguous()
        if features.numel() == 0:
            features = None
        else:
            features = features.transpose(1, 2).contiguous()
        return xyz, features

    def forward(self, points_xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points_xyz: (B, N, 3)

        Returns:
            features: (B, N, C)
        """
        if points_xyz.dim() != 3 or points_xyz.shape[-1] < 3:
            raise ValueError("points_xyz must have shape (B, N, 3)")

        xyz, features = self._break_up_pc(points_xyz)

        l_xyz: List[torch.Tensor] = [xyz]
        l_features: List[Optional[torch.Tensor]] = [features]

        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        # Feature propagation from deep to shallow
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        point_level_features = l_features[0]  # (B, 128, N)
        point_level_features = self.proj(point_level_features)  # (B, out_channels, N)
        return point_level_features.transpose(1, 2).contiguous()  # (B, N, C)


__all__ = ["PointNet2SSGEncoder"]



