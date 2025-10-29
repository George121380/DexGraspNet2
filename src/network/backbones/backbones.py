import torch
import torch.nn as nn


def get_backbone(
    backbone_name: str, 
    feature_dim: int, 
    backbone_config: dict
):
    """
    Get backbone model from config
    
    Args: 
    - backbone: str, backbone name
    - feature_dim: int, feature dimension
    - backbone_config: dict, backbone config
    
    Returns:
    - backbone: nn.Module, backbone model
    """
    if backbone_name == "sparseconv":
        # Lazy import to avoid hard dependency when not used
        from .conv_unet import MinkUNet14D
        return MinkUNet14D(
            in_channels=3, out_channels=feature_dim, D=3)
    elif backbone_name == "sparse_glob_conv":
        # Lazy import to avoid hard dependency when not used
        from .conv import ResNet14D
        return ResNet14D(
            in_channels=3, out_channels=feature_dim, D=3)
    elif backbone_name == "pointnet2":
        # Lazy import to avoid hard dependency when not used
        from .pointnet2_encoder import PointNet2SSGEncoder
        use_xyz = bool(backbone_config.get('use_xyz', True)) if isinstance(backbone_config, dict) else True
        return PointNet2SSGEncoder(out_channels=feature_dim, use_xyz=use_xyz)
    else:
        raise ValueError(f"Backbone {backbone_name} not supported")

def get_feature(
    backbone_name: str, 
    backbone: nn.Module, 
    data: dict
):
    """
    Get feature from backbone
    
    Args:
    - backbone_name: str, backbone name
    - backbone: nn.Module, backbone model
    - data: dict, input data, format: {
        'point_clouds': torch.Tensor[B, N, 3, torch.float32], point clouds
        'coors': torch.Tensor[M, 4, torch.int32], batch id and coordinates
        'feats': torch.Tensor[M, 3, torch.float32], features, ones
        'original2quantize': torch.Tensor[M, torch.int64], original2quantize
        'quantize2original': torch.Tensor[B * N, torch.int64], quantize2original
    }
    
    Returns:
    - feature: torch.Tensor[B, N, C], feature
    """
    pc = data['point_clouds']
    batch_size, point_num, _ = pc.shape
    if backbone_name in ['sparseconv', 'sparse_glob_conv']:
        import MinkowskiEngine as ME  # local import to avoid hard dependency when pointnet2 is used
        coor = data['coors']
        feat = data['feats']
        mink_input = ME.SparseTensor(feat, coordinates=coor)
        mink_output = backbone(mink_input).F
        if backbone_name == 'sparseconv':
            feature = mink_output[data['quantize2original']].view(batch_size, point_num, -1)
        else:
            feature = mink_output
    elif backbone_name == 'pointnet2':
        # Directly consume dense point cloud and return (B, N, C)
        feature = backbone(pc)
    else:
        raise ValueError(f"Backbone {backbone_name} not supported")
    return feature
