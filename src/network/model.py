import torch


def get_model(config: dict):
    """
        get model by config
    """
    if config.type in ['graspness_isa', 'graspness_diffusion', 'graspness_cvae']:
        from src.network.graspness_sample import GraspnessSample
        return GraspnessSample(config)
    elif config.type in ['glob_diff']:
        from src.network.diffusion_sample import DiffusionSample
        return DiffusionSample(config)
    elif config.type in ['bimanual_diff']:
        from src.network_biman import BimanualDiffusion
        return BimanualDiffusion(config)
    raise NotImplementedError()