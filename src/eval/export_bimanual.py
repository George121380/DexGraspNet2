import os
import sys
import numpy as np
import torch
from argparse import ArgumentParser
from tqdm import tqdm

cur = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(cur)
sys.path.append(os.path.realpath('.'))

from src.utils.config import load_config
from src.utils.dataset_bimanual import BimanualDataset, minkowski_collate_fn_biman, JOINT_ORDER
from src.network.model import get_model
from torch.utils.data import DataLoader


def vector_to_qpos_dict(vec: np.ndarray) -> dict:
    return {name: float(val) for name, val in zip(JOINT_ORDER, vec.tolist())}


def main():
    ap = ArgumentParser()
    ap.add_argument('--yaml', type=str, required=True)
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--out_root', type=str, default=None)
    ap.add_argument('--scene', type=str, default='100015')
    ap.add_argument('--samples', type=int, default=1)
    ap.add_argument('--ref_scale_file', type=str, default=None, help='Optional npy file to mimic per-sample hand scale')
    args = ap.parse_args()

    config = load_config(args.yaml)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = BimanualDataset(config, split='eval', is_train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=minkowski_collate_fn_biman)

    model = get_model(config.model)
    model.to(device)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()

    # default output directory: experiments/<exp_name>
    out_root = args.out_root or os.path.join('experiments', config.exp_name)
    os.makedirs(out_root, exist_ok=True)
    out_path = os.path.join(out_root, f"{args.scene}.npy")

    # optional reference scales
    ref_scales = None
    if args.ref_scale_file is not None and os.path.exists(args.ref_scale_file):
        try:
            ref_arr = np.load(args.ref_scale_file, allow_pickle=True)
            ref_scales = [float(x.get('scale', 1.0)) for x in ref_arr]
        except Exception:
            ref_scales = None

    results = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
            left, right, logp = model.sample(data, sample_num=args.samples)
            # use the best sample (argmax logp) per batch=1
            idx = logp[0].argmax().item()
            left_vec = left[0, idx].cpu().numpy()
            right_vec = right[0, idx].cpu().numpy()
            qpos_left = vector_to_qpos_dict(left_vec)
            qpos_right = vector_to_qpos_dict(right_vec)
            scale_val = 1.0
            if ref_scales is not None and i < len(ref_scales):
                scale_val = ref_scales[i]
            entry = dict(
                qpos_left=qpos_left,
                qpos_right=qpos_right,
                scale=float(scale_val),
                index_left=int(0),
                index_right=int(0),
                E_pen_left=float(0.0),
                E_pen_right=float(0.0),
            )
            results.append(entry)

    np.save(out_path, np.array(results, dtype=object))
    print('Saved to', out_path)


if __name__ == '__main__':
    main()


