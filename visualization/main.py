import os
import sys
sys.path.append(os.path.realpath('.'))

import cProfile
try:
    import memory_profiler
    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False

import argparse
import multiprocessing
import numpy as np
import torch
from tqdm import tqdm
import random
import transforms3d
import math
import shutil
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.initializations import initialize_dual_hand, initialize_dual_hand_at_targets
from utils.bimanual_energy import calculate_energy, cal_energy, BimanualEnergyComputer
from utils.bimanual_optimizer import MALAOptimizer
from utils.common import robust_compute_rotation_matrix_from_ortho6d
from torch.multiprocessing import set_start_method
import plotly.graph_objects as go
from utils.common import Logger
from utils.config import ExperimentConfig, create_config_from_args
from utils.visualization_recorder import FrameRecorder
from utils.metrics_recorder import SimpleMetricsRecorder
from utils.interactive_select import select_two_points_interactive, select_two_points_interactive_from_disk
from utils.bimanual_handler import (
    BimanualPair, hand_pose_to_dict, save_grasp_results, EnergyTerms
)
from utils.common import setup_device, set_random_seeds, ensure_directory


class GraspExperiment:
    """
    Main experiment class for bimanual grasp generation.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = None
        self.bimanual_pair = None
        self.object_model = None
        self.optimizer = None
        self.energy_computer = None
        self.logger = None
        
        # Profiling
        self.profiler = cProfile.Profile()
        
        # State tracking
        self.left_hand_pose_st = None
        self.right_hand_pose_st = None
        
    def setup_environment(self):
        """Setup device, random seeds, and environment variables."""
        self.device = setup_device(self.config.gpu)
        set_random_seeds(self.config.seed)
        np.seterr(all='raise')
        
        print(f"Using device: {self.device}")
        
    def setup_models(self):
        """Initialize hand and object models."""
        print("Setting up models...")
        
        # Create right hand model
        right_hand_model = HandModel(
            mjcf_path=self.config.paths.right_hand_mjcf,
            mesh_path=self.config.paths.mesh_path,
            contact_points_path=self.config.paths.right_contact_points,
            penetration_points_path=self.config.paths.penetration_points,
            device=self.device,
            n_surface_points=self.config.model.n_surface_points,
            handedness='right_hand'
        )
        
        # Create object model
        self.object_model = ObjectModel(
            data_root_path=self.config.paths.data_root_path,
            batch_size_each=self.config.model.batch_size,
            num_samples=self.config.model.num_samples,
            device=self.device,
            size=self.config.model.size
        )
        self.object_model.initialize(self.config.object_code_list)
        
        # Initialize dual hands (support target-based init & interactive selection)
        if getattr(self.config.initialization, 'interact', False):
            results_path = self.config.paths.get_experiment_results_path(self.config.name)
            try:
                A_scaled, B_scaled, preview = select_two_points_interactive(self.object_model, self.config.initialization, results_path)
            except Exception:
                # fallback without ObjectModel (e.g., missing deps in env)
                A_scaled, B_scaled, preview = select_two_points_interactive_from_disk(self.config.paths, self.config.object_code_list[0], self.config.initialization, results_path)
            # Inject into initialization config and enable target-based init
            self.config.initialization.left_target = f"{A_scaled[0]} {A_scaled[1]} {A_scaled[2]}"
            self.config.initialization.right_target = f"{B_scaled[0]} {B_scaled[1]} {B_scaled[2]}"
            self.config.initialization.init_at_targets = True
            # Default symmetry for interact mode: mirror + pi offset (keeps fingertip symmetry)
            if not getattr(self.config.initialization, 'twist_mirror', False):
                self.config.initialization.twist_mirror = True
            if float(getattr(self.config.initialization, 'right_twist_offset', 0.0)) == 0.0:
                # set to pi if not provided
                self.config.initialization.right_twist_offset = math.pi

        if getattr(self.config.initialization, 'init_at_targets', False):
            left_hand_model, right_hand_model = initialize_dual_hand_at_targets(
                right_hand_model, self.object_model, self.config.initialization
            )
        else:
            left_hand_model, right_hand_model = initialize_dual_hand(
                right_hand_model, self.object_model, self.config.initialization
            )
        self.bimanual_pair = BimanualPair(left_hand_model, right_hand_model, self.device)
        
        # Save initial poses for optional debugging
        self.left_hand_pose_st = left_hand_model.hand_pose.detach()
        self.right_hand_pose_st = right_hand_model.hand_pose.detach()
        
        print(f'Left hand contact candidates: {left_hand_model.n_contact_candidates}')
        print(f'Right hand contact candidates: {right_hand_model.n_contact_candidates}')
        print(f'Total batch size: {self.config.total_batch_size}')
    
    def setup_optimization(self):
        """Initialize optimizer and energy computer."""
        
        # Create energy computer with optimized FC+VEW computation
        self.energy_computer = BimanualEnergyComputer(self.config.energy, self.device)
        
        # Create optimizer
        self.optimizer = MALAOptimizer(
            self.bimanual_pair.left, 
            self.bimanual_pair.right, 
            config=self.config.optimizer,
            device=self.device
        )
        
    def setup_logging(self):
        """Setup experiment logging and result directories."""
        
        # Create directories
        logs_path = self.config.paths.get_experiment_logs_path(self.config.name)
        results_path = self.config.paths.get_experiment_results_path(self.config.name)
        
        ensure_directory(logs_path, clean=True)
        ensure_directory(results_path, clean=True)
        
        # Create logger
        self.logger = Logger(
            log_dir=logs_path,
            thres_fc=self.config.energy.thres_fc,
            thres_dis=self.config.energy.thres_dis,
            thres_pen=self.config.energy.thres_pen
        )
        
        # Save experiment configuration
        config_path = os.path.join(results_path, 'config.txt')
        with open(config_path, 'w') as f:
            f.write(str(self.config))
        
    def run_optimization(self):
        """Run the optimization loop."""
        print("Starting optimization...")
        
        self.profiler.enable()
        
        # Initial energy computation
        energy_terms = self.energy_computer.compute_all_energies(
            self.bimanual_pair, self.object_model, verbose=True
        )
        
        energy_terms.total.sum().backward(retain_graph=True)
        self.logger.log(
            energy_terms.total, energy_terms.force_closure, energy_terms.distance,
            energy_terms.penetration, energy_terms.self_penetration, 
            energy_terms.joint_limits, 0, show=False
        )
        
        results_path = self.config.paths.get_experiment_results_path(self.config.name)

        # Initialize visualization recorder if enabled
        recorders = []
        if getattr(self.config, 'vis', None) and self.config.vis.enabled:
            # Support multiple recordings in a single batch
            n_record = max(1, int(self.config.vis.record_num))
            base_local = int(self.config.vis.sample_local_index)
            for k in range(n_record):
                global_idx = self.config.vis.sample_object_index * self.object_model.batch_size_each + (base_local + k)
                frames_dir = os.path.join(results_path, f"{self.config.vis.out_dirname}_{k}")
                recorder = FrameRecorder(
                    object_model=self.object_model,
                    bimanual_pair=self.bimanual_pair,
                    global_idx=global_idx,
                    frames_dir=frames_dir,
                    width=self.config.vis.width,
                    height=self.config.vis.height,
                    show_contacts=self.config.vis.show_contacts,
                    bg_color=self.config.vis.bg_color
                )
                recorder.capture(step=0)
                recorders.append((k, recorder))
        
        # Initialize metrics recorder if enabled
        metrics_recorder = None
        if getattr(self.config, 'metrics', None) and self.config.metrics.enabled:
            metrics_dir = os.path.join(results_path, 'metrics')
            metrics_recorder = SimpleMetricsRecorder(metrics_dir)

        # Main optimization loop
        for step in tqdm(range(1, self.config.optimizer.num_iterations + 1), desc='optimizing'):
            # MALA proposal step with Langevin dynamics
            step_size = self.optimizer.langevin_proposal()
            
            # Zero gradients and compute new energy
            self.optimizer.zero_grad()
            new_energy_terms = self.energy_computer.compute_all_energies(
                self.bimanual_pair, self.object_model, verbose=True
            )
            
            new_energy_terms.total.sum().backward(retain_graph=True)
            
            # Metropolis-Hastings acceptance step
            with torch.no_grad():
                accept, temperature = self.optimizer.metropolis_hastings_step(
                    energy_terms.total, new_energy_terms.total
                )
                
                # Update energies for accepted samples
                energy_terms.total[accept] = new_energy_terms.total[accept]
                energy_terms.distance[accept] = new_energy_terms.distance[accept]
                energy_terms.force_closure[accept] = new_energy_terms.force_closure[accept]
                energy_terms.penetration[accept] = new_energy_terms.penetration[accept]
                energy_terms.self_penetration[accept] = new_energy_terms.self_penetration[accept]
                energy_terms.joint_limits[accept] = new_energy_terms.joint_limits[accept]
                energy_terms.wrench_volume[accept] = new_energy_terms.wrench_volume[accept]
                
                # Log progress
                self.logger.log(
                    energy_terms.total, energy_terms.force_closure, energy_terms.distance,
                    energy_terms.penetration, energy_terms.self_penetration,
                    energy_terms.joint_limits, step, show=False
                )

            # Visualization capture per stride
            if recorders and (step % max(1, self.config.vis.frame_stride) == 0):
                for k, rec in recorders:
                    try:
                        rec.capture(step)
                    except Exception as e:
                        print(f"[WARN] Visualization capture failed at step {step} for recorder {k}: {e}")

            # Metrics recording (batch means)
            if metrics_recorder is not None and (step % max(1, self.config.metrics.stride) == 0):
                try:
                    accept_rate = accept.float().mean().item()
                    temperature = (self.optimizer.initial_temperature *
                                   self.optimizer.cooling_schedule ** torch.div(self.optimizer.step, self.optimizer.annealing_period, rounding_mode='floor'))
                    step_size_cur = (self.optimizer.step_size *
                                     self.optimizer.cooling_schedule ** torch.div(self.optimizer.step, self.optimizer.step_size_period, rounding_mode='floor'))
                    # Weighted contributions
                    w_dis = self.config.energy.w_dis
                    w_pen = self.config.energy.w_pen
                    w_spen = self.config.energy.w_spen
                    w_joints = self.config.energy.w_joints
                    w_vew = self.config.energy.w_vew
                    row = {
                        'step': step,
                        'total': energy_terms.total.mean().item(),
                        'w_dis*dis': (w_dis * energy_terms.distance.mean()).item(),
                        'w_pen*pen': (w_pen * energy_terms.penetration.mean()).item(),
                        'w_spen*spen': (w_spen * energy_terms.self_penetration.mean()).item(),
                        'w_joints*joints': (w_joints * energy_terms.joint_limits.mean()).item(),
                        'w_vew*vew': (w_vew * energy_terms.wrench_volume.mean()).item(),
                        'accept_rate': accept_rate,
                        'temperature': float(temperature.detach().cpu().item()) if hasattr(temperature, 'detach') else float(temperature),
                        'step_size': float(step_size_cur.detach().cpu().item()) if hasattr(step_size_cur, 'detach') else float(step_size_cur),
                    }
                    metrics_recorder.log_row(row)
                except Exception as e:
                    print(f"[WARN] Metrics recording failed at step {step}: {e}")
        
        self.profiler.disable()

        # Finalize videos if enabled
        if recorders:
            for k, rec in recorders:
                try:
                    base, ext = os.path.splitext(self.config.vis.video_filename)
                    video_name = f"{base}_{k}{ext or '.mp4'}"
                    video_path = os.path.join(results_path, video_name)
                    rec.finalize(video_path, fps=self.config.vis.fps)
                    print(f"Saved optimization video to: {video_path}")
                except Exception as e:
                    print(f"[WARN] Visualization finalize failed for recorder {k}: {e}")
        # Finalize metrics (write CSV and plot)
        if metrics_recorder is not None:
            try:
                metrics_recorder.finalize()
                print(f"Saved metrics to: {os.path.join(results_path, 'metrics')}")
            except Exception as e:
                print(f"[WARN] Metrics finalize failed: {e}")
        return energy_terms
    
    def save_intermediate_results(self, step: int, energy_terms: EnergyTerms):
        """Save intermediate results during optimization."""
        results_path = self.config.paths.get_experiment_results_path(self.config.name)
        
        save_grasp_results(
            results_path, self.config.object_code_list, self.config.model.batch_size,
            self.object_model, self.bimanual_pair,
            self.left_hand_pose_st, self.right_hand_pose_st,
            energy_terms, step=step
        )
    
    def save_final_results(self, energy_terms: EnergyTerms):
        """Save final optimization results."""
        print("Saving final results...")
        
        results_path = self.config.paths.get_experiment_results_path(self.config.name)
        
        save_grasp_results(
            results_path, self.config.object_code_list, self.config.model.batch_size,
            self.object_model, self.bimanual_pair,
            self.left_hand_pose_st, self.right_hand_pose_st,
            energy_terms, step=None
        )
    
    def print_performance_stats(self):
        """Print performance and profiling statistics."""
        print("\n=== Performance Statistics ===")
        # self.profiler.print_stats()
        
        if HAS_MEMORY_PROFILER:
            try:
                memory_usage = memory_profiler.memory_usage(-1, interval=1)
                print(f"Peak memory usage: {max(memory_usage):.2f} MB")
            except (RuntimeError, OSError) as e:
                print(f"Memory profiling error: {e}")
        else:
            print("Memory profiling unavailable (memory_profiler not installed)")
    
    def run_full_experiment(self) -> EnergyTerms:
        """Run the complete experiment pipeline."""
        print(f"Starting experiment: {self.config.name}")
        
        # Setup pipeline
        self.setup_environment()
        self.setup_models()
        self.setup_optimization()
        self.setup_logging()
        
        # Run optimization
        final_energy_terms = self.run_optimization()
        
        # Save results
        self.save_final_results(final_energy_terms)

        # Print statistics
        self.print_performance_stats()
        
        print(f"Experiment completed: {self.config.name}")
        return final_energy_terms


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bimanual Grasp Generation')
    
    # Core experiment settings
    parser.add_argument('--name', default='test', type=str, help='Experiment name')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--gpu', default="0", type=str)
    
    # Object and batch configuration
    parser.add_argument('--object_code_list', default=
    [
        # 'Cole_Hardware_Dishtowel_Multicolors',
        # 'Curver_Storage_Bin_Black_Small', 
        # 'Hasbro_Monopoly_Hotels_Game',
        # 'Breyer_Horse_Of_The_Year_2015',
        # 'Schleich_S_Bayala_Unicorn_70432',
        # '11pro_SL_TRX_FG'
        # 'pot_rec',
        '100015',
        '100021',
        '100023',
        '100025',
        '100028',
        '100032',
        '100040',
        '100045',
        '100047',
        '100051',
        '100054',
        '100057',
        '100058'
    ]
    , type=list, help='List of object codes to process')
    
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size per object')
    parser.add_argument('--num_iterations', default=10000, type=int, help='Number of optimization iterations')
    parser.add_argument('--object_code', default=None, type=str, help='Single object code to override list')
    
    # Visualization CLI
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--vis_frame_stride', default=50, type=int)
    parser.add_argument('--vis_obj', default=0, type=int)
    parser.add_argument('--vis_local', default=0, type=int)
    parser.add_argument('--vis_fps', default=30, type=int)
    parser.add_argument('--vis_width', default=900, type=int)
    parser.add_argument('--vis_height', default=900, type=int)
    parser.add_argument('--vis_contacts', action='store_true')
    parser.add_argument('--vis_record_num', default=1, type=int)

    # Target init CLI
    parser.add_argument('--init_at_targets', action='store_true')
    parser.add_argument('--left_target', default=None, type=str)
    parser.add_argument('--right_target', default=None, type=str)
    parser.add_argument('--target_distance', default=0.22, type=float)
    parser.add_argument('--target_jitter_dist', default=0.0, type=float)
    parser.add_argument('--target_jitter_angle', default=0.0, type=float)
    parser.add_argument('--target_twist_range', default=0.0, type=float)
    parser.add_argument('--twist_mirror', action='store_true')
    parser.add_argument('--right_twist_offset', default=0.0, type=float)
    parser.add_argument('--interact', action='store_true')
    parser.add_argument('--snap_to_surface', action='store_true')
    parser.add_argument('--ui_port', default=8050, type=int)
    
    # Metrics CLI (minimal)
    parser.add_argument('--metrics', action='store_true')
    parser.add_argument('--metrics_stride', default=50, type=int)

    # Dataset collection loop
    parser.add_argument('--target_total_per_object', default=0, type=int, help='If >0, collect at least this many grasps per object by running multiple rounds')
    parser.add_argument('--max_rounds', default=1000, type=int, help='Safety limit on collection rounds')
    parser.add_argument('--seed_base', default=None, type=int, help='Optional base seed; if not provided, uses current --seed')
    
    
    args = parser.parse_args()
    
    # Create config from args or use defaults
    if hasattr(args, 'config_file') and args.config_file:
        # Future: Load from config file
        config = ExperimentConfig()
        config.update_from_args(args)
    else:
        config = create_config_from_args(args)

    # Optional: override object list with single code
    if getattr(args, 'object_code', None):
        config.object_code_list = [args.object_code]
    
    # Print configuration summary
    print("=== Experiment Configuration ===")
    print(f"Name: {config.name}")
    print(f"Objects: {len(config.object_code_list)} objects")
    print(f"Batch size: {config.model.batch_size} per object")
    print(f"Total batch: {config.total_batch_size}")
    print(f"iterations: {config.optimizer.num_iterations}")
    print(f"Energy weights: dis={config.energy.w_dis}, pen={config.energy.w_pen}, vew={config.energy.w_vew}")
    print(f"temperature: {config.optimizer.initial_temperature}")
    print(f"Langevin noise: {config.optimizer.langevin_noise_factor}")
    print("=" * 45)
    
    def accumulate_results(src_results_path: str, dst_results_path: str, object_codes):
        os.makedirs(dst_results_path, exist_ok=True)
        added = {obj: 0 for obj in object_codes}
        for obj in object_codes:
            src_file = os.path.join(src_results_path, f'{obj}.npy')
            if not os.path.exists(src_file):
                continue
            cur_list = np.load(src_file, allow_pickle=True).tolist()
            dst_file = os.path.join(dst_results_path, f'{obj}.npy')
            if os.path.exists(dst_file):
                agg_list = np.load(dst_file, allow_pickle=True).tolist()
            else:
                agg_list = []
            # Append all from current round; trimming will be handled by caller based on target
            agg_list.extend(cur_list)
            np.save(dst_file, agg_list, allow_pickle=True)
            added[obj] = len(cur_list)
        return added

    def counts_in_results(dst_results_path: str, object_codes):
        counts = {}
        for obj in object_codes:
            dst_file = os.path.join(dst_results_path, f'{obj}.npy')
            if os.path.exists(dst_file):
                counts[obj] = len(np.load(dst_file, allow_pickle=True).tolist())
            else:
                counts[obj] = 0
        return counts

    target_total = int(getattr(args, 'target_total_per_object', 0) or 0)
    if target_total <= 0:
        # Single run mode
        experiment = GraspExperiment(config)
        final_energy_terms = experiment.run_full_experiment()
    else:
        original_name = config.name
        base_seed = int(args.seed_base) if getattr(args, 'seed_base', None) is not None else int(args.seed)
        dataset_results_path = config.paths.get_experiment_results_path(original_name + '_dataset')
        os.makedirs(dataset_results_path, exist_ok=True)
        round_idx = 0
        while round_idx < int(args.max_rounds):
            # Check current counts
            current = counts_in_results(dataset_results_path, config.object_code_list)
            if all(cnt >= target_total for cnt in current.values()):
                print('Target totals reached for all objects:', current)
                break
            # Prepare round config
            config.name = f"{original_name}_r{round_idx}"
            config.seed = base_seed + round_idx
            print(f"[Collection] Round {round_idx} with seed {config.seed}")
            experiment = GraspExperiment(config)
            _ = experiment.run_full_experiment()
            # Accumulate into dataset folder
            round_results_path = config.paths.get_experiment_results_path(config.name)
            accumulate_results(round_results_path, dataset_results_path, config.object_code_list)
            # Trim to target
            for obj in config.object_code_list:
                dst_file = os.path.join(dataset_results_path, f'{obj}.npy')
                if os.path.exists(dst_file):
                    lst = np.load(dst_file, allow_pickle=True).tolist()
                    if len(lst) > target_total:
                        lst = lst[:target_total]
                        np.save(dst_file, lst, allow_pickle=True)
            round_idx += 1
        # Restore name
        config.name = original_name
