import os
import csv
from typing import Dict, List


class SimpleMetricsRecorder:
    """
    Simple metrics recorder that collects batch-mean metrics every N steps
    and writes a CSV, plus renders a simple matplotlib plot at the end.
    """

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.rows: List[Dict[str, float]] = []
        self.csv_path = os.path.join(self.out_dir, 'metrics_summary.csv')
        self.plot_path = os.path.join(self.out_dir, 'metrics_plot.png')

    def log_row(self, row: Dict[str, float]):
        # Keep consistent key order on write
        self.rows.append(row)

    def finalize(self):
        if not self.rows:
            return
        # Write CSV
        fieldnames = list(self.rows[0].keys())
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.rows:
                writer.writerow(r)

        # Save simple plot (requires matplotlib)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            steps = [r['step'] for r in self.rows]
            # Plot weighted metrics if available, else raw
            metrics_to_plot = []
            if 'w_dis*dis' in self.rows[0]:
                metrics_to_plot.extend([
                    ('total', 'Total Energy'),
                    ('w_dis*dis', 'w_dis*dis'),
                    ('w_pen*pen', 'w_pen*pen'),
                    ('w_spen*spen', 'w_spen*spen'),
                    ('w_joints*joints', 'w_joints*joints'),
                    ('w_vew*vew', 'w_vew*vew'),
                ])
            else:
                metrics_to_plot.extend([
                    ('total', 'Total Energy'),
                    ('fc', 'Force Closure'),
                    ('dis', 'Distance'),
                    ('pen', 'Penetration'),
                    ('spen', 'Self Penetration'),
                    ('joints', 'Joint Limits'),
                    ('vew', 'Wrench Volume'),
                ])
            plt.figure(figsize=(10, 6))
            for key, label in metrics_to_plot:
                if key in self.rows[0]:
                    plt.plot(steps, [r[key] for r in self.rows], label=label)
            if 'accept_rate' in self.rows[0]:
                plt.plot(steps, [r['accept_rate'] for r in self.rows], label='Accept Rate')
            plt.xlabel('Step')
            plt.ylabel('Batch Mean')
            plt.title('Optimization Metrics')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.plot_path)
            plt.close()
        except Exception:
            # Silently ignore plotting errors
            pass

