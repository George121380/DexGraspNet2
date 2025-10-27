import os
import plotly.graph_objects as go


class FrameRecorder:
    """
    Capture plotly frames during optimization and stitch them into a video.
    Relies on kaleido for image export and imageio for video writing.
    """

    def __init__(self, object_model, bimanual_pair, global_idx, frames_dir,
                 width=900, height=900, show_contacts=False, bg_color="#E2F0D9"):
        self.object_model = object_model
        self.bimanual_pair = bimanual_pair
        self.global_idx = int(global_idx)
        self.frames_dir = frames_dir
        self.width = int(width)
        self.height = int(height)
        self.show_contacts = bool(show_contacts)
        self.bg_color = bg_color
        os.makedirs(self.frames_dir, exist_ok=True)

    def _build_figure(self):
        left_traces = self.bimanual_pair.left.get_plotly_data(
            i=self.global_idx, opacity=1.0, color='lightslategray',
            with_contact_points=self.show_contacts
        )
        right_traces = self.bimanual_pair.right.get_plotly_data(
            i=self.global_idx, opacity=1.0, color='lightslategray',
            with_contact_points=self.show_contacts
        )
        obj_traces = self.object_model.get_plotly_data(
            i=self.global_idx, color='seashell', opacity=1.0
        )
        fig = go.Figure(right_traces + obj_traces + left_traces)
        fig.update_layout(paper_bgcolor=self.bg_color, plot_bgcolor=self.bg_color)
        fig.update_layout(scene_aspectmode='data')
        fig.update_layout(scene=dict(
            xaxis=dict(visible=False, showgrid=False, showline=False, zeroline=False, showticklabels=False),
            yaxis=dict(visible=False, showgrid=False, showline=False, zeroline=False, showticklabels=False),
            zaxis=dict(visible=False, showgrid=False, showline=False, zeroline=False, showticklabels=False)
        ))
        return fig

    def capture(self, step: int):
        fig = self._build_figure()
        out_path = os.path.join(self.frames_dir, f"frame_{int(step):06d}.png")
        fig.write_image(out_path, width=self.width, height=self.height)

    def finalize(self, video_path: str, fps: int = 30):
        import imageio
        frames = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.png')])
        if len(frames) == 0:
            return
        with imageio.get_writer(video_path, fps=int(fps)) as writer:
            for fname in frames:
                writer.append_data(imageio.v2.imread(os.path.join(self.frames_dir, fname)))

