import os
import json
import threading
import numpy as np
import torch
import plotly.graph_objects as go
import trimesh as tm
from dash import Dash, dcc, html, Input, Output, State
from dash import callback_context as ctx


def _to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def save_preview(object_model, A_scaled, B_scaled, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # Use first object's mesh for preview
    mesh = object_model.object_mesh_list[0]
    scale = float(object_model.object_scale_tensor[0][0].item()) if object_model.object_scale_tensor.ndim == 2 else float(object_model.object_scale_tensor[0].item())
    verts = mesh.vertices * scale
    faces = mesh.faces
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], color='lightgray', opacity=0.9))
    A = _to_numpy(A_scaled)
    B = _to_numpy(B_scaled)
    fig.add_trace(go.Scatter3d(x=[A[0]], y=[A[1]], z=[A[2]], mode='markers', marker=dict(size=6, color='red'), name='Left'))
    fig.add_trace(go.Scatter3d(x=[B[0]], y=[B[1]], z=[B[2]], mode='markers', marker=dict(size=6, color='blue'), name='Right'))
    fig.update_layout(scene_aspectmode='data')
    png_path = os.path.join(out_dir, 'interaction_preview.png')
    html_path = os.path.join(out_dir, 'interaction_preview.html')
    try:
        fig.write_image(png_path, width=900, height=900)
    except Exception:
        pass
    fig.write_html(html_path)
    return png_path, html_path


def select_two_points_interactive(object_model, init_cfg, results_path):
    """
    Launch a simple Dash UI to pick two points (Left then Right) with sliders and confirm buttons.
    Returns the two scaled points and saves an HTML/PNG preview.
    """
    mesh = object_model.object_mesh_list[0]
    scale = float(object_model.object_scale_tensor[0][0].item()) if object_model.object_scale_tensor.ndim == 2 else float(object_model.object_scale_tensor[0].item())
    verts = mesh.vertices * scale
    faces = mesh.faces
    bounds_min = verts.min(axis=0)
    bounds_max = verts.max(axis=0)
    center = verts.mean(axis=0)

    # Initial defaults
    A0 = np.array([center[0] + 0.1, center[1], center[2]], dtype=np.float32)
    B0 = np.array([center[0] - 0.1, center[1], center[2]], dtype=np.float32)

    state = {
        'step': 'left',
        'left': A0.copy(),
        'right': B0.copy(),
        'done': threading.Event(),
    }

    def make_figure(point, step_color='red'):
        fig = go.Figure()
        fig.add_trace(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], color='lightgray', opacity=0.9))
        fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]], mode='markers', marker=dict(size=6, color=step_color)))
        fig.update_layout(scene_aspectmode='data')
        return fig

    app = Dash(__name__)
    x_range = [float(bounds_min[0] - 0.2), float(bounds_max[0] + 0.2)]
    y_range = [float(bounds_min[1] - 0.2), float(bounds_max[1] + 0.2)]
    z_range = [float(bounds_min[2] - 0.2), float(bounds_max[2] + 0.2)]

    app.layout = html.Div([
        html.H4(id='title', children='Select Left hand position'),
        dcc.Graph(id='scene', figure=make_figure(state['left'], 'red')),
        html.Div([
            html.Div(['X', dcc.Slider(id='x', min=x_range[0], max=x_range[1], step=0.005,
                                       value=float(state['left'][0]), marks={}, updatemode='drag',
                                       tooltip={'always_visible': False, 'placement': 'bottom'})]),
            html.Div(['Y', dcc.Slider(id='y', min=y_range[0], max=y_range[1], step=0.005,
                                       value=float(state['left'][1]), marks={}, updatemode='drag',
                                       tooltip={'always_visible': False, 'placement': 'bottom'})]),
            html.Div(['Z', dcc.Slider(id='z', min=z_range[0], max=z_range[1], step=0.005,
                                       value=float(state['left'][2]), marks={}, updatemode='drag',
                                       tooltip={'always_visible': False, 'placement': 'bottom'})]),
        ], style={'margin': '10px'}),
        html.Div([
            html.Button('Confirm', id='confirm', n_clicks=0),
        ], style={'margin': '10px'}),
        html.Div(id='status'),
        html.Div(id='xyzreadout', style={'margin': '10px', 'fontFamily': 'monospace'})
    ])

    @app.callback(
        Output('scene', 'figure'), Output('title', 'children'), Output('status', 'children'),
        Output('x', 'value'), Output('y', 'value'), Output('z', 'value'), Output('xyzreadout', 'children'),
        Input('x', 'value'), Input('y', 'value'), Input('z', 'value'),
        Input('confirm', 'n_clicks'),
        prevent_initial_call=False
    )
    def update_scene(x, y, z, confirm_clicks):
        # Current point depending on step
        if state['step'] == 'left':
            cur = state['left']
            color = 'red'
            title = 'Select Left hand position'
        else:
            cur = state['right']
            color = 'blue'
            title = 'Select Right hand position'

        # Update by sliders
        if x is not None:
            cur[0] = float(x)
        if y is not None:
            cur[1] = float(y)
        if z is not None:
            cur[2] = float(z)

        changed = ctx.triggered
        msg = ''
        if changed:
            trig = changed[0]['prop_id'].split('.')[0]
            if trig == 'confirm':
                if state['step'] == 'left':
                    state['left'] = cur.copy()
                    state['step'] = 'right'
                    msg = f'Left fixed at {state["left"].tolist()}'
                    # Reset sliders to right defaults
                    cur = state['right']
                    x, y, z = float(cur[0]), float(cur[1]), float(cur[2])
                else:
                    state['right'] = cur.copy()
                    msg = f'Right fixed at {state["right"].tolist()} - Done'
                    state['done'].set()

        fig = make_figure(cur, color)
        # Ensure sliders reflect current point and show current xyz
        xyz_txt = f"Current XYZ: [{cur[0]:.4f}, {cur[1]:.4f}, {cur[2]:.4f}]"
        return fig, title, msg, float(cur[0]), float(cur[1]), float(cur[2]), xyz_txt

    # Run server in background thread
    port = int(getattr(init_cfg, 'ui_port', 8050))
    t = threading.Thread(target=lambda: app.run(debug=False, use_reloader=False, port=port, host='0.0.0.0'))
    t.daemon = True
    t.start()

    # Wait until user confirms both points
    state['done'].wait()

    A = state['left']
    B = state['right']
    png_path, html_path = save_preview(object_model, A, B, results_path)
    info = {
        'left_target_scaled': A.tolist(),
        'right_target_scaled': B.tolist(),
        'snap_to_surface': bool(getattr(init_cfg, 'snap_to_surface', False))
    }
    with open(os.path.join(results_path, 'interaction.json'), 'w') as f:
        json.dump(info, f, indent=2)
    with open(os.path.join(results_path, 'interaction.txt'), 'w') as f:
        f.write(json.dumps(info))
    return A, B, {'png': png_path, 'html': html_path}


def select_two_points_interactive_from_disk(paths_cfg, object_code: str, init_cfg, results_path):
    """
    Lightweight fallback that loads mesh directly from disk without ObjectModel,
    computes a default scale (0.15), proposes symmetric targets, and saves preview.
    """
    os.makedirs(results_path, exist_ok=True)
    mesh_path = os.path.join(paths_cfg.data_root_path, object_code, 'coacd', 'decomposed.obj')
    mesh = tm.load(mesh_path, force='mesh', process=False)
    scale = 0.15
    verts = mesh.vertices * scale
    faces = mesh.faces
    center = verts.mean(axis=0)

    def parse_vec3(s, default):
        if s is None:
            return np.array(default, dtype=np.float32)
        parts = s.strip().split()
        if len(parts) != 3:
            return np.array(default, dtype=np.float32)
        return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)

    A = parse_vec3(getattr(init_cfg, 'left_target', None), default=[center[0] + 0.1, center[1], center[2]])
    B = parse_vec3(getattr(init_cfg, 'right_target', None), default=[center[0] - 0.1, center[1], center[2]])

    fig = go.Figure()
    fig.add_trace(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], color='lightgray', opacity=0.9))
    fig.add_trace(go.Scatter3d(x=[A[0]], y=[A[1]], z=[A[2]], mode='markers', marker=dict(size=6, color='red'), name='Left'))
    fig.add_trace(go.Scatter3d(x=[B[0]], y=[B[1]], z=[B[2]], mode='markers', marker=dict(size=6, color='blue'), name='Right'))
    fig.update_layout(scene_aspectmode='data')
    png_path = os.path.join(results_path, 'interaction_preview.png')
    html_path = os.path.join(results_path, 'interaction_preview.html')
    try:
        fig.write_image(png_path, width=900, height=900)
    except Exception:
        pass
    fig.write_html(html_path)

    info = {
        'object_code': object_code,
        'left_target_scaled': A.tolist(),
        'right_target_scaled': B.tolist(),
        'snap_to_surface': bool(getattr(init_cfg, 'snap_to_surface', False))
    }
    with open(os.path.join(results_path, 'interaction.json'), 'w') as f:
        json.dump(info, f, indent=2)
    with open(os.path.join(results_path, 'interaction.txt'), 'w') as f:
        f.write(json.dumps(info))
    return A, B, {'png': png_path, 'html': html_path}


