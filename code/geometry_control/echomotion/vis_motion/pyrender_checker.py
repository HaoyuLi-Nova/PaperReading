import os
import sys
import argparse
import numpy as np
import torch
import cv2
import trimesh
import pyrender
import colorsys
import smplx
from torch.nn import functional as F
import torchgeometry as tgm
from tqdm import tqdm

import sys
file_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(file_path))
smplx_root = torch.tensor([[ 0.0031, -0.3514,  0.0120]], dtype=torch.float32)

from common.motion_rep import motion_rep_to_SMPL, apply_inverse_rotation
from common.constants import JOINT_NUM

os.environ['PYOPENGL_PLATFORM'] = 'egl'
SMPL_PATH = 'extract_motion/CameraHMR/data/models/SMPL/SMPL_NEUTRAL.pkl'


class Renderer:
    def __init__(self, focal_length=1000, img_w=512, img_h=512, faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_w, viewport_height=img_h, point_size=1.0)
        self.camera_center = [img_w / 2., img_h / 2.]
        self.focal_length = focal_length
        self.faces = faces

    def render_view(self, verts, R_view, T_view):
        scene = pyrender.Scene(bg_color=(1., 1., 1., 0.), ambient_light=np.array([0.1, 0.1, 0.1, 1.0]))
        
        camera = pyrender.camera.IntrinsicsCamera(
            fx=self.focal_length, fy=self.focal_length,
            cx=self.camera_center[0], cy=self.camera_center[1]
        )
        
        camera_R = R_view.T
        camera_T = -R_view.T @ T_view
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = camera_R
        camera_pose[:3, 3] = camera_T
        scene.add(camera, pose=camera_pose)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        scene.add(light, pose=camera_pose)

        # flip x-axis
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])

        mesh_pose = np.eye(4)
        mesh_pose[:3, :3] = cv2.Rodrigues(np.array([0., 0., 0.]))[0]

        # for n in range(verts.shape[0]):
        mesh = trimesh.Trimesh(verts[0], self.faces)
        # apply rotation
        mesh.apply_transform(rot)

        mesh_color = colorsys.hsv_to_rgb(0.6, 0.5, 1.0)
        material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.1, alphaMode='OPAQUE', baseColorFactor=mesh_color)
        mesh_node = pyrender.Mesh.from_trimesh(mesh, material=material)
        scene.add(mesh_node, pose=mesh_pose)

        color_rgba, _ = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        return color_rgba[:, :, :3]


    def delete(self):
        self.renderer.delete()

class VideoRenderer:
    def __init__(self, renderer, output_path, fps=30):
        self.renderer = renderer
        self.output_path = output_path
        self.fps = fps
        self.writer = None

    def _initialize_writer(self, frame_w, frame_h):
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (frame_w, frame_h))

    def render_video(self, verts_sequence, R_sequence, T_sequence, desc=""):
        num_frames = verts_sequence.shape[0]
        for i in tqdm(range(num_frames), desc=desc, leave=False):
            current_verts = verts_sequence[i:i+1]
            current_R = R_sequence[i]
            current_T = T_sequence[i]
            rendered_frame = self.renderer.render_view(current_verts, R_view=current_R, T_view=current_T)
            if self.writer is None:
                h, w, _ = rendered_frame.shape
                self._initialize_writer(w, h)
            frame_to_write = cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR)
            self.writer.write(frame_to_write)
        if self.writer: self.writer.release()

def get_R_T(joints, set_center=True, zero_trans=False):
    # joints: (seq_len, joints_num, 3), roots: (seq_len, 3)
    # R: (seq_len, 3, 3), T: (seq_len, 3)
    seq_len = joints.shape[0]
    roots = joints[:, 0]
    xyz_move = roots.amax(dim=0) - roots.amin(dim=0)
    y_max = roots.amax(dim=0)[1]
    x_move = xyz_move[0]
    z_move = xyz_move[2]
    depth_offset = 2.5 + 1.0 * x_move + 2.0 * z_move + y_max
    # Calculate movement ranges and means
    R = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=torch.float32, device=joints.device)
    T = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=joints.device) * depth_offset
    if set_center:
        T[1] = - roots[0, 2]
    if zero_trans:
        T = torch.zeros_like(T)
    R, T = R[None,].repeat(seq_len, 1, 1), T[None].repeat(seq_len, 1)
    return R, T

def rot6d_to_axis_angle(rot6d):
    batch_size = rot6d.shape[0]

    rot6d = rot6d.view(batch_size, 3, 2)
    a1 = rot6d[:, :, 0]
    a2 = rot6d[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2, dim=-1)
    rot_mat = torch.stack((b1, b2, b3), dim=-1)  # 3x3 rotation matrix

    rot_mat = torch.cat([rot_mat, torch.zeros((batch_size, 3, 1), device=rot_mat.device).float()],
                        2)  # 3x4 rotation matrix

    axis_angle = tgm.rotation_matrix_to_angle_axis(rot_mat).reshape(-1, 3)  # axis-angle
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle

def axis_angle_to_rotation_matrix(axis_angle):
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / (angle + 1e-8)
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    outer = torch.bmm(axis.unsqueeze(-1), axis.unsqueeze(-2))
    skew = torch.zeros((*axis.shape[:-1], 3, 3), device=axis.device)
    skew[..., 0, 1] = -axis[..., 2]
    skew[..., 0, 2] = axis[..., 1]
    skew[..., 1, 0] = axis[..., 2]
    skew[..., 1, 2] = -axis[..., 0]
    skew[..., 2, 0] = -axis[..., 1]
    skew[..., 2, 1] = axis[..., 0]
    rotation_matrix = cos_angle.unsqueeze(-1) * torch.eye(3, device=axis.device) + \
                      (1 - cos_angle.unsqueeze(-1)) * outer + \
                      sin_angle.unsqueeze(-1) * skew
    return rotation_matrix



def rot6d_to_mat3x3(rot6d):
    rot6d = rot6d.view(-1, 3, 2)
    a1 = rot6d[:, :, 0]
    a2 = rot6d[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def estimate_focal_length(img_w, img_h, fov=55):
    fov_rad = fov * np.pi / 180
    return max(img_w, img_h) / (2 * np.tan(fov_rad / 2))



def visualize_single_motion(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        print(f"Loading motion file: {args.motion_file}")
        motion_data = torch.load(args.motion_file, map_location=device, weights_only=True)
        motion_tensor = motion_data["motion"]
        
        smpl_params, joints = motion_rep_to_SMPL(motion_tensor, use_shape=args.use_shape, fetch_local_motion=True, init_canonical=True)

        print("Estimating camera parameters using a simple, robust method for debugging...")
        R_view_seq, T_view_seq = get_R_T(joints, zero_trans=args.zero_trans)
        
        focal = estimate_focal_length(args.W, args.H)
        
        print("Initializing SMPL model...")
        kwargs = {'model_path': SMPL_PATH, 'gender': 'neutral', 'num_betas': 10, 'batch_size': motion_tensor.shape[0]}
        smpl_model = smplx.SMPLLayer(**kwargs).to(device)

        smpl_params['body_pose'] = axis_angle_to_rotation_matrix(smpl_params['body_pose'].reshape(-1, 3)).reshape(joints.shape[0], -1, 3)
        print(smpl_params['body_pose'].shape, smpl_params['global_orient'].shape, smpl_params['transl'].shape)
        model_output = smpl_model(**smpl_params)
        verts_seq = model_output.vertices.detach().cpu().numpy()
        faces_data = smpl_model.faces

        os.makedirs(args.output_dir, exist_ok=True)
        motion_name = os.path.splitext(os.path.basename(args.motion_file))[0]
        output_video_path = os.path.join(args.output_dir, f"{motion_name}_rendered.mp4")
        
        if os.path.exists(output_video_path) and not args.overwrite:
            print(f"Output video already exists: {output_video_path}. Use --overwrite to replace.")
            return

        # --- Render ---
        renderer = Renderer(focal_length=focal, img_w=args.W, img_h=args.H, faces=faces_data)
        video_renderer = VideoRenderer(renderer, output_video_path, fps=args.fps)
        
        R_view_np = R_view_seq.detach().cpu().numpy()
        T_view_np = T_view_seq.detach().cpu().numpy()
        
        video_renderer.render_video(
            verts_sequence=verts_seq,
            R_sequence=R_view_np,
            T_sequence=T_view_np,
            desc=motion_name
        )
        renderer.delete()
        print(f"Successfully saved video to {output_video_path}")

    except Exception as e:
        print(f"\nAn unexpected error occurred:")
        import traceback
        traceback.print_exc()

def motion_vis_during_validation(
    motion_data=None, 
    output_dir=None,
    H=1024, 
    W=1024, 
    fps=20, 
    motion_name=None, 
    zero_trans=False, 
    device='cuda:0', 
    verbose=False, 
    output_video_path='render_out.mp4', 
    is_6d=True, 
    use_shape=False,
):
    seq_len = motion_data.shape[0]

    if is_6d:
        body_pose_6d = motion_data[:, 6:6+(JOINT_NUM-1)*6].reshape(-1, 6)
        body_pose_aa = rot6d_to_axis_angle(body_pose_6d).reshape(-1, 3)
        body_pose_mat_flat = axis_angle_to_rotation_matrix(body_pose_aa) 
        body_pose_final = body_pose_mat_flat.reshape(seq_len, -1, 3, 3)

        smpl_params = {
            'global_orient': rot6d_to_axis_angle(motion_data[:, :6]),
            'body_pose': body_pose_final,
            'transl': motion_data[:, 6+(JOINT_NUM-1)*6:9+(JOINT_NUM-1)*6], 
            'betas': motion_data[:, -10:] if use_shape else None,
        }
        joints = torch.zeros(seq_len, JOINT_NUM, 3, device=device)

    # --- 2. SMPL model and vertex generation (using new, correct logic) ---
    if verbose: print(f"Initializing SMPL model for visualization...")
    
    R_motionx_to_amass = torch.tensor([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=torch.float32, device=device) 
    smpl_params = apply_inverse_rotation(smpl_params, R_motionx_to_amass, smplx_root)

    if JOINT_NUM == 24: # use smpl model
        kwargs = {'model_path': SMPL_PATH, 
                'gender': 'neutral',
                'num_betas': 10,
                'batch_size': seq_len}
        smpl_params['global_orient'] = smpl_params['global_orient'].reshape(seq_len, -1, 3, 3)

        smpl_model = smplx.SMPLLayer(**kwargs).to(device)
        model_output = smpl_model(**smpl_params)
        verts_seq = model_output.vertices.detach().cpu().numpy()
        faces_data = smpl_model.faces
    
    elif JOINT_NUM == 22: # use smplx model
        kwargs = {'model_path': SMPLX_PATH, 
                'gender': 'neutral',
                'num_betas': 10,
                'batch_size': seq_len}
        
        assert smpl_params['betas'] == None, "smplx model do not use now shape"
        smpl_model = smplx.SMPLX(**kwargs).to(device)
        model_output = smpl_model(**smpl_params)
        verts_seq = model_output.vertices.detach().cpu().numpy()
        faces_data = smpl_model.faces


    R_view_seq, T_view_seq = get_R_T(joints, zero_trans=zero_trans)
    focal = estimate_focal_length(W, H)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.basename(output_video_path)
        final_video_path = os.path.join(output_dir, fname)
    else:
        final_video_path = output_video_path

    os.makedirs(os.path.dirname(final_video_path), exist_ok=True)
    
    if verbose: print(f"Rendering video to: {final_video_path}")
    
    renderer = Renderer(focal_length=focal, img_w=W, img_h=H, faces=faces_data)
    video_renderer = VideoRenderer(renderer, final_video_path, fps=fps)
    
    R_view_np = R_view_seq.detach().cpu().numpy()
    T_view_np = T_view_seq.detach().cpu().numpy()
    
    video_renderer.render_video(
        verts_sequence=verts_seq,
        R_sequence=R_view_np,
        T_sequence=T_view_np,
        desc=motion_name if motion_name else "rendering"
    )
    renderer.delete()

    if verbose: 
        print(f"Visualization saved at: {final_video_path}")
        
    return smpl_params, joints
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize a single motion file using Pyrender.')
    parser.add_argument('--motion_file', type=str, required=True, help='Path to the .pt motion file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for the rendered video')
    parser.add_argument('--H', type=int, default=512, help='Image height')
    parser.add_argument('--W', type=int, default=512, help='Image width')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for the output video')
    parser.add_argument('--zero_trans', action='store_true', help='Force using zero translation when estimating camera')
    parser.add_argument('--use_shape', action='store_true', help='Use shape (betas) from motion data')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output video')
    
    args = parser.parse_args()
    
    torch.set_grad_enabled(False)
    visualize_single_motion(args)

