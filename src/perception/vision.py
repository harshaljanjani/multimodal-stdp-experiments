import cupy as cp
import numpy as np
import cv2
from pathlib import Path
import omni.usd
from pxr import UsdGeom, Gf

# cover for all versions.
try:
    from isaacsim.core.api.utils.camera import Camera
except ImportError:
    try:
        from omni.isaac.core.utils.camera import Camera
    except ImportError:
        from omni.isaac.sensor import Camera

class VisionSystem:
    def __init__(self, camera_prim_path, attachment_prim_path, resolution=(128, 128), offset_position=None):
        self.camera_prim_path = camera_prim_path
        self.attachment_prim_path = attachment_prim_path
        self.resolution = resolution
        self.offset_position = offset_position or [0.1, 0.0, 0.05]
        self.camera = None
        self.debug_frame_count = 0
        self.debug_dir = Path("_debug_vision")
        self.debug_dir.mkdir(exist_ok=True)
        print(f"[DEBUG] Vision debug frames will be saved to: {self.debug_dir.resolve()}")

    def _save_debug_frame(self, rgba_data):
        if self.debug_frame_count % 50 == 0:
            img_bgr = cv2.cvtColor(rgba_data, cv2.COLOR_RGBA2BGRA)
            filepath = str(self.debug_dir / f"frame_{self.debug_frame_count:04d}.png")
            cv2.imwrite(filepath, img_bgr)
        self.debug_frame_count += 1

    def _ensure_camera_prim_exists(self):
        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(self.camera_prim_path)
        is_camera = camera_prim.IsValid() and camera_prim.IsA(UsdGeom.Camera)
        if not is_camera:
            if camera_prim.IsValid():
                print(f"[VISION] Prim exists at {self.camera_prim_path} but is not a Camera type, recreating...")
                stage.RemovePrim(self.camera_prim_path)
            else:
                print(f"[VISION] Camera prim not found at {self.camera_prim_path}, creating...")
            camera = UsdGeom.Camera.Define(stage, self.camera_prim_path)
            xform = camera.AddTranslateOp()
            xform.Set(Gf.Vec3d(0.1, 0, 0.05))
            print(f"[VISION] Camera prim created at {self.camera_prim_path}")
        else:
            print(f"[VISION] Valid Camera prim found at {self.camera_prim_path}")

    def create_mono_camera(self, camera_prim_path):
        self._ensure_camera_prim_exists()
        # https://math.stackexchange.com/questions/1499415/finding-the-quaternion-that-performs-a-rotation    
        width, height = 1280, 800
        camera = Camera(
            prim_path=camera_prim_path,
            resolution=(width, height),
        )
        camera.initialize()
        # using coefficients from OAK-D Pro W Camera
        camera_matrix = [[569.0758666992188, 0.0, 632.8007202148438],[0.0, 569.033935546875, 367.555908203125],[0.0, 0.0, 1.0]]
        # these are the coefficiens for the Rational Polynomial Model
        # k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tx, ty
        distortion_coefficients = [10.102782249450684, 3.5927071571350098, -1.2150891052442603e-05, -0.00010162794205825776, 0.07836529612541199, 10.494465827941895, 6.792006969451904, 0.5550446510314941, 0.0, 0.0, 0.0, 0.0, -0.00024047934857662767, -0.0019406505161896348]
        distortion_coefficients = distortion_coefficients[:8]
        pixel_size = 3 * 1e-3   # in microns, 3 microns is common
        f_stop = 2.0            # f-number, the ratio of the lens focal length to the diameter of the entrance pupil
        focus_distance = 0.18   # in meters, the distance from the camera to the object plane
        diagonal_fov = 150      # in degrees, the diagonal field of view to be rendered
        effective_focal_length = 1.69   # in mm
        ((fx,_,cx),(_,fy,cy),(_,_,_)) = camera_matrix
        horizontal_aperture = pixel_size * width
        vertical_aperture = pixel_size * height
        camera.set_focal_length(effective_focal_length/100.0)
        camera.set_horizontal_aperture(horizontal_aperture/100.0)
        camera.set_vertical_aperture(vertical_aperture/100.0)
        camera.set_lens_aperture(f_stop*1000.0)
        camera.set_focus_distance(focus_distance)
        camera.set_clipping_range(0.05, 1.0e5)
        camera.set_projection_type("fisheyePolynomial")
        camera.set_rational_polynomial_properties(width, height, cx, cy, diagonal_fov, distortion_coefficients)
        return camera

    def initialize(self, world):
        self.camera = self.create_mono_camera(self.camera_prim_path)
        print(f"[DEBUG] Camera initialized with OAK-D Pro W fisheye configuration")
        self._setup_camera_tracking()
    
    def _setup_camera_tracking(self):
        import omni.usd
        from pxr import UsdGeom, Gf
        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(self.camera_prim_path)
        attachment_prim = stage.GetPrimAtPath(self.attachment_prim_path)
        if camera_prim.IsValid() and attachment_prim.IsValid():
            print(f"[DEBUG] Camera tracking set up for {self.attachment_prim_path}")
    
    def update_camera_pose(self):
        import omni.usd
        from pxr import UsdGeom, Gf
        stage = omni.usd.get_context().get_stage()
        camera_xform = UsdGeom.Xformable(stage.GetPrimAtPath(self.camera_prim_path))
        attachment_xform = UsdGeom.Xformable(stage.GetPrimAtPath(self.attachment_prim_path))
        if camera_xform and attachment_xform:
            attachment_matrix = attachment_xform.ComputeLocalToWorldTransform(0)
            offset_vec = Gf.Vec3d(*self.offset_position)
            offset_matrix = Gf.Matrix4d(1.0)
            offset_matrix.SetTranslateOnly(offset_vec)
            final_matrix = offset_matrix * attachment_matrix
            camera_xform.ClearXformOpOrder()
            translate_op = camera_xform.AddTranslateOp()
            translate_op.Set(final_matrix.ExtractTranslation())

    def process_image_to_spikes(self, pop_info, pop_name):
        self.camera.get_current_frame()
        rgba_data = self.camera.get_rgba()
        if rgba_data is None or rgba_data.shape[0] == 0:
            return None
        self._save_debug_frame(np.copy(rgba_data))
        img_gpu = cp.asarray(rgba_data[..., :3])
        target_color = cp.array([1.0, 0.0, 0.0], dtype=cp.float32)
        color_threshold = 0.8
        distances = cp.linalg.norm(img_gpu - target_color, axis=2)
        detected_pixels = cp.sum(distances < color_threshold)
        detection_threshold = 20
        if detected_pixels < detection_threshold:
            return None
        if self.debug_frame_count % 25 == 0:
            print(f"[VISION] Detected {detected_pixels} red pixels!")
        pop = pop_info[pop_name]
        # fire more neurons the more red we see
        proportion_seen = min(1.0, detected_pixels.item() / (self.resolution[0] * 20.0))
        num_firing = int(pop['count'] * proportion_seen)
        if num_firing == 0:
            return None
        indices = cp.random.choice(
            cp.arange(pop['start'], pop['end']),
            size=num_firing,
            replace=False
        )
        return indices.astype(cp.int32)
