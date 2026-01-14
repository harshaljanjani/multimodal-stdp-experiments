import cupy as cp
import numpy as np

class ObjectTracker:
    def __init__(self, color_tolerance=0.3, min_pixel_threshold=50, max_distance_merge=0.15):
        self.objects = {}
        self.next_id = 0
        self.color_tolerance = color_tolerance
        self.min_pixel_threshold = min_pixel_threshold
        self.max_distance_merge = max_distance_merge
        self.known_colors = {
            "red": cp.array([1.0, 0.0, 0.0], dtype=cp.float32),
            "blue": cp.array([0.0, 0.0, 1.0], dtype=cp.float32),
            "green": cp.array([0.0, 1.0, 0.0], dtype=cp.float32),
            "yellow": cp.array([1.0, 1.0, 0.0], dtype=cp.float32)
        }
        print("[COGNITION] ObjectTracker initialized (Color-Based Segmentation)")

    def _detect_blobs(self, img_gpu, target_color):
        # simple color-based blob detection
        if img_gpu is None or img_gpu.shape[0] == 0:
            return None
        distances = cp.linalg.norm(img_gpu - target_color, axis=2)
        mask = distances < self.color_tolerance
        pixel_count = cp.sum(mask).item()
        if pixel_count < self.min_pixel_threshold:
            return None
        y_coords, x_coords = cp.where(mask)
        if len(y_coords) == 0:
            return None
        centroid_y = float(cp.mean(y_coords).item())
        centroid_x = float(cp.mean(x_coords).item())
        height, width = img_gpu.shape[:2]
        normalized_pos = np.array([centroid_x / width, centroid_y / height])
        return {"position": normalized_pos, "pixel_count": pixel_count}

    def _find_matching_object(self, new_position, color_name):
        # find existing object within merge distance
        for obj_id, obj_data in self.objects.items():
            if obj_data["color"] == color_name:
                dist = np.linalg.norm(obj_data["position"] - new_position)
                if dist < self.max_distance_merge:
                    return obj_id
        return None

    def update_from_vision(self, img_gpu, current_time_ms):
        # detect all color blobs and update registry
        detected_this_frame = set()
        for color_name, color_value in self.known_colors.items():
            blob = self._detect_blobs(img_gpu, color_value)
            if blob is None:
                continue
            matching_id = self._find_matching_object(blob["position"], color_name)
            if matching_id is not None:
                # update existing object
                obj = self.objects[matching_id]
                dt = (current_time_ms - obj["last_seen"]) / 1000.0
                if dt > 0:
                    velocity = (blob["position"] - obj["position"]) / dt
                    obj["velocity"] = 0.7 * obj["velocity"] + 0.3 * velocity
                obj["position"] = blob["position"]
                obj["pixel_count"] = blob["pixel_count"]
                obj["last_seen"] = current_time_ms
                detected_this_frame.add(matching_id)
            else:
                # create new object
                new_obj = {
                    "position": blob["position"],
                    "velocity": np.zeros(2),
                    "color": color_name,
                    "pixel_count": blob["pixel_count"],
                    "last_seen": current_time_ms,
                    "created_at": current_time_ms
                }
                self.objects[self.next_id] = new_obj
                detected_this_frame.add(self.next_id)
                self.next_id += 1
        # remove stale objects (not seen for 5 seconds)
        stale_threshold = 5000
        to_remove = []
        for obj_id, obj_data in self.objects.items():
            if (current_time_ms - obj_data["last_seen"]) > stale_threshold:
                to_remove.append(obj_id)
        for obj_id in to_remove:
            del self.objects[obj_id]

    def get_all_objects(self):
        return self.objects

    def get_object(self, obj_id):
        return self.objects.get(obj_id)

    def get_objects_by_color(self, color_name):
        return {k: v for k, v in self.objects.items() if v["color"] == color_name}

    def print_status(self):
        if not self.objects:
            print("[OBJECT_TRACKER] No objects detected")
            return
        print(f"[OBJECT_TRACKER] Tracking {len(self.objects)} objects:")
        for obj_id, obj_data in self.objects.items():
            vel_mag = np.linalg.norm(obj_data["velocity"])
            print(f"  id={obj_id} | color={obj_data['color']} | pos={obj_data['position']} | vel={vel_mag:.3f}")
