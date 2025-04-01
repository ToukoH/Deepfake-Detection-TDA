import numpy as np
from PIL import Image

class LBPExtractor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = self._load_image_as_array(image_path)
        self.lbp_image = None
    
    def _load_image_as_array(self, path):
        with Image.open(path) as img:
            gray_img = img.convert("L")
            return np.array(gray_img, dtype=np.uint8)
    
    def compute_lbp(self):
        height, width = self.image.shape

        neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1), (0, 1),
            (1, 1), (1, 0), (1, -1), (0, -1)
        ]
        
        lbp_result = np.zeros((height, width), dtype=np.uint8)
        for r in range(1, height-1):
            for c in range(1, width-1):
                center_val = self.image[r, c]
                code = 0
                for i, (dr, dc) in enumerate(neighbor_offsets):
                    neighbor_val = self.image[r + dr, c + dc]
                    bit = 1 if neighbor_val >= center_val else 0
                    code |= (bit << (7 - i))
                lbp_result[r, c] = code
        
        self.lbp_image = lbp_result
        return lbp_result
    
    def get_uniform_mask(self, max_transitions=2):
        if self.lbp_image is None:
            raise ValueError("call compute_lbp() first")
        
        def count_transitions(code):
            binary_str = f"{code:08b}"
            transitions = 0
            for i in range(8):
                if binary_str[i] != binary_str[(i+1) % 8]:
                    transitions += 1
            return transitions
        
        uniform_mask = np.zeros_like(self.lbp_image, dtype=np.uint8)
        for r in range(1, self.lbp_image.shape[0]-1):
            for c in range(1, self.lbp_image.shape[1]-1):
                code = self.lbp_image[r, c]
                if count_transitions(code) <= max_transitions:
                    uniform_mask[r, c] = 1
        
        return uniform_mask
    
    def extract_point_cloud(self, target_codes):
        if self.lbp_image is None:
            raise ValueError("call compute_lbp() first")
        
        points = []
        rows, cols = np.where(np.isin(self.lbp_image, target_codes))
        for r, c in zip(rows, cols):
            points.append((r, c))
        return points
    
    def show_info(self):
        print(f"Image shape: {self.image.shape}")
        if self.lbp_image is not None:
            print("LBP computed")
        else:
            print("LBP not computed")
