import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from sklearn.preprocessing import normalize


class FeatureExtractor:
    def __init__(self, use_deep_features=True):
        self.use_deep_features = use_deep_features
        
        if use_deep_features:
            self.init_deep_extractor()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),  # Standard person re-id size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def init_deep_extractor(self):
        # Use ResNet18 as backbone
        self.backbone = resnet18(pretrained=True)
        
        # Remove classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Set to evaluation mode
        self.backbone.eval()
        
        # Freeze parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.backbone.to(self.device)
        
    def extract_features(self, image_crop):
        if image_crop is None or image_crop.size == 0:
            return np.zeros(512)  # Return zero vector for invalid crops
            
        # Resize crop to minimum size if too small
        h, w = image_crop.shape[:2]
        if h < 32 or w < 16:
            image_crop = cv2.resize(image_crop, (32, 64))
            
        if self.use_deep_features:
            deep_features = self.extract_deep_features(image_crop)
            color_features = self.extract_color_features(image_crop)
            texture_features = self.extract_texture_features(image_crop)
            
            # Combine all features
            combined_features = np.concatenate([
                deep_features,
                color_features,
                texture_features
            ])
        else:
            # Use only handcrafted features
            color_features = self.extract_color_features(image_crop)
            texture_features = self.extract_texture_features(image_crop)
            spatial_features = self.extract_spatial_features(image_crop)
            
            combined_features = np.concatenate([
                color_features,
                texture_features,
                spatial_features
            ])
        
        # Normalize features
        combined_features = normalize([combined_features])[0]
        
        return combined_features
    
    def extract_deep_features(self, image_crop):
        try:
            # Convert BGR to RGB
            if len(image_crop.shape) == 3:
                image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_GRAY2RGB)
            
            # Preprocess image
            input_tensor = self.transform(image_rgb).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.backbone(input_tensor)
                features = features.squeeze().cpu().numpy()
            
            return features
            
        except Exception as e:
            print(f"Error in deep feature extraction: {e}")
            return np.zeros(512)
    
    def extract_color_features(self, image_crop):
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms for each channel
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256])
        
        # Normalize histograms
        h_hist = h_hist.flatten() / (h_hist.sum() + 1e-6)
        s_hist = s_hist.flatten() / (s_hist.sum() + 1e-6)
        v_hist = v_hist.flatten() / (v_hist.sum() + 1e-6)
        
        # Dominant colors in each region (upper/lower body)
        height = image_crop.shape[0]
        upper_half = image_crop[:height//2, :]
        lower_half = image_crop[height//2:, :]
        
        upper_color = self.get_dominant_color(upper_half)
        lower_color = self.get_dominant_color(lower_half)
        
        color_features = np.concatenate([
            h_hist, s_hist, v_hist,
            upper_color, lower_color
        ])
        
        return color_features
    
    def extract_texture_features(self, image_crop):
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        
        # Local Binary Pattern
        lbp = self.calculate_lbp(gray)
        lbp_hist = cv2.calcHist([lbp], [0], None, [16], [0, 256])
        lbp_hist = lbp_hist.flatten() / (lbp_hist.sum() + 1e-6)
        
        # Gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        grad_hist = cv2.calcHist([grad_mag.astype(np.uint8)], [0], None, [16], [0, 256])
        grad_hist = grad_hist.flatten() / (grad_hist.sum() + 1e-6)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        texture_features = np.concatenate([
            lbp_hist,
            grad_hist,
            [edge_density]
        ])
        
        return texture_features
    
    def extract_spatial_features(self, image_crop):
        height, width = image_crop.shape[:2]
        
        # Aspect ratio
        aspect_ratio = width / height
        
        # Body part ratios (assuming person detection)
        # Upper body (head + torso) vs lower body (legs)
        upper_third = image_crop[:height//3, :]
        middle_third = image_crop[height//3:2*height//3, :]
        lower_third = image_crop[2*height//3:, :]
        
        # Color variance in each region
        upper_var = np.var(cv2.cvtColor(upper_third, cv2.COLOR_BGR2GRAY))
        middle_var = np.var(cv2.cvtColor(middle_third, cv2.COLOR_BGR2GRAY))
        lower_var = np.var(cv2.cvtColor(lower_third, cv2.COLOR_BGR2GRAY))
        
        # Symmetry measure
        left_half = image_crop[:, :width//2]
        right_half = cv2.flip(image_crop[:, width//2:], 1)
        
        # Resize to match if different sizes
        if left_half.shape != right_half.shape:
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
        
        symmetry = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
        if np.isnan(symmetry):
            symmetry = 0
        
        spatial_features = np.array([
            aspect_ratio,
            upper_var / 1000.0,  # Normalize
            middle_var / 1000.0,
            lower_var / 1000.0,
            symmetry
        ])
        
        return spatial_features
    
    def get_dominant_color(self, image_region):
        if image_region.size == 0:
            return np.array([0, 0, 0])
        
        # Reshape image to list of pixels
        pixels = image_region.reshape(-1, 3)
        
        # Use k-means to find dominant color
        from sklearn.cluster import KMeans
        
        try:
            kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
            kmeans.fit(pixels)
            dominant_color = kmeans.cluster_centers_[0]
        except:
            # Fallback to mean color
            dominant_color = np.mean(pixels, axis=0)
        
        return dominant_color / 255.0  # Normalize to [0, 1]
    
    def calculate_lbp(self, gray_image, radius=1, n_points=8):
        def get_pixel(img, center, x, y):
            new_value = 0
            try:
                if img[x][y] >= center:
                    new_value = 1
            except:
                pass
            return new_value
        
        lbp = np.zeros_like(gray_image, dtype=np.uint8)
        
        for i in range(radius, gray_image.shape[0] - radius):
            for j in range(radius, gray_image.shape[1] - radius):
                center = gray_image[i, j]
                val = 0
                
                # 8-point LBP
                val += get_pixel(gray_image, center, i-1, j-1) * 1
                val += get_pixel(gray_image, center, i-1, j) * 2
                val += get_pixel(gray_image, center, i-1, j+1) * 4
                val += get_pixel(gray_image, center, i, j+1) * 8
                val += get_pixel(gray_image, center, i+1, j+1) * 16
                val += get_pixel(gray_image, center, i+1, j) * 32
                val += get_pixel(gray_image, center, i+1, j-1) * 64
                val += get_pixel(gray_image, center, i, j-1) * 128
                
                lbp[i, j] = val
        
        return lbp


class SimpleFeatureExtractor:
    
    def __init__(self):
        pass
    
    def extract_features(self, image_crop):
        if image_crop is None or image_crop.size == 0:
            return np.zeros(64)
        
        # Resize to standard size
        image_crop = cv2.resize(image_crop, (32, 64))
        
        # Color histogram in HSV
        hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [8], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256])
        
        # Normalize
        h_hist = h_hist.flatten() / (h_hist.sum() + 1e-6)
        s_hist = s_hist.flatten() / (s_hist.sum() + 1e-6)
        v_hist = v_hist.flatten() / (v_hist.sum() + 1e-6)
        
        # Simple texture - edge density
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Dominant colors
        upper_half = image_crop[:32, :]
        lower_half = image_crop[32:, :]
        
        upper_mean = np.mean(upper_half.reshape(-1, 3), axis=0) / 255.0
        lower_mean = np.mean(lower_half.reshape(-1, 3), axis=0) / 255.0
        
        # Combine features
        features = np.concatenate([
            h_hist, s_hist, v_hist,
            [edge_density],
            upper_mean,
            lower_mean
        ])
        
        # Normalize
        features = normalize([features])[0]
        
        return features