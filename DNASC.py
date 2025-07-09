import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def compute_scores(rgb_dir, output_dir, error_map_dir=None, mask_dir=None, error_weight=0.3):
    # Create output directory if it doesn’t exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of RGB files
    rgb_files = [f for f in os.listdir(rgb_dir) if f.lower().endswith('.png')]

    # Process each RGB view
    for rgb_file in rgb_files:
        # Extract view_id as the filename without the .png extension
        view_id = os.path.splitext(rgb_file)[0]

        # Load RGB image
        rgb_path = os.path.join(rgb_dir, rgb_file)
        rgb_map = cv.imread(rgb_path).astype(np.float32) / 255.0  # Shape: [H, W, 3]
        rgb_gray = cv.cvtColor(rgb_map, cv.COLOR_BGR2GRAY)  # Shape: [H, W]

        # Check if RGB map loaded correctly
        if rgb_map is None:
            raise ValueError(f"Failed to load RGB image: {rgb_path}")

        # Load mask if provided
        if mask_dir and os.path.exists(os.path.join(mask_dir, f"{view_id}.png")):
            mask_path = os.path.join(mask_dir, f"{view_id}.png")
            mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            if mask is None:
                raise ValueError(f"Failed to load mask image: {mask_path}")
            mask = (mask > 0.5).astype(np.float32)  # Binary mask: 1 for foreground, 0 for background
        else:
            mask = np.ones_like(rgb_gray, dtype=np.float32)

        # Apply minimal Gaussian blur to preserve fine details
        rgb_gray_smooth = gaussian_filter(rgb_gray, sigma=0.3)

        # Edge detection using Canny to highlight fine structures
        edges = cv.Canny((rgb_gray_smooth * 255).astype(np.uint8), 100, 200) / 255.0

        # Compute intensity variance at multiple scales
        kernel_size_small = 3
        kernel_small = np.ones((kernel_size_small, kernel_size_small), dtype=np.float32)
        mean_small = cv.filter2D(rgb_gray_smooth, -1, kernel_small / (kernel_size_small * kernel_size_small))
        squared_small = rgb_gray_smooth**2
        mean_squared_small = cv.filter2D(squared_small, -1, kernel_small / (kernel_size_small * kernel_size_small))
        variance_small = mean_squared_small - mean_small**2

        kernel_size_large = 7
        kernel_large = np.ones((kernel_size_large, kernel_size_large), dtype=np.float32)
        mean_large = cv.filter2D(rgb_gray_smooth, -1, kernel_large / (kernel_size_large * kernel_size_large))
        mean_squared_large = cv.filter2D(squared_small, -1, kernel_large / (kernel_size_large * kernel_size_large))
        variance_large = mean_squared_large - mean_large**2

        # Combine variances from both scales
        variance = 0.6 * variance_small + 0.4 * variance_large

        # Apply logarithmic transformation
        variance = np.log1p(variance * 2000)

        # Normalize variance
        variance_norm = variance / (np.max(variance) + 1e-8)

        # Apply histogram equalization
        variance_norm = cv.equalizeHist((variance_norm * 255).astype(np.uint8)).astype(np.float32) / 255.0

        # Combine variance with edge map and apply mask
        S_i = (0.95 * variance_norm + 0.05 * edges) * mask

        # Incorporate error map if provided
        if error_map_dir and os.path.exists(os.path.join(error_map_dir, f"{view_id}.npy")):
            error_map = np.load(os.path.join(error_map_dir, f"{view_id}.npy"))
            error_map = error_map / (np.max(error_map) + 1e-8)  # Normalize
            # Resize error_map to match S_i dimensions
            H, W = S_i.shape
            error_map = cv.resize(error_map, (W, H), interpolation=cv.INTER_LINEAR)
            # Apply mask to error_map
            error_map = error_map * mask
            S_i = (1 - error_weight) * S_i + error_weight * error_map

        # Normalize the combined score
        S_i = S_i / (np.max(S_i) + 1e-8)

        # Apply threshold
        S_i = np.where(S_i > 0.001, S_i, 0)

        # Debug: Log distribution
        print(f"View {view_id} - S_i Min: {S_i.min():.4f}, Max: {S_i.max():.4f}, Mean: {S_i.mean():.4f}, Std: {S_i.std():.4f}")
        print(f"View {view_id} - Fraction of non-zero scores: {np.sum(S_i > 0) / S_i.size:.4f}")

        # Save score
        np.save(os.path.join(output_dir, f"{view_id}.npy"), S_i)

        # Visualization - Save RGB image
        plt.figure(figsize=(10, 10), dpi=300)
        plt.imshow(rgb_map)
        plt.title(f"RGB Image (View {view_id})")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"rgb_view_{view_id}.png"), bbox_inches='tight')
        plt.close()

        # Visualization - Save Thin Structure Score image
        plt.figure(figsize=(10, 10), dpi=300)
        plt.imshow(S_i, cmap='hot')
        plt.title(f"Thin Structure Score (View {view_id})")
        plt.colorbar(label='Score Intensity')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"score_view_{view_id}.png"), bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    rgb_dir = ""
    output_dir = ""
    mask_dir = ""
    compute_scores(rgb_dir, output_dir, mask_dir=mask_dir)