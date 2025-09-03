import os
import cv2
import random
import numpy as np

# ==================== CONFIGURATION ====================
# Update these paths to match your dataset structure
DATASET_PATH = 'vehicle'
IMAGES_PATH = os.path.join(DATASET_PATH, 'images/train')
ANNOTATIONS_PATH = os.path.join(DATASET_PATH, 'labels/train')
VALID_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']
CLASSES = [
    "person", "rider", "car", "bus", "truck",
    "bike", "motor", "traffic light", "traffic sign", "train"
]
# ==================== HELPER FUNCTIONS ====================
def plot_one_box(image, bbox, class_name, color=(0, 255, 0), line_thickness=2):
    """
    Draws a single bounding box and its label on an image.
    """
    xmin, ymin, xmax, ymax = map(int, bbox)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, line_thickness)
    cv2.putText(image, class_name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, line_thickness)
    return image

def create_image_grid(images, grid_dims, thumb_size=(640, 480)):
    """
    Creates a grid of images.
    """
    rows, cols = grid_dims
    thumb_w, thumb_h = thumb_size
    grid_img = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        if img is None: continue
        img_resized = cv2.resize(img, thumb_size, interpolation=cv2.INTER_AREA)
        row_idx, col_idx = i // cols, i % cols
        y_offset, x_offset = row_idx * thumb_h, col_idx * thumb_w
        grid_img[y_offset:y_offset + thumb_h, x_offset:x_offset + thumb_w] = img_resized
    return grid_img

# ==================== MAIN VISUALIZATION FUNCTION ====================
def generate_multiple_grids(num_grids_to_save=1, images_per_grid=20, grid_layout=(5, 4)):
    """
    Generates and saves a specified number of grid images, each containing
    a unique set of randomly selected and annotated images.
    """
    print(f"Preparing to create {num_grids_to_save} grid(s), each with {images_per_grid} images.")

    output_dir = 'output/annotated_grids'
    os.makedirs(output_dir, exist_ok=True)

    try:
        all_annotation_files = [f for f in os.listdir(ANNOTATIONS_PATH) if f.endswith('.txt')]
        if not all_annotation_files:
            print(f"Error: No annotation files found in '{ANNOTATIONS_PATH}'.")
            return
    except FileNotFoundError:
        print(f"Error: Annotation directory not found at '{ANNOTATIONS_PATH}'.")
        return

    # Create each grid one by one
    for i in range(num_grids_to_save):
        print(f"\n--- Generating Grid {i + 1}/{num_grids_to_save} ---")

        if len(all_annotation_files) < images_per_grid:
            print(f"Warning: Not enough unique images left to create a full grid of {images_per_grid}.")
            print(f"Only {len(all_annotation_files)} images remaining. Creating a partial grid.")
            if not all_annotation_files:
                print("No more images to process. Stopping.")
                break
            images_to_process = len(all_annotation_files)
        else:
            images_to_process = images_per_grid

        # Select a random sample and remove them from the main list to ensure uniqueness
        selected_files = random.sample(all_annotation_files, images_to_process)
        all_annotation_files = [f for f in all_annotation_files if f not in selected_files]

        annotated_images_for_grid = []
        for anno_file in selected_files:
            base_name = os.path.splitext(anno_file)[0]
            image_path = None
            for ext in VALID_IMAGE_EXTENSIONS:
                potential_path = os.path.join(IMAGES_PATH, base_name + ext)
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
            
            if not image_path: continue
            
            image = cv2.imread(image_path)
            if image is None: continue

            h, w, _ = image.shape
            with open(os.path.join(ANNOTATIONS_PATH, anno_file), 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5: continue
                
                class_id = int(parts[0])
                x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])
                
                xmin = int((x_center - bbox_width / 2) * w)
                ymin = int((y_center - bbox_height / 2) * h)
                xmax = int((x_center + bbox_width / 2) * w)
                ymax = int((y_center + bbox_height / 2) * h)

                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(w, xmax), min(h, ymax)

                class_name = CLASSES[class_id] if 0 <= class_id < len(CLASSES) else "Unknown"
                image = plot_one_box(image, (xmin, ymin, xmax, ymax), class_name)
            
            annotated_images_for_grid.append(image)

        if not annotated_images_for_grid:
            print("No images were processed successfully for this grid. Skipping.")
            continue

        # Create and save the grid image with a unique name
        image_grid = create_image_grid(annotated_images_for_grid, grid_dims=grid_layout)
        output_grid_path = os.path.join(output_dir, f'annotated_grid_{i + 1}.jpg')
        cv2.imwrite(output_grid_path, image_grid)
        print(f"Successfully saved grid {i + 1} to: {output_grid_path}")

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # Change 'num_grids_to_save' to the number of output images you want.
    # Each output image will contain 20 concatenated images in a 5x4 grid.
    generate_multiple_grids(
        num_grids_to_save=10, 
        images_per_grid=20, 
        grid_layout=(5, 4)
    )

