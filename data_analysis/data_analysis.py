import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt


# ==================== CONFIGURATION ====================
DATASET_PATH = 'vehicle_data'
TRAIN_LABELS_JSON = os.path.join(DATASET_PATH, 'labels/bdd100k_labels_images_train.json')
VAL_LABELS_JSON = os.path.join(DATASET_PATH, 'labels/bdd100k_labels_images_val.json')
IMAGES_PATH_TRAIN = os.path.join(DATASET_PATH, 'images/train')
IMAGES_PATH_VAL = os.path.join(DATASET_PATH, 'images/val')

CLASSES = [
    "person", "rider", "car", "bus", "truck",
    "bike", "motor", "traffic light", "traffic sign", "train"
]


# ==================== DATA LOADING AND PARSING for JSON annotations ====================
def load_json_annotations(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    parsed = {'annotations': [], 'images': []}
    for item in data:
        image_id = item.get('name') or item.get('image_id')
        if not image_id:
            continue
        parsed['images'].append(image_id)
        objects = []
        for label in item.get('labels', []):
            cls = label.get('category')
            box2d = label.get('box2d')
            if cls in CLASSES and box2d:
                xmin = int(box2d['x1'])
                ymin = int(box2d['y1'])
                xmax = int(box2d['x2'])
                ymax = int(box2d['y2'])
                objects.append({'class': cls, 'bbox': (xmin, ymin, xmax, ymax)})
        parsed['annotations'].append({'image': image_id, 'objects': objects})
    return parsed


# ==================== QUALITY CONTROL AND VALIDATION ====================
def validate_annotations(data, images_path):
    print("Performing annotation validation...")
    for annotation in data['annotations']:
        image_name = annotation['image']
        img = cv2.imread(os.path.join(images_path, image_name))
        if img is None:
            print(f"Warning: Image not found or corrupted {image_name}")
            continue

        h, w, _ = img.shape
        for obj in annotation['objects']:
            xmin, ymin, xmax, ymax = obj['bbox']
            if not (0 <= xmin < xmax <= w and 0 <= ymin < ymax <= h):
                print(f"Invalid bbox in {image_name} for class {obj['class']}: "
                      f"({xmin}, {ymin}, {xmax}, {ymax}) outside bounds ({w}, {h}).")
            bbox_w = xmax - xmin
            bbox_h = ymax - ymin
            if bbox_w < 5 or bbox_h < 5:
                print(f"Tiny bbox in {image_name} for class {obj['class']}: "
                      f"({bbox_w}, {bbox_h})")


# ==================== DATA DISTRIBUTION ANALYSIS ====================
def plot_class_distribution(data, split_name):
    class_counts = {cls: 0 for cls in CLASSES}
    for annotation in data['annotations']:
        for obj in annotation['objects']:
            if obj['class'] in class_counts:
                class_counts[obj['class']] += 1

    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.title(f'Object Class Distribution - {split_name}')
    plt.xlabel('Class')
    plt.ylabel('Number of Objects')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'class_distribution_{split_name}.png')
    plt.close()


def plot_bbox_size_distribution(data, split_name):
    all_areas = []
    for annotation in data['annotations']:
        for obj in annotation['objects']:
            xmin, ymin, xmax, ymax = obj['bbox']
            area = (xmax - xmin) * (ymax - ymin)
            all_areas.append(area)

    plt.figure(figsize=(10, 6))
    plt.hist(all_areas, bins=50, color='lightgreen', edgecolor='black')
    plt.title(f'Bounding Box Area Distribution - {split_name}')
    plt.xlabel('Area (pixels)')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'size_distribution_{split_name}.png')
    plt.close()


def plot_image_properties(data, images_path, split_name):
    widths, heights = [], []
    for image_name in data['images']:
        img = cv2.imread(os.path.join(images_path, image_name))
        if img is not None:
            h, w, _ = img.shape
            heights.append(h)
            widths.append(w)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=30, color='coral', edgecolor='black')
    plt.title(f'Image Width Distribution - {split_name}')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=30, color='teal', edgecolor='black')
    plt.title(f'Image Height Distribution - {split_name}')
    plt.xlabel('Height (pixels)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f'properties_{split_name}.png')
    plt.close()


# ==================== EDGE CASE VISUALIZATION ====================
def visualize_difficult_cases(data, images_path, split_name, num_examples=5, min_bbox_area=50, max_bbox_area=500):
    print(f"Visualizing {num_examples} difficult cases - {split_name}...")
    difficult_images = []
    for annotation in data['annotations']:
        for obj in annotation['objects']:
            xmin, ymin, xmax, ymax = obj['bbox']
            area = (xmax - xmin) * (ymax - ymin)
            if min_bbox_area <= area <= max_bbox_area:
                difficult_images.append(annotation['image'])
                break

    if not difficult_images:
        print(f"No difficult cases found for {split_name} dataset.")
        return

    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5 * num_examples))
    sample_images = np.random.choice(difficult_images, min(num_examples, len(difficult_images)), replace=False)

    for i, image_name in enumerate(sample_images):
        img_path = os.path.join(images_path, image_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for annotation in data['annotations']:
            if annotation['image'] == image_name:
                for obj in annotation['objects']:
                    xmin, ymin, xmax, ymax = obj['bbox']
                    color = (0, 255, 0)
                    cv2.rectangle(img_rgb, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(img_rgb, obj['class'], (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        plt.subplot(num_examples, 1, i + 1)
        plt.imshow(img_rgb)
        plt.title(f"Difficult Case: {image_name} ({split_name})")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'difficult_cases_{split_name}.png')
    plt.show()


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    print("=== TRAIN SPLIT ANALYSIS ===")
    train_data = load_json_annotations(TRAIN_LABELS_JSON)
    validate_annotations(train_data, IMAGES_PATH_TRAIN)
    plot_class_distribution(train_data, 'train')
    plot_bbox_size_distribution(train_data, 'train')
    plot_image_properties(train_data, IMAGES_PATH_TRAIN, 'train')
    visualize_difficult_cases(train_data, IMAGES_PATH_TRAIN, 'train')

    print("=== VALIDATION SPLIT ANALYSIS ===")
    val_data = load_json_annotations(VAL_LABELS_JSON)
    validate_annotations(val_data, IMAGES_PATH_VAL)
    plot_class_distribution(val_data, 'val')
    plot_bbox_size_distribution(val_data, 'val')
    plot_image_properties(val_data, IMAGES_PATH_VAL, 'val')
    visualize_difficult_cases(val_data, IMAGES_PATH_VAL, 'val')
