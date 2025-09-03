#!/usr/bin/env python3
"""
BDD100K to COCO Format Converter

This script converts BDD100K JSON annotation format to COCO format
for object detection tasks. Supports the 10 detection classes specified
in the Bosch Applied CV Assignment.

Author: Assignment Implementation
Usage: python convert_bdd_to_coco.py --input <bdd_json> --output <coco_json>
"""

import json
import os
import argparse
import logging
from typing import Dict, List, Any


def setup_logging() -> None:
    """Configure logging for the conversion process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def get_category_mapping() -> Dict[str, int]:
    """
    Define the mapping from BDD100K category names to COCO category IDs.
    
    Returns:
        Dict[str, int]: Mapping from category name to category ID
    """
    return {
        "person": 1,
        "rider": 2, 
        "car": 3,
        "bus": 4,
        "truck": 5,
        "bike": 6,
        "motor": 7,
        "traffic light": 8,
        "traffic sign": 9,
        "train": 10
    }


def create_coco_categories(category_mapping: Dict[str, int]) -> List[Dict[str, Any]]:
    """
    Create COCO format category entries.
    
    Args:
        category_mapping: Dictionary mapping category names to IDs
        
    Returns:
        List[Dict[str, Any]]: COCO format category list
    """
    categories = []
    for cat_name, cat_id in category_mapping.items():
        categories.append({
            "id": cat_id,
            "name": cat_name,
            "supercategory": "object"
        })
    return categories


def validate_bounding_box(box2d: Dict[str, float], img_width: int, img_height: int) -> bool:
    """
    Validate bounding box coordinates.
    
    Args:
        box2d: BDD100K bounding box dictionary with x1, y1, x2, y2
        img_width: Image width
        img_height: Image height
        
    Returns:
        bool: True if bounding box is valid, False otherwise
    """
    x1, y1, x2, y2 = box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']
    
    # Check if coordinates are within image bounds
    if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
        return False
        
    # Check if box has positive area
    if x2 <= x1 or y2 <= y1:
        return False
        
    return True


def convert_bdd_to_coco(input_json_path: str, output_json_path: str) -> None:
    """
    Convert BDD100K JSON annotations to COCO format.
    
    Args:
        input_json_path: Path to BDD100K JSON file
        output_json_path: Path where COCO JSON will be saved
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading BDD100K data from {input_json_path}")
    
    # Load BDD100K data
    try:
        with open(input_json_path, 'r') as f:
            bdd_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_json_path}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
        return
    
    logger.info(f"Loaded {len(bdd_data)} images from BDD100K dataset")
    
    # Initialize COCO format structure
    category_mapping = get_category_mapping()
    coco_output = {
        "info": {
            "description": "BDD100K Dataset converted to COCO format",
            "version": "1.0",
            "year": 2024,
            "contributor": "Bosch Applied CV Assignment",
            "date_created": "2024"
        },
        "images": [],
        "annotations": [],
        "categories": create_coco_categories(category_mapping)
    }
    
    annotation_id = 1
    image_id = 1
    image_id_map = {}
    
    valid_annotations = 0
    invalid_annotations = 0
    
    # Process each image in BDD100K data
    for item in bdd_data:
        img_name = item.get('name')
        if img_name is None:
            logger.warning("Skipping item with no image name")
            continue
        
        # Get image dimensions (BDD100K uses different field names)
        width = item.get('videoW') or item.get('width', 1280)
        height = item.get('videoH') or item.get('height', 720)
        
        # Add image entry to COCO format
        coco_output["images"].append({
            "id": image_id,
            "file_name": img_name,
            "width": int(width),
            "height": int(height)
        })
        
        image_id_map[img_name] = image_id
        labels = item.get('labels', [])
        
        # Process annotations for this image
        for label in labels:
            cat_name = label.get('category')
            if cat_name not in category_mapping:
                continue  # Skip categories not in our 10-class list
            
            cat_id = category_mapping[cat_name]
            box2d = label.get('box2d')
            
            if box2d is None:
                continue
            
            # Validate bounding box
            if not validate_bounding_box(box2d, width, height):
                invalid_annotations += 1
                logger.warning(f"Invalid bounding box in {img_name}: {box2d}")
                continue
            
            # Convert BDD100K box format (x1,y1,x2,y2) to COCO format (x,y,w,h)
            x1, y1, x2, y2 = box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']
            width_box = x2 - x1
            height_box = y2 - y1
            
            # COCO bounding box format: [x, y, width, height]
            bbox = [x1, y1, width_box, height_box]
            area = width_box * height_box
            
            # Create COCO annotation
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": bbox,
                "area": float(area),
                "iscrowd": 0,
                "segmentation": []  # Empty for bounding box detection
            }
            
            coco_output["annotations"].append(annotation)
            annotation_id += 1
            valid_annotations += 1
        
        image_id += 1
    
    # Save COCO format JSON
    logger.info(f"Saving COCO format data to {output_json_path}")
    try:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f_out:
            json.dump(coco_output, f_out, indent=2)
    except IOError as e:
        logger.error(f"Error saving output file: {e}")
        return
    
    # Log conversion statistics
    logger.info("Conversion completed successfully!")
    logger.info(f"Images processed: {len(coco_output['images'])}")
    logger.info(f"Valid annotations: {valid_annotations}")
    logger.info(f"Invalid annotations skipped: {invalid_annotations}")
    logger.info(f"Categories: {len(coco_output['categories'])}")
    
    # Print category distribution
    category_counts = {}
    for ann in coco_output['annotations']:
        cat_id = ann['category_id']
        cat_name = next(cat['name'] for cat in coco_output['categories'] if cat['id'] == cat_id)
        category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
    
    logger.info("Category distribution:")
    for cat_name, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {cat_name}: {count}")


def main():
    """Main function to handle command line arguments and execute conversion."""
    parser = argparse.ArgumentParser(
        description='Convert BDD100K JSON labels to COCO format JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_bdd_to_coco.py --input bdd100k_labels_images_train.json --output train_coco.json
  python convert_bdd_to_coco.py --input bdd100k_labels_images_val.json --output val_coco.json
        """
    )
    
    parser.add_argument(
        '--input', 
        required=True, 
        help='Path to BDD100K input JSON label file'
    )
    parser.add_argument(
        '--output', 
        required=True, 
        help='Path to output COCO-style JSON file'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
        return 1
    
    # Perform conversion
    convert_bdd_to_coco(args.input, args.output)
    print(f"Conversion completed: {args.input} -> {args.output}")
    
    return 0


if __name__ == '__main__':
    exit(main())
