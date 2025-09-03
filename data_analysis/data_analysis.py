import os
import json
import pandas as pd
from collections import defaultdict

# Your existing DataVisualizer class
class DataVisualizer:
    """Utility for analyzing and plotting dataset distributions."""
    def __init__(self, config, train=None, val=None, filtered=False):
        self.config = config
        self.train = pd.DataFrame(train or [])
        self.val = pd.DataFrame(val or [])
        # Create visualization directory
        base_dir = config.get("paths", {}).get("vis_dir", "./cfg/data/insights")
        self.output_dir = os.path.join(base_dir, "filtered") if filtered else base_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _distribution(self, df, column):
        """Count occurrences in a column, return as dict."""
        return df[column].value_counts().to_dict() if column in df else {}
        
    def by_class(self, training=True):
        return self._distribution(self.train if training else self.val, "category")
        
    def by_weather(self, training=True):
        return self._distribution(self.train if training else self.val, "weather")
        
    def by_scene(self, training=True):
        return self._distribution(self.train if training else self.val, "scene")
        
    def plot_single(self, data, title="", filename="bar_chart.png"):
        """Plot a single bar chart."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(8, 6))
        palette = plt.cm.viridis(np.linspace(0, 1, len(data)))
        plt.bar(data.keys(), data.values(), color=palette, edgecolor="black")
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Categories")
        plt.ylabel("Frequency")
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        
    def plot_overview(self, training=False):
        """Plot class, weather, and scene distributions side-by-side."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        dataset = "train" if training else "val"
        fig, axes = plt.subplots(3, 1, figsize=(8, 12))
        plots = [
            (self.by_class(training), "Class Distribution"),
            (self.by_weather(training), "Weather Distribution"),
            (self.by_scene(training), "Scene Distribution"),
        ]
        for ax, (data, title) in zip(axes, plots):
            palette = plt.cm.plasma(np.linspace(0, 1, len(data)))
            ax.bar(data.keys(), data.values(), color=palette, edgecolor="black")
            ax.set_title(title, fontsize=12, fontweight="semibold")
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(data.keys(), rotation=30, ha="right")
            ax.grid(axis="y", linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{dataset}_distributions.png"))
        plt.close()
        
    def compare_distributions(self, train_data, val_data, title, filename):
        """Compare train vs val side by side."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        categories = sorted(set(train_data) | set(val_data))
        train_vals = [train_data.get(cat, 0) for cat in categories]
        val_vals = [val_data.get(cat, 0) for cat in categories]
        x = np.arange(len(categories))
        width = 0.4
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, train_vals, width, label="Train", color="steelblue")
        plt.bar(x + width/2, val_vals, width, label="Val", color="salmon")
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Categories")
        plt.ylabel("Counts")
        plt.xticks(x, categories, rotation=30, ha="right")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        
    def compare_all(self):
        """Generate comparison plots for class, weather, and scene."""
        self.compare_distributions(
            self.by_class(True),
            self.by_class(False),
            "Class: Train vs Val",
            "class_comparison.png",
        )
        self.compare_distributions(
            self.by_weather(True),
            self.by_weather(False),
            "Weather: Train vs Val",
            "weather_comparison.png",
        )
        self.compare_distributions(
            self.by_scene(True),
            self.by_scene(False),
            "Scene: Train vs Val",
            "scene_comparison.png",
        )


def load_bdd_data(json_path):
    """Load BDD100K JSON and extract relevant data."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    records = []
    for item in data:
        image_name = item.get('name', '')
        
        # Extract image-level attributes
        attributes = item.get('attributes', {})
        weather = attributes.get('weather', 'unknown')
        scene = attributes.get('scene', 'unknown')
        timeofday = attributes.get('timeofday', 'unknown')
        
        # Extract object-level data
        labels = item.get('labels', [])
        if labels:
            for label in labels:
                category = label.get('category', 'unknown')
                if category in ['person', 'rider', 'car', 'bus', 'truck', 
                               'bike', 'motor', 'traffic light', 'traffic sign', 'train']:
                    records.append({
                        'image_name': image_name,
                        'category': category,
                        'weather': weather,
                        'scene': scene,
                        'timeofday': timeofday
                    })
        else:
            # If no objects, still record image attributes
            records.append({
                'image_name': image_name,
                'category': 'none',
                'weather': weather,
                'scene': scene,
                'timeofday': timeofday
            })
    
    return records


def main():
    """Main function to run BDD data visualization."""
    
    # Configuration
    config = {
        "paths": {
            "vis_dir": "./bdd_analysis_results"
        }
    }
    
    # Paths to your BDD data
    TRAIN_JSON = "./vehicle_data/labels/bdd100k_labels_images_train.json"
    VAL_JSON = "./vehicle_data/labels/bdd100k_labels_images_val.json"
    
    print("Loading BDD training data...")
    train_records = load_bdd_data(TRAIN_JSON)
    print(f"Loaded {len(train_records)} training records")
    
    print("Loading BDD validation data...")
    val_records = load_bdd_data(VAL_JSON)
    print(f"Loaded {len(val_records)} validation records")
    
    # Create visualizer
    visualizer = DataVisualizer(config, train=train_records, val=val_records)
    
    print("Generating individual distribution plots...")
    
    # Generate individual plots for training data
    visualizer.plot_single(
        visualizer.by_class(training=True), 
        "Training Set - Class Distribution", 
        "train_classes.png"
    )
    
    visualizer.plot_single(
        visualizer.by_weather(training=True), 
        "Training Set - Weather Distribution", 
        "train_weather.png"
    )
    
    visualizer.plot_single(
        visualizer.by_scene(training=True), 
        "Training Set - Scene Distribution", 
        "train_scenes.png"
    )
    
    # Generate individual plots for validation data
    visualizer.plot_single(
        visualizer.by_class(training=False), 
        "Validation Set - Class Distribution", 
        "val_classes.png"
    )
    
    visualizer.plot_single(
        visualizer.by_weather(training=False), 
        "Validation Set - Weather Distribution", 
        "val_weather.png"
    )
    
    visualizer.plot_single(
        visualizer.by_scene(training=False), 
        "Validation Set - Scene Distribution", 
        "val_scenes.png"
    )
    
    print("Generating overview plots...")
    
    # Generate overview plots
    visualizer.plot_overview(training=True)   # train_distributions.png
    visualizer.plot_overview(training=False)  # val_distributions.png
    
    print("Generating comparison plots...")
    
    # Generate comparison plots
    visualizer.compare_all()
    
    print("Analysis complete!")
    print(f"Results saved to: {visualizer.output_dir}")
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Training samples: {len(visualizer.train)}")
    print(f"Validation samples: {len(visualizer.val)}")
    
    print("\nTop 5 classes in training:")
    class_dist = visualizer.by_class(training=True)
    for cls, count in sorted(class_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {cls}: {count}")
    
    print("\nWeather distribution in training:")
    weather_dist = visualizer.by_weather(training=True)
    for weather, count in sorted(weather_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"  {weather}: {count}")
    
    print("\nScene distribution in training:")
    scene_dist = visualizer.by_scene(training=True)
    for scene, count in sorted(scene_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"  {scene}: {count}")


if __name__ == "__main__":
    main()
