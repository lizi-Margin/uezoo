import pandas as pd
import matplotlib.pyplot as plt
import csv
from pathlib import Path

# 1. Organize render data
render_data = [
    # With 360° camera movement
    {"camera_movement": "With 360° Movement", "resolution": "240x240", "fps_setting": 15, 
     "render_fps": 15.76153363022635, "total_frames": 331, "real_time_s": 21.001495599746704, "virtual_time_s": 10},
    {"camera_movement": "With 360° Movement", "resolution": "480x480", "fps_setting": 15, 
     "render_fps": 7.2589418615494425, "total_frames": 331, "real_time_s": 45.59893250465393, "virtual_time_s": 10},
    {"camera_movement": "With 360° Movement", "resolution": "480x480", "fps_setting": 30, 
     "render_fps": 7.035200317170496, "total_frames": 480, "real_time_s": 68.22833442687988, "virtual_time_s": 10},
    # Without 360° camera movement
    {"camera_movement": "Without 360° Movement", "resolution": "240x240", "fps_setting": 15, 
     "render_fps": 11.814586296546443, "total_frames": 150, "real_time_s": 12.696170330047607, "virtual_time_s": 10},
    {"camera_movement": "Without 360° Movement", "resolution": "480x480", "fps_setting": 15, 
     "render_fps": 5.936521335945651, "total_frames": 151, "real_time_s": 25.435771465301514, "virtual_time_s": 10},
    {"camera_movement": "Without 360° Movement", "resolution": "480x480", "fps_setting": 30, 
     "render_fps": 5.967402636250935, "total_frames": 299, "real_time_s": 50.10655236244202, "virtual_time_s": 10}
]

# 2. Generate CSV file
csv_path = Path("render_performance_data_en.csv")
with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=render_data[0].keys())
    writer.writeheader()
    writer.writerows(render_data)

print(f"CSV file generated: {csv_path.absolute()}")

# 3. Load data and create visualization
df = pd.read_csv(csv_path)

# Split data into two groups
df_with_movement = df[df["camera_movement"] == "With 360° Movement"].reset_index(drop=True)
df_without_movement = df[df["camera_movement"] == "Without 360° Movement"].reset_index(drop=True)

# Define visualization function for reusability
def create_comparison_chart(data, title_suffix, save_filename):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Prepare X-axis labels (parameter combination)
    x_labels = [f"{row['resolution']}\nSet FPS:{row['fps_setting']}" 
                for _, row in data.iterrows()]
    x_pos = range(len(x_labels))
    
    # Left Y-axis: Real Time (blue bar chart)
    bars = ax1.bar([p - 0.2 for p in x_pos], data["real_time_s"], 
                   width=0.4, label="Real Time (s)", color='#1f77b4', alpha=0.8)
    ax1.set_xlabel("Render Parameters", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Real Time (Seconds)", color='#1f77b4', fontsize=11, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=9)
    
    # Add value labels for real time
    for bar, time in zip(bars, data["real_time_s"]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 f"{time:.1f}", ha='center', va='bottom', 
                 fontsize=8, fontweight='bold', color='#1f77b4')
    
    # Right Y-axis: Render FPS (orange line chart)
    ax2 = ax1.twinx()
    line = ax2.plot(x_pos, data["render_fps"], marker='o', linewidth=3, 
                    markersize=8, label="Render FPS", color='#ff7f0e', alpha=0.8)
    ax2.set_ylabel("Render FPS (Reference)", color='#ff7f0e', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    
    # Add value labels for render FPS
    for pos, fps in zip(x_pos, data["render_fps"]):
        ax2.text(pos, fps + 0.3, f"{fps:.1f}", ha='center', va='bottom', 
                 fontsize=8, fontweight='bold', color='#ff7f0e')
    
    # Chart title and legend
    full_title = f"Render Performance: {title_suffix}\n(Note: Character disappearance bug exists with 360° movement)"
    plt.title(full_title, fontsize=12, fontweight='bold', pad=15)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Chart saved: {save_filename}")

# Create charts for both groups
create_comparison_chart(
    data=df_with_movement,
    title_suffix="With 360° Camera Movement",
    save_filename="render_chart_with_360_movement.png"
)

create_comparison_chart(
    data=df_without_movement,
    title_suffix="Without 360° Camera Movement",
    save_filename="render_chart_without_360_movement.png"
)

# Print data overview
print("\n=== Data Overview ===")
print("\n1. With 360° Camera Movement:")
print(df_with_movement[["resolution", "fps_setting", "render_fps", "real_time_s"]].round(2))

print("\n2. Without 360° Camera Movement:")
print(df_without_movement[["resolution", "fps_setting", "render_fps", "real_time_s"]].round(2))
    