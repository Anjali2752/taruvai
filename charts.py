import matplotlib
matplotlib.use('Agg')   # 👈 VERY IMPORTANT
import numpy as np

import matplotlib.pyplot as plt
import io
import base64

def bar_trees_summary(existing_trees, predicted_trees):
    categories = ['Existing Trees', 'Predicted Trees']
    counts = [existing_trees, predicted_trees]

    plt.figure(figsize=(6,4))
    bars = plt.bar(categories, counts, color=['green', '#0f88f2'])
    plt.title('Trees Summary')
    plt.ylabel('Number of Trees')

    # Add value labels on top
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 500, yval, ha='center', va='bottom')

    # Save figure to a bytes buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    # Encode to base64 to embed in HTML
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
    return img_base64

def pie_tree_distribution(existing_trees, predicted_trees):
    labels = ['Existing Trees', 'Trees to Plant']
    sizes = [existing_trees, predicted_trees]
    colors = ['green', '#0f88f2']

    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Tree Distribution')
    plt.axis('equal')  # Equal aspect ratio ensures pie is circular

    # Save figure to buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
    return img_base64


import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from matplotlib.patches import Wedge, Circle

def aqi_gauge_attractive(current_aqi):
    fig, ax = plt.subplots(figsize=(7, 4), subplot_kw={'aspect':'equal'})
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.axis('off')

    # AQI ranges, colors, and labels
    ranges = [0, 50, 100, 150, 200, 300, 500]
    colors = ['#00e400','#ffff00','#ff7e00','#ff0000','#8f3f97','#7e0023']
    labels = ['Good','Moderate','Unhealthy for SG','Unhealthy','Very Unhealthy','Hazardous']

    # Draw colored arcs
    start_angle = 180
    for i in range(len(ranges)-1):
        wedge = Wedge(center=(0,0), r=1, theta1=start_angle-(ranges[i+1]-ranges[i])/500*180,
                      theta2=start_angle, width=0.3, facecolor=colors[i], edgecolor='white')
        ax.add_patch(wedge)
        # Add label
        angle_label = np.deg2rad(start_angle-(ranges[i+1]-ranges[i])/500*180/2)
        ax.text(0.7*np.cos(angle_label), 0.7*np.sin(angle_label), labels[i],
                horizontalalignment='center', verticalalignment='center', fontsize=8, rotation=0)
        start_angle -= (ranges[i+1]-ranges[i])/500*180

    # Draw needle
    theta = (current_aqi / 500) * 180
    needle_angle = np.deg2rad(180 - theta)
    ax.plot([0, 0.6*np.cos(needle_angle)], [0, 0.6*np.sin(needle_angle)],
             color='black', linewidth=4, zorder=5)
    # Needle center circle
    ax.add_patch(Circle((0,0),0.05,color='black', zorder=6))

    # Add AQI number
    ax.text(0, -0.15, f'AQI: {current_aqi}', fontsize=16, fontweight='bold', ha='center')

    # Background circle for better aesthetics
    bg_circle = Circle((0,0),1.05,fill=False, edgecolor='gray', linewidth=1.5)
    ax.add_patch(bg_circle)

    # Save to buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', transparent=True)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf8')
