import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_flat_pipeline():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # --- Style Definitions ---
    # Flat colors, no gradients, solid borders
    colors = {
        'input': '#E1D5E7', 'stroke_input': '#9673A6',
        'prep': '#DAE8FC', 'stroke_prep': '#6C8EBF',
        'noise': '#F8CECC', 'stroke_noise': '#B85450',
        'denoise': '#FFF2CC', 'stroke_denoise': '#D6B656',
        'seg': '#D5E8D4', 'stroke_seg': '#82B366',
        'feat': '#E6E6E6', 'stroke_feat': '#666666',
        'model': '#F5F5F5', 'stroke_model': '#333333',
        'arrow': '#555555'
    }
    
    def add_box(x, y, width, height, text, color_key, subtext=None):
        # Main box
        rect = patches.FancyBboxPatch((x, y), width, height, boxstyle='square,pad=0', 
                                      linewidth=1.5, edgecolor=colors[f'stroke_{color_key}'], 
                                      facecolor=colors[color_key], zorder=2)
        ax.add_patch(rect)
        # Main text
        ax.text(x + width/2, y + height/2 + (0.15 if subtext else 0), text, 
                ha='center', va='center', fontsize=10, fontweight='bold', color='#333333', zorder=3)
        # Subtext (details)
        if subtext:
            ax.text(x + width/2, y + height/2 - 0.25, subtext, 
                    ha='center', va='center', fontsize=8, color='#555555', zorder=3)
        return rect

    def add_arrow(x_start, y_start, x_end, y_end):
        ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                    arrowprops=dict(arrowstyle='->', linewidth=1.5, color=colors['arrow']), zorder=1)

    # --- 1. Input Stage ---
    add_box(0.5, 3.5, 2, 1, "Raw sEMG Input", 'input', "(NinaPro DB5)")

    # --- 2. Basic Filtering Stage ---
    add_box(3.0, 3.5, 2, 1, "Preprocessing Stage", 'prep', "Bandpass (20-200Hz)\n+ Notch (50Hz)")
    add_arrow(2.5, 4.0, 3.0, 4.0)

    # --- Path Split ---
    # Up: Robustness Testing Path
    add_arrow(5.0, 4.5, 5.5, 6.0)
    add_box(5.5, 5.5, 2.5, 1.5, "Noise Injection\n(Robustness Check)", 'noise', 
            "WGN, Hum, Drift,\nMotion, Pink, Spikes")
    
    add_arrow(8.0, 6.25, 8.5, 6.25)
    add_box(8.5, 5.5, 2.5, 1.5, "Denoising & Restoration", 'denoise',
            "Wavelet, Kalman,\nPCA, Notch")
    
    add_arrow(11.0, 6.0, 11.5, 4.5) # Path merges back

    # Straight: Clean Path (Implicit)
    add_arrow(5.0, 4.0, 8.5, 4.0)
    ax.text(6.75, 4.15, "Clean Path", ha='center', fontsize=8, color=colors['arrow'])
    add_arrow(8.5, 4.0, 11.5, 4.0)

    # --- 3. Segmentation Stage ---
    add_box(11.5, 3.5, 2, 1, "Adaptive Segmentation", 'seg', "(Dual-Threshold Energy)")

    # --- 4. Feature Extraction Stage (Container) ---
    # Outer container to group them
    feat_container = patches.Rectangle((1.5, 0.5), 5.5, 2.5, linewidth=1.5, edgecolor=colors['stroke_feat'], 
                                       facecolor='none', linestyle='--')
    ax.add_patch(feat_container)
    ax.text(4.25, 2.75, "Multi-domain Feature Engineering (12-dim/ch)", ha='center', fontweight='bold', fontsize=9)

    # Inner parallel boxes
    add_box(1.75, 1.0, 1.5, 1.2, "Time Domain\n(TD)", 'feat', "RMS, MAV, WL,\nZC, SSC")
    add_box(3.5, 1.0, 1.5, 1.2, "Freq Domain\n(FD)", 'feat', "MNF, MDF, SE")
    add_box(5.25, 1.0, 1.5, 1.2, "Auto-Regressive\n(AR)", 'feat', "Yule-Walker\n(Order 4)")
    
    # Arrow from Seg to Feat container
    add_arrow(12.5, 3.5, 12.5, 1.75) # Down
    add_arrow(12.5, 1.75, 7.0, 1.75) # Left into container

    # --- 5. Model Training Stage (Container) ---
    model_container = patches.Rectangle((8.0, 0.5), 5.5, 2.5, linewidth=1.5, edgecolor=colors['stroke_model'], 
                                       facecolor='none', linestyle='--')
    ax.add_patch(model_container)
    ax.text(10.75, 2.75, "Classification & Evaluation (Stratified 5-Fold CV)", ha='center', fontweight='bold', fontsize=9)

    # Inner parallel models
    add_box(8.25, 1.0, 1.5, 1.2, "Random Forest\n(RF)", 'model', "Best Performer")
    add_box(10.0, 1.0, 1.5, 1.2, "SVM\n(RBF Kernel)", 'model')
    add_box(11.75, 1.0, 1.5, 1.2, "LDA\n(Linear)", 'model', "Baseline")

    # Arrow between containers
    add_arrow(7.0, 1.75, 8.0, 1.75)

    # --- Final Output ---
    add_arrow(13.5, 1.75, 14.0, 1.75) # Out of model container
    # Using text annotation instead of box for final output to look cleaner
    ax.text(14.2, 1.75, "Performance Metrics\n(Accuracy, F1-Score)", ha='left', va='center', fontweight='bold', fontsize=10)

    plt.title("Figure 2-1: System Workflow Diagram of the Robust sEMG Gesture Recognition Pipeline", fontsize=12, y=1.05, fontweight='bold')
    plt.tight_layout()
    plt.savefig('flat_pipeline_flowchart.png', dpi=300, bbox_inches='tight')
    print("Generated flat style flowchart: flat_pipeline_flowchart.png")

draw_flat_pipeline()