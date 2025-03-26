import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageDraw
from zero_sum_eval.analysis.calculate_ratings import calculate_ratings

# Set the style to a more modern look
plt.style.use('seaborn-v0_8-whitegrid')

# Custom color palette - more visually appealing
CUSTOM_COLORS = [
    "#4C72B0",  # blue
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B3",  # purple
    "#CCB974",  # yellow
    "#64B5CD",  # light blue
    "#4C72B0",  # blue (repeat with lower alpha for more games)
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B3",  # purple
]

# Font settings for a more professional look
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.title_fontsize': 14
})

# Function to load and resize logo with circular cropping and guaranteed white background
def get_logo(logo_path, size=0.15):
    try:
        # Use PIL directly to load the image
        pil_img = Image.open(logo_path)
        
        # Force conversion to RGBA first to properly handle all image types
        if pil_img.mode == 'P':  # Palette mode
            pil_img = pil_img.convert('RGBA')
        elif pil_img.mode != 'RGBA':
            pil_img = pil_img.convert('RGBA')
        
        # Create a square image by cropping
        width, height = pil_img.size
        size_px = min(width, height)
        
        # Calculate crop box (centered)
        left = (width - size_px) // 2
        top = (height - size_px) // 2
        right = left + size_px
        bottom = top + size_px
        
        # Crop to square
        square_img = pil_img.crop((left, top, right, bottom))
        
        # Create a solid white background image
        white_bg = Image.new('RGB', (size_px, size_px), (255, 255, 255))
        
        # Create a circular mask
        mask = Image.new('L', (size_px, size_px), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size_px, size_px), fill=255)
        
        # Paste the logo onto the white background using the mask
        # This is the key step - we're using the RGBA image as the source but pasting onto RGB
        white_bg.paste(square_img, (0, 0), square_img.split()[3])  # Use alpha channel as mask
        
        # Add a border
        draw_border = ImageDraw.Draw(white_bg)
        draw_border.ellipse((0, 0, size_px-1, size_px-1), outline=(50, 50, 50), width=2)
        
        # Resize to a standard size
        target_size = (100, 100)
        resized_img = white_bg.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array for matplotlib
        img_array = np.array(resized_img)
        
        # Create an OffsetImage with the standardized image
        offset_img = OffsetImage(img_array, zoom=size)
        
        # Force the image to have a white background in matplotlib
        offset_img.set_zorder(10)  # Ensure it's on top
        
        return offset_img
    except Exception as e:
        print(f"Error loading logo {logo_path}: {e}")
        # Create a fallback simple circle with the first letter of the model
        model_name = os.path.basename(logo_path).split('.')[0]
        first_letter = model_name[0].upper() if model_name else "?"
        
        # Create a white circle with text
        fallback = Image.new('RGB', (100, 100), (255, 255, 255))
        draw = ImageDraw.Draw(fallback)
        draw.ellipse((0, 0, 99, 99), outline=(100, 100, 100), width=2)
        
        # Add text (centered)
        try:
            # Try to use a font if available
            from PIL import ImageFont
            font = ImageFont.truetype("Arial", 40)
            text_width, text_height = draw.textsize(first_letter, font=font)
            draw.text((50-text_width//2, 50-text_height//2), first_letter, fill=(0, 0, 0), font=font)
        except:
            # Fallback if font not available
            draw.text((40, 30), first_letter, fill=(0, 0, 0))
        
        img_array = np.array(fallback)
        return OffsetImage(img_array, zoom=size)

# Map model names to their logo files
LOGO_DIR = "paper/logos"
LOGO_MAPPING = {
    "gpt-4o": os.path.join(LOGO_DIR, "gpt-4.png"),
    "claude-3.7-sonnet": os.path.join(LOGO_DIR, "claude.png"),
    "claude-3.7-sonnet-thinking": os.path.join(LOGO_DIR, "claude.png"),
    "gemini-2.0-flash": os.path.join(LOGO_DIR, "gemini.png"),
    "llama-3.3-70b": os.path.join(LOGO_DIR, "llama.png"),
    "llama-3.1-405b": os.path.join(LOGO_DIR, "llama.png"),
    "llama-3.1-70b": os.path.join(LOGO_DIR, "llama.png"),
    "deepseek-chat": os.path.join(LOGO_DIR, "deepseek.png"),
    "deepseek-r1": os.path.join(LOGO_DIR, "deepseek.png"),
    "qwen2.5-32b": os.path.join(LOGO_DIR, "qwen2.png"),
    "qwq-32b": os.path.join(LOGO_DIR, "qwen2.png"),
    "o3-mini-high": os.path.join(LOGO_DIR, "openai.png")
}

ROOT_DIR = "/Users/haidark/Library/CloudStorage/GoogleDrive-haidark@gmail.com/My Drive/Zero Sum Eval/rankings-3-9-25/"
ALL_DIRS = {
    "chess": "rankings-3-9-25_chess_predict_vs_cot",
    "mathquiz": "rankings-3-9-25_mathquiz_predict_vs_cot",
}

ROLE_WEIGHTS = {
    "chess": {
        "white": 1.0,
        "black": 2.0
    },
    "debate": None,
    "gandalf": {
        "sentinel": 1.0,
        "infiltrator": 2.0
    },
    "liars_dice": None,
    "mathquiz": {
        "student": 1.0,
        "teacher": 2.0
    },
    "poker": None,
    "pyjail": {
        "defender": 2.0,    
        "attacker": 1.0
    }
}

# Define model pairs (base model and its CoT variant)
MODEL_PAIRS = [
    ("gpt-4o", "gpt-4o-cot"),
    ("claude-3.7-sonnet", "claude-3.7-sonnet-cot"),
    ("gemini-2.0-flash", "gemini-2.0-flash-cot"),
    ("llama-3.1-70b", "llama-3.1-70b-cot"),
    ("llama-3.3-70b", "llama-3.3-70b-cot"),
    ("llama-3.1-405b", "llama-3.1-405b-cot"),
    ("qwen2.5-32b", "qwen2.5-32b-cot"),
    ("deepseek-chat", "deepseek-chat-cot"),
]

# Prepare data for plotting
game_ratings = {}
all_models = []

for game, dir in ALL_DIRS.items():
    if dir is None:
        continue
    
    # Extract all models for this comparison
    models_to_include = []
    for base, cot in MODEL_PAIRS:
        models_to_include.extend([base, cot])
    
    ratings = calculate_ratings(os.path.join(ROOT_DIR, dir), 
                                bootstrap_rounds=100, 
                                max_time_per_player=None,
                                models=models_to_include,
                                role_weights=ROLE_WEIGHTS[game])
    
    game_ratings[game] = ratings
    all_models.extend([model for model in ratings.index if model in models_to_include])

# Remove duplicates while preserving order
all_models = list(dict.fromkeys(all_models))

# Create a figure with a specific aspect ratio
fig, axes = plt.subplots(1, len(MODEL_PAIRS), figsize=(18, 10), dpi=300, sharey=True)

# Set background color
fig.patch.set_facecolor('#F8F8F8')
for ax in axes:
    ax.set_facecolor('#F8F8F8')

# Plot the data for each model pair
for j, (base_model, cot_model) in enumerate(MODEL_PAIRS):
    ax = axes[j]
    
    # Set up bar positions
    games = list(game_ratings.keys())
    x_pos = np.arange(len(games))
    bar_width = 0.35
    
    # Plot bars for each game
    for i, game in enumerate(games):
        ratings = game_ratings[game]
        
        # Get ratings for base model
        base_rating = ratings['rating']['predicted'].get(base_model, 0)
        base_lower = ratings['rating']['lower'].get(base_model, 0)
        base_upper = ratings['rating']['upper'].get(base_model, 0)
        
        # Get ratings for CoT model
        cot_rating = ratings['rating']['predicted'].get(cot_model, 0)
        cot_lower = ratings['rating']['lower'].get(cot_model, 0)
        cot_upper = ratings['rating']['upper'].get(cot_model, 0)
        
        # Plot base model bar
        ax.bar(x_pos[i] - bar_width/2, base_rating, width=bar_width, 
               color=CUSTOM_COLORS[0], 
               edgecolor='white', linewidth=0.5,
               label="Predict" if i == 0 and j == 0 else "")
        
        # Plot CoT model bar
        ax.bar(x_pos[i] + bar_width/2, cot_rating, width=bar_width, 
               color=CUSTOM_COLORS[1], 
               edgecolor='white', linewidth=0.5,
               label="CoT" if i == 0 and j == 0 else "")
        
        # Add error bars
        ax.errorbar(x_pos[i] - bar_width/2, base_rating, 
                   yerr=[[max(0, base_rating-base_lower)], [max(0, base_upper-base_rating)]], 
                   fmt='none', ecolor='black', capsize=5, alpha=0.5)
        ax.errorbar(x_pos[i] + bar_width/2, cot_rating, 
                   yerr=[[max(0, cot_rating-cot_lower)], [max(0, cot_upper-cot_rating)]], 
                   fmt='none', ecolor='black', capsize=5, alpha=0.5)
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='#333333', linestyle='-', linewidth=1, alpha=0.3)
    
    # Customize the grid
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#333333')
    ax.set_axisbelow(True)  # Put grid behind bars
    
    # Set title and labels
    ax.set_title(f"{base_model}", fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([game.capitalize() for game in games], rotation=45, ha='right')
    
    # Format y-axis with fewer ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    
    # Add a subtle box around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#DDDDDD')
        spine.set_linewidth(0.5)
    
    # Add model logo if available
    if base_model in LOGO_MAPPING:
        logo = get_logo(LOGO_MAPPING[base_model], size=0.15)
        if logo:
            ab = AnnotationBbox(logo, (0.5, 0.95), xycoords='axes fraction', 
                               frameon=False, box_alignment=(0.5, 1))
            ax.add_artist(ab)

# Set common y-label
fig.text(0.02, 0.5, 'Rating', va='center', rotation='vertical', fontsize=16)

# Add a main title
fig.suptitle('Predict vs. Chain-of-Thought Performance Comparison', 
             fontsize=20, fontweight='bold', y=0.98)

# Create a common legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05),
          ncol=2, frameon=True, fancybox=True, shadow=True, fontsize=14)

# Add a subtle watermark
fig.text(0.99, 0.01, 'ZeroSumEval', fontsize=10, color='gray', ha='right', va='bottom', alpha=0.5)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.15, top=0.85, wspace=0.05)

# Save the figure with high resolution
plt.savefig('paper/figures/llama_predict_vs_cot.pdf', dpi=300, bbox_inches='tight')

# Alternative visualization that focuses on the difference between CoT and Predict
def create_difference_plot():
    fig, ax = plt.subplots(figsize=(15, 10), dpi=300)
    
    # Set background color
    fig.patch.set_facecolor('#F8F8F8')
    ax.set_facecolor('#F8F8F8')
    
    # Set up positions
    games = list(game_ratings.keys())
    num_games = len(games)
    num_models = len(MODEL_PAIRS)
    
    # Calculate positions for grouped bars
    indices = np.arange(num_games)
    width = 0.8 / num_models
    
    # Plot bars for each model pair
    for i, (base_model, cot_model) in enumerate(MODEL_PAIRS):
        differences = []
        error_bars = []
        
        # Calculate differences for each game
        for game in games:
            ratings = game_ratings[game]
            
            # Get ratings
            base_rating = ratings['rating']['predicted'].get(base_model, 0)
            cot_rating = ratings['rating']['predicted'].get(cot_model, 0)
            
            # Calculate difference (CoT - Predict)
            difference = cot_rating - base_rating
            differences.append(difference)
            
            # Calculate error for the difference
            base_lower = ratings['rating']['lower'].get(base_model, 0)
            base_upper = ratings['rating']['upper'].get(base_model, 0)
            cot_lower = ratings['rating']['lower'].get(cot_model, 0)
            cot_upper = ratings['rating']['upper'].get(cot_model, 0)
            
            # Simple error propagation (not statistically rigorous but visually useful)
            error = np.sqrt((base_upper - base_lower)**2 + (cot_upper - cot_lower)**2) / 2
            error_bars.append(error)
        
        # Plot the differences
        positions = indices + i * width - (num_models - 1) * width / 2
        bars = ax.bar(positions, differences, width, 
                     label=f"{base_model.replace('llama-', 'Llama ')}",
                     color=CUSTOM_COLORS[i], alpha=0.8)
        
        # Add error bars
        ax.errorbar(positions, differences, yerr=error_bars, fmt='none', 
                   ecolor='black', capsize=5, alpha=0.5)
        
        # Add value labels on top of bars
        for j, (pos, diff) in enumerate(zip(positions, differences)):
            color = 'green' if diff > 0 else 'red'
            ax.text(pos, diff + (0.1 if diff > 0 else -0.1), 
                   f"{diff:.2f}", ha='center', va='bottom' if diff > 0 else 'top',
                   fontsize=10, fontweight='bold', color=color)
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='#333333', linestyle='-', linewidth=1)
    
    # Customize the grid
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#333333')
    ax.set_axisbelow(True)
    
    # Set labels and title
    ax.set_title('Impact of Chain-of-Thought vs. Predict Approach', fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Game', fontsize=16, labelpad=15)
    ax.set_ylabel('Rating Difference (CoT - Predict)', fontsize=16, labelpad=15)
    
    # Set x-ticks
    ax.set_xticks(indices)
    ax.set_xticklabels([game.capitalize() for game in games], fontsize=14)
    
    # Add a legend
    ax.legend(title='Model', fontsize=12, title_fontsize=14, 
             loc='upper center', bbox_to_anchor=(0.5, -0.15),
             ncol=num_models, frameon=True, fancybox=True, shadow=True)
    
    # Add annotations
    ax.text(0.01, 0.99, 'CoT Better →', transform=ax.transAxes, 
           fontsize=12, ha='left', va='top', color='green')
    ax.text(0.01, 0.01, '← Predict Better', transform=ax.transAxes, 
           fontsize=12, ha='left', va='bottom', color='red')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    # Save the figure
    plt.savefig('paper/figures/llama_cot_vs_predict_difference.pdf', dpi=300, bbox_inches='tight')

# Call the function to create the difference plot
create_difference_plot()

# Add a heatmap visualization function
def create_heatmap():
    # Get all games and models
    games = list(game_ratings.keys())
    
    # Create matrices for the data
    diff_matrix = np.zeros((len(MODEL_PAIRS), len(games)))
    
    # Fill the matrices
    for i, (base_model, cot_model) in enumerate(MODEL_PAIRS):
        for j, game in enumerate(games):
            ratings = game_ratings[game]
            
            # Get ratings
            base_rating = ratings['rating']['predicted'].get(base_model, 0)
            cot_rating = ratings['rating']['predicted'].get(cot_model, 0)
            
            # Calculate difference (CoT - Predict)
            diff_matrix[i, j] = cot_rating - base_rating
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    
    # Create the heatmap
    cmap = plt.cm.RdBu_r  # Red-Blue diverging colormap
    norm = plt.Normalize(vmin=-max(abs(diff_matrix.min()), abs(diff_matrix.max())), 
                         vmax=max(abs(diff_matrix.min()), abs(diff_matrix.max())))
    
    im = ax.imshow(diff_matrix, cmap=cmap, norm=norm)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Rating Difference (CoT - Predict)", rotation=-90, va="bottom", fontsize=12)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(games)))
    ax.set_yticks(np.arange(len(MODEL_PAIRS)))
    
    ax.set_xticklabels([game.capitalize() for game in games], fontsize=12)
    ax.set_yticklabels([base_model.replace('llama-', 'Llama ') for base_model, _ in MODEL_PAIRS], fontsize=12)
    
    # Rotate the x labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations with the values
    for i in range(len(MODEL_PAIRS)):
        for j in range(len(games)):
            value = diff_matrix[i, j]
            text_color = "white" if abs(value) > (norm.vmax - norm.vmin) / 3 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", 
                   color=text_color, fontweight="bold", fontsize=10)
    
    # Add title and labels
    ax.set_title("Impact of Chain-of-Thought vs. Predict by Model and Game", fontsize=16, pad=20)
    
    # Add annotations explaining the color scheme
    fig.text(0.01, 0.95, "Blue = CoT Better", fontsize=12, color='blue', ha='left')
    fig.text(0.01, 0.92, "Red = Predict Better", fontsize=12, color='red', ha='left')
    
    # Add model logos
    for i, (base_model, _) in enumerate(MODEL_PAIRS):
        if base_model in LOGO_MAPPING:
            logo = get_logo(LOGO_MAPPING[base_model], size=0.1)
            if logo:
                ab = AnnotationBbox(logo, (-0.6, i), xycoords=('data', 'data'),
                                  frameon=False, box_alignment=(0.5, 0.5))
                ax.add_artist(ab)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('paper/figures/cot_vs_predict_heatmap.pdf', dpi=300, bbox_inches='tight')

# Call the function to create the heatmap
create_heatmap()

# Add a radar chart visualization
def create_radar_chart():
    # Get all games
    games = list(game_ratings.keys())
    num_games = len(games)
    
    # Create figure
    fig = plt.figure(figsize=(15, 12), dpi=300)
    
    # Calculate the number of rows and columns for subplots
    n_rows = (len(MODEL_PAIRS) + 1) // 2
    n_cols = min(2, len(MODEL_PAIRS))
    
    # Set up angles for radar chart
    angles = np.linspace(0, 2*np.pi, num_games, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Add game names to the list and close the loop
    game_labels = [game.capitalize() for game in games]
    game_labels += game_labels[:1]
    
    # Create subplots for each model
    for i, (base_model, cot_model) in enumerate(MODEL_PAIRS):
        ax = fig.add_subplot(n_rows, n_cols, i+1, polar=True)
        
        # Get data for base and CoT models
        base_values = []
        cot_values = []
        
        for game in games:
            ratings = game_ratings[game]
            
            # Get ratings
            base_rating = ratings['rating']['predicted'].get(base_model, 0)
            cot_rating = ratings['rating']['predicted'].get(cot_model, 0)
            
            base_values.append(base_rating)
            cot_values.append(cot_rating)
        
        # Close the loop
        base_values += base_values[:1]
        cot_values += cot_values[:1]
        
        # Plot the data
        ax.plot(angles, base_values, 'o-', linewidth=2, label='Predict', color=CUSTOM_COLORS[0])
        ax.plot(angles, cot_values, 'o-', linewidth=2, label='CoT', color=CUSTOM_COLORS[1])
        ax.fill(angles, base_values, alpha=0.1, color=CUSTOM_COLORS[0])
        ax.fill(angles, cot_values, alpha=0.1, color=CUSTOM_COLORS[1])
        
        # Set labels and title
        ax.set_thetagrids(np.degrees(angles[:-1]), game_labels[:-1])
        ax.set_title(f"{base_model}", fontsize=14, pad=20)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        # Add model logo if available
        if base_model in LOGO_MAPPING:
            logo = get_logo(LOGO_MAPPING[base_model], size=0.1)
            if logo:
                ab = AnnotationBbox(logo, (0.5, 1.15), xycoords='axes fraction',
                                  frameon=False, box_alignment=(0.5, 0.5))
                ax.add_artist(ab)
    
    # Add a main title
    fig.suptitle('Predict vs. Chain-of-Thought Performance by Game', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save the figure
    plt.savefig('paper/figures/cot_vs_predict_radar.pdf', dpi=300, bbox_inches='tight')

# Call the function to create the radar chart
create_radar_chart()

# Improved lollipop chart with better logo and value placement
def create_lollipop_chart():
    # Get all games
    games = list(game_ratings.keys())
    
    # Create figure with subplots - one per game
    fig, axes = plt.subplots(len(games), 1, figsize=(14, 3.5*len(games)), dpi=300)
    if len(games) == 1:
        axes = [axes]
    
    # Set background color
    fig.patch.set_facecolor('#FFFFFF')
    for ax in axes:
        ax.set_facecolor('#FFFFFF')
    
    # Plot data for each game
    for i, game in enumerate(games):
        ax = axes[i]
        ratings = game_ratings[game]
        
        # Calculate differences for each model
        model_names = []
        differences = []
        colors = []
        
        for j, (base_model, cot_model) in enumerate(MODEL_PAIRS):
            # Get ratings
            base_rating = ratings['rating']['predicted'].get(base_model, 0)
            cot_rating = ratings['rating']['predicted'].get(cot_model, 0)
            
            # Calculate difference (CoT - Predict)
            difference = cot_rating - base_rating
            differences.append(difference)
            
            # Add model name
            model_names.append(base_model)
            
            # Determine color based on difference
            colors.append('#2E8B57' if difference > 0 else '#CD5C5C')  # SeaGreen vs IndianRed
        
        # Sort by difference value
        sorted_indices = np.argsort(differences)
        model_names = [model_names[idx] for idx in sorted_indices]
        differences = [differences[idx] for idx in sorted_indices]
        colors = [colors[idx] for idx in sorted_indices]
        
        # Calculate spacing for model names and logos
        max_name_length = max([len(name) for name in model_names])
        if game == "chess":
            left_text_offset = -1 - (max_name_length * 0.01)  # For positive deltas
            right_text_offset = 1 + (max_name_length * 0.01)  # For negative deltas
            # Adjust value offset based on scale
            value_offset_pos = 5
            value_offset_neg = -5
        else:
            left_text_offset = -0.15 - (max_name_length * 0.01)  # For positive deltas
            right_text_offset = 0.15 + (max_name_length * 0.01)  # For negative deltas
            # Smaller offset for mathquiz
            value_offset_pos = 0.5
            value_offset_neg = -0.5
        
        # Plot horizontal lines from zero to the difference
        for j, (model, diff, color) in enumerate(zip(model_names, differences, colors)):
            # Plot line
            ax.plot([0, diff], [j, j], color=color, linestyle='-', linewidth=2.5, alpha=0.7)
            
            # Format model name
            formatted_name = model
            
            # Position model name based on whether delta is positive or negative
            if diff >= 0:  # Positive delta - name on left
                # Add model name on left
                ax.text(left_text_offset, j, formatted_name, 
                       ha='right', va='center', fontsize=11, fontweight='medium')
            else:  # Negative delta - name on right
                # Add model name on right
                ax.text(right_text_offset, j, formatted_name, 
                       ha='left', va='center', fontsize=11, fontweight='medium')
            
            # Add difference value near the logo
            if abs(diff) > 1:  # Only show non-zero differences
                # Position value based on direction
                value_offset = value_offset_pos if diff > 0 else value_offset_neg
                # Format value based on game (chess has larger numbers)
                value_text = f"{diff:.0f}"
                
                # Add text with white outline for better visibility
                text_obj = ax.text(diff + value_offset, j, value_text,
                       ha='left' if diff > 0 else 'right',
                       va='center', fontsize=10, color='black', fontweight='bold',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
            
            # Add logo at the tip of the lollipop with increased size
            if model in LOGO_MAPPING:
                logo = get_logo(LOGO_MAPPING[model], size=0.15)
                if logo:
                    # Place logo at the tip of the lollipop
                    ab = AnnotationBbox(logo, (diff, j), xycoords='data',
                                      frameon=False, box_alignment=(0.5, 0.5))
                    ax.add_artist(ab)
        
        # Add a vertical line at x=0
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # Set title for each game
        ax.set_title(f"{game.capitalize()}", fontsize=16, fontweight='bold')
        
        # Set x-axis label
        if i == len(games) - 1:  # Only add label to bottom subplot
            ax.set_xlabel('Rating Difference (CoT - Predict)', fontsize=12)
        
        # Remove y-axis ticks and labels
        ax.set_yticks([])
        ax.set_yticklabels([])
        
        # Set x-axis limits with some padding
        max_abs_diff = max(abs(min(differences)), abs(max(differences)))
        padding = max(0.5, max_abs_diff * 0.3)  # Increased padding to accommodate larger logos
        ax.set_xlim(-max_abs_diff - padding - 0.5, max_abs_diff + padding + 0.5)
        
        # Add annotations for interpretation
        ax.text(max_abs_diff + padding*0.8, len(model_names)-1, 'CoT Better →', 
               fontsize=11, ha='right', va='top', color='#2E8B57', fontweight='bold')
        ax.text(-max_abs_diff - padding*0.8, len(model_names)-1, '← Predict Better', 
               fontsize=11, ha='left', va='top', color='#CD5C5C', fontweight='bold')
        
        # Add subtle grid lines
        ax.grid(axis='x', linestyle='--', alpha=0.2)
        ax.set_axisbelow(True)
        
        # Clean up the frame
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#DDDDDD')
    
    # Add a subtle watermark
    fig.text(0.98, 0.02, 'ZeroSumEval', fontsize=8, color='gray', 
            ha='right', va='bottom', alpha=0.5, style='italic')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.4)
    
    # Save the figure
    plt.savefig('paper/figures/cot_vs_predict_lollipop.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('paper/figures/cot_vs_predict_lollipop.png', dpi=300, bbox_inches='tight')

# Call the function to create the improved lollipop chart
create_lollipop_chart()

# Add a slope chart visualization
def create_slope_chart():
    # Get all games
    games = list(game_ratings.keys())
    
    # Create figure with subplots - one per game
    fig, axes = plt.subplots(1, len(games), figsize=(6*len(games), 10), dpi=300, sharey=True)
    if len(games) == 1:
        axes = [axes]
    
    # Set background color
    fig.patch.set_facecolor('#F8F8F8')
    for ax in axes:
        ax.set_facecolor('#F8F8F8')
    
    # Find global min and max for consistent y-axis
    global_min = float('inf')
    global_max = float('-inf')
    
    for game in games:
        ratings = game_ratings[game]
        for base_model, cot_model in MODEL_PAIRS:
            base_rating = ratings['rating']['predicted'].get(base_model, 0)
            cot_rating = ratings['rating']['predicted'].get(cot_model, 0)
            global_min = min(global_min, base_rating, cot_rating)
            global_max = max(global_max, base_rating, cot_rating)
    
    # Add some padding
    y_padding = (global_max - global_min) * 0.1
    global_min -= y_padding
    global_max += y_padding
    
    # Plot data for each game
    for i, game in enumerate(games):
        ax = axes[i]
        ratings = game_ratings[game]
        
        # Set up x positions
        x_left = 0
        x_right = 1
        
        # Plot lines for each model pair
        for j, (base_model, cot_model) in enumerate(MODEL_PAIRS):
            # Get ratings
            base_rating = ratings['rating']['predicted'].get(base_model, 0)
            cot_rating = ratings['rating']['predicted'].get(cot_model, 0)
            
            # Determine color based on difference
            color = 'green' if cot_rating > base_rating else 'red'
            
            # Plot the line
            ax.plot([x_left, x_right], [base_rating, cot_rating], 
                   color=color, linewidth=2, alpha=0.7)
            
            # Add points
            ax.scatter(x_left, base_rating, color=color, s=100, zorder=5)
            ax.scatter(x_right, cot_rating, color=color, s=100, zorder=5)
            
            # Add model name
            ax.text(x_left-0.05, base_rating, base_model, 
                   ha='right', va='center', fontsize=10)
            
            # Add rating values
            ax.text(x_left+0.05, base_rating, f"{base_rating:.2f}", 
                   ha='left', va='center', fontsize=9)
            ax.text(x_right-0.05, cot_rating, f"{cot_rating:.2f}", 
                   ha='right', va='center', fontsize=9)
            
            # Add model logo if available
            if base_model in LOGO_MAPPING:
                logo = get_logo(LOGO_MAPPING[base_model], size=0.05)
                if logo:
                    ab = AnnotationBbox(logo, (x_left-0.15, base_rating), 
                                      frameon=False, box_alignment=(1, 0.5))
                    ax.add_artist(ab)
        
        # Set title and labels
        ax.set_title(f"{game.capitalize()}", fontsize=16, pad=20)
        
        # Set x-axis ticks and labels
        ax.set_xticks([x_left, x_right])
        ax.set_xticklabels(['Predict', 'CoT'], fontsize=14)
        
        # Set y-axis limits
        ax.set_ylim(global_min, global_max)
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add grid lines
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
    
    # Add a common y-label
    fig.text(0.02, 0.5, 'Rating', va='center', rotation='vertical', fontsize=16)
    
    # Add a main title
    fig.suptitle('Predict vs. Chain-of-Thought Performance Comparison', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, top=0.9)
    
    # Save the figure
    plt.savefig('paper/figures/cot_vs_predict_slope.pdf', dpi=300, bbox_inches='tight')

# Call the function to create the slope chart
create_slope_chart()
