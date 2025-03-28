import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
from get_stats import get_all_stats
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# Set the style to a more modern look
plt.style.use('seaborn-v0_8-whitegrid')

# Custom color palette - using different shades of blue
CUSTOM_COLORS = [
    (14/255, 140/255, 247/255),   # Bright blue
    (41/255, 44/255, 147/255),    # Deep blue
    (0/255, 84/255, 159/255),     # Navy blue
    (86/255, 180/255, 233/255),   # Sky blue
    (120/255, 180/255, 210/255),  # Darker light blue for mathquiz
    (0/255, 119/255, 182/255),    # Medium blue
    (65/255, 105/255, 225/255)    # Royal blue
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

# Function to load and resize logo
def get_logo(logo_path, size=0.15):
    try:
        pil_img = Image.open(logo_path)
        if pil_img.mode != 'RGBA':
            pil_img = pil_img.convert('RGBA')
        target_size = (100, 100)
        aspect = pil_img.width / pil_img.height
        if aspect > 1:
            new_width = target_size[0]
            new_height = int(new_width / aspect)
        else:
            new_height = target_size[1]
            new_width = int(new_height * aspect)
        new_img = Image.new('RGBA', target_size, (0, 0, 0, 0))
        resized_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        paste_x = (target_size[0] - new_width) // 2
        paste_y = (target_size[1] - new_height) // 2
        new_img.paste(resized_img, (paste_x, paste_y), resized_img if resized_img.mode == 'RGBA' else None)
        img_array = np.array(new_img)
        return OffsetImage(img_array, zoom=size)
    except Exception as e:
        print(f"Error loading logo {logo_path}: {e}")
        return None

# Map model names to their logo files
LOGO_DIR = "paper/logos"
LOGO_MAPPING = {
    "gpt-4o": os.path.join(LOGO_DIR, "openai.png"),
    "claude-3.7-sonnet": os.path.join(LOGO_DIR, "claude.png"),
    "claude-3.7-sonnet-thinking": os.path.join(LOGO_DIR, "claude.png"),
    "gemini-2.0-flash": os.path.join(LOGO_DIR, "gemini.png"),
    "llama-3.3-70b": os.path.join(LOGO_DIR, "llama.png"),
    "llama-3.1-405b": os.path.join(LOGO_DIR, "llama.png"),
    "llama-3.1-70b": os.path.join(LOGO_DIR, "llama.png"),
    "llama-3.1-8b": os.path.join(LOGO_DIR, "llama.png"),
    "deepseek-chat": os.path.join(LOGO_DIR, "deepseek.png"),
    "deepseek-r1": os.path.join(LOGO_DIR, "deepseek.png"),
    "qwen2.5-32b": os.path.join(LOGO_DIR, "qwen2.png"),
    "qwq-32b": os.path.join(LOGO_DIR, "qwen2.png"),
    "o3-mini-high": os.path.join(LOGO_DIR, "openai.png")
}

def plot_chess_performance(chess_data, output_dir):
    """Plot chess-specific performance metrics."""
    models = list(chess_data.keys())
    
    # Sort models by win rate
    win_rates = []
    for model in models:
        stats = chess_data[model]
        total_games = stats['wins_by_max_attempts'] + stats['loss_by_max_attempts']
        win_rate = stats['wins_by_max_attempts'] / total_games * 100 if total_games > 0 else 0
        win_rates.append((model, win_rate))
    
    sorted_models = [x[0] for x in sorted(win_rates, key=lambda x: x[1], reverse=True)]
    
    # Extract data for plotting
    wins_by_max_attempts = [chess_data[model]['wins_by_max_attempts'] for model in sorted_models]
    wins_by_checkmate = [chess_data[model].get('wins_by_checkmate', 0) for model in sorted_models]
    
    # Create the bar plot
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    
    # Set background color
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Plot wins
    x = np.arange(len(sorted_models))
    width = 0.35
    
    ax.bar(x - width/2, wins_by_max_attempts, width, label='Wins by Max Attempts', color=CUSTOM_COLORS[0])
    ax.bar(x + width/2, wins_by_checkmate, width, label='Wins by Checkmate', color=CUSTOM_COLORS[1])
    
    ax.set_xlabel('Models', fontsize=16, labelpad=15)
    ax.set_ylabel('Number of Wins', fontsize=16, labelpad=15)
    ax.set_title('Chess Performance: Wins by Type', fontsize=20, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_models, rotation=45, ha='right', fontsize=12)
    ax.legend(fontsize=12)
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='#333333', linestyle='-', linewidth=1, alpha=0.3)
    
    # Customize the grid
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#333333')
    ax.set_axisbelow(True)  # Put grid behind bars
    
    # Add a subtle box around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#DDDDDD')
        spine.set_linewidth(0.5)
    
    # Add logos on top of each bar
    for i, model in enumerate(sorted_models):
        if model in LOGO_MAPPING:
            logo_path = LOGO_MAPPING[model]
            logo_image = get_logo(logo_path, size=0.4)
            if logo_image:
                ab = AnnotationBbox(
                    logo_image, 
                    (i, max(wins_by_max_attempts[i], wins_by_checkmate[i])), 
                    xybox=(0, 40), 
                    xycoords='data',
                    boxcoords="offset points",
                    frameon=False
                )
                ax.add_artist(ab)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'chess_performance_wins.png'))
    plt.close()

def plot_gandalf_performance(gandalf_data, output_dir):
    """Plot Gandalf-specific performance metrics."""
    models = list(gandalf_data.keys())
    
    # Sort models by sentinel win rate
    sentinel_win_rates = []
    for model in models:
        stats = gandalf_data[model]
        total_games = stats['sentinel_wins'] + stats['infiltrator_wins']
        sentinel_win_rate = stats['sentinel_wins'] / total_games * 100 if total_games > 0 else 0
        sentinel_win_rates.append((model, sentinel_win_rate))
    
    sorted_models = [x[0] for x in sorted(sentinel_win_rates, key=lambda x: x[1], reverse=True)]
    
    # Extract data for plotting
    sentinel_wins = [gandalf_data[model]['sentinel_wins'] for model in sorted_models]
    infiltrator_wins = [gandalf_data[model]['infiltrator_wins'] for model in sorted_models]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    
    # Set background color
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    x = np.arange(len(sorted_models))
    width = 0.35
    
    ax.bar(x - width/2, sentinel_wins, width, label='Sentinel Wins', color=CUSTOM_COLORS[2])
    ax.bar(x + width/2, infiltrator_wins, width, label='Infiltrator Wins', color=CUSTOM_COLORS[3])
    
    ax.set_xlabel('Models', fontsize=16, labelpad=15)
    ax.set_ylabel('Number of Wins', fontsize=16, labelpad=15)
    ax.set_title('Gandalf Performance: Sentinel vs Infiltrator Wins', fontsize=20, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_models, rotation=45, ha='right', fontsize=12)
    ax.legend(fontsize=12)
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='#333333', linestyle='-', linewidth=1, alpha=0.3)
    
    # Customize the grid
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#333333')
    ax.set_axisbelow(True)  # Put grid behind bars
    
    # Add a subtle box around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#DDDDDD')
        spine.set_linewidth(0.5)
    
    # Add logos on top of each bar
    for i, model in enumerate(sorted_models):
        if model in LOGO_MAPPING:
            logo_path = LOGO_MAPPING[model]
            logo_image = get_logo(logo_path, size=0.4)
            if logo_image:
                ab = AnnotationBbox(
                    logo_image, 
                    (i, max(sentinel_wins[i], infiltrator_wins[i])), 
                    xybox=(0, 40), 
                    xycoords='data',
                    boxcoords="offset points",
                    frameon=False
                )
                ax.add_artist(ab)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gandalf_performance.png'))
    plt.close()

def plot_mathquiz_performance(mathquiz_data, output_dir):
    """Plot MathQuiz-specific performance metrics."""
    models = list(mathquiz_data.keys())
    
    # Sort models by student correct answer rate
    correct_answer_rates = []
    for model in models:
        stats = mathquiz_data[model]
        total_wins = stats['wins_by_student_correct_answer'] + stats['wins_by_verification_failed'] + stats['wins_by_student_incorrect_answer']
        correct_rate = stats['wins_by_student_correct_answer'] / total_wins * 100 if total_wins > 0 else 0
        correct_answer_rates.append((model, correct_rate))
    
    sorted_models = [x[0] for x in sorted(correct_answer_rates, key=lambda x: x[1], reverse=True)]
    
    # Extract data for plotting
    correct_answers = [mathquiz_data[model]['wins_by_student_correct_answer'] for model in sorted_models]
    verification_failed = [mathquiz_data[model]['wins_by_verification_failed'] for model in sorted_models]
    incorrect_answers = [mathquiz_data[model]['wins_by_student_incorrect_answer'] for model in sorted_models]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    
    # Set background color
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    x = np.arange(len(sorted_models))
    width = 0.25
    
    ax.bar(x - width, correct_answers, width, label='Correct Answers', color=CUSTOM_COLORS[4])
    ax.bar(x, verification_failed, width, label='Verification Failed', color=CUSTOM_COLORS[5])
    ax.bar(x + width, incorrect_answers, width, label='Incorrect Answers', color=CUSTOM_COLORS[6])
    
    ax.set_xlabel('Models', fontsize=16, labelpad=15)
    ax.set_ylabel('Number of Outcomes', fontsize=16, labelpad=15)
    ax.set_title('MathQuiz Performance: Answer Outcomes', fontsize=20, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_models, rotation=45, ha='right', fontsize=12)
    ax.legend(fontsize=12)
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='#333333', linestyle='-', linewidth=1, alpha=0.3)
    
    # Customize the grid
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#333333')
    ax.set_axisbelow(True)  # Put grid behind bars
    
    # Add a subtle box around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#DDDDDD')
        spine.set_linewidth(0.5)
    
    # Add logos on top of each bar
    for i, model in enumerate(sorted_models):
        if model in LOGO_MAPPING:
            logo_path = LOGO_MAPPING[model]
            logo_image = get_logo(logo_path, size=0.4)
            if logo_image:
                ab = AnnotationBbox(
                    logo_image, 
                    (i, max(correct_answers[i], verification_failed[i], incorrect_answers[i])), 
                    xybox=(0, 40), 
                    xycoords='data',
                    boxcoords="offset points",
                    frameon=False
                )
                ax.add_artist(ab)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mathquiz_performance.png'))
    plt.close()

def plot_poker_performance(poker_data, output_dir):
    """Plot Poker-specific performance metrics."""
    models = list(poker_data.keys())
    
    # Calculate average winning chip differences
    avg_chip_diffs = []
    for model in models:
        stats = poker_data[model]
        avg_diff = np.mean(stats['winning_chip_differences']) if stats['winning_chip_differences'] else 0
        avg_chip_diffs.append((model, avg_diff))
    
    sorted_models = [x[0] for x in sorted(avg_chip_diffs, key=lambda x: x[1], reverse=True)]
    
    # Extract data for plotting
    avg_chips = [np.mean(poker_data[model]['winning_chip_differences']) for model in sorted_models]
    max_chips = [max(poker_data[model]['winning_chip_differences']) if poker_data[model]['winning_chip_differences'] else 0 for model in sorted_models]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    
    # Set background color
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    x = np.arange(len(sorted_models))
    width = 0.35
    
    ax.bar(x - width/2, avg_chips, width, label='Average Winning Chips', color=CUSTOM_COLORS[0])
    ax.bar(x + width/2, max_chips, width, label='Maximum Winning Chips', color=CUSTOM_COLORS[1])
    
    ax.set_xlabel('Models', fontsize=16, labelpad=15)
    ax.set_ylabel('Chip Difference', fontsize=16, labelpad=15)
    ax.set_title('Poker Performance: Chip Differences', fontsize=20, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_models, rotation=45, ha='right', fontsize=12)
    ax.legend(fontsize=12)
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='#333333', linestyle='-', linewidth=1, alpha=0.3)
    
    # Customize the grid
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#333333')
    ax.set_axisbelow(True)  # Put grid behind bars
    
    # Add a subtle box around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#DDDDDD')
        spine.set_linewidth(0.5)
    
    # Add logos on top of each bar
    for i, model in enumerate(sorted_models):
        if model in LOGO_MAPPING:
            logo_path = LOGO_MAPPING[model]
            logo_image = get_logo(logo_path, size=0.4)
            if logo_image:
                ab = AnnotationBbox(
                    logo_image, 
                    (i, max(avg_chips[i], max_chips[i])), 
                    xybox=(0, 40), 
                    xycoords='data',
                    boxcoords="offset points",
                    frameon=False
                )
                ax.add_artist(ab)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'poker_performance.png'))
    plt.close()

def plot_model_comparison(games_data, output_dir):
    """Create a comprehensive comparison of models across all games."""
    # Collect all unique models
    all_models = set()
    for game_data in games_data.values():
        all_models.update(game_data.keys())
    
    all_models = list(all_models)
    
    # Calculate win rates for each model and game
    win_rates = {}
    for game, data in games_data.items():
        win_rates[game] = {}
        for model in all_models:
            if model in data and 'wins_by_max_attempts' in data[model] and 'loss_by_max_attempts' in data[model]:
                total_games = data[model]['wins_by_max_attempts'] + data[model]['loss_by_max_attempts']
                if total_games > 0:
                    win_rates[game][model] = data[model]['wins_by_max_attempts'] / total_games * 100
                else:
                    win_rates[game][model] = 0
            else:
                win_rates[game][model] = 0
    
    # Calculate average win rate across all games
    avg_win_rates = {}
    for model in all_models:
        rates = [win_rates[game][model] for game in games_data.keys() if model in win_rates[game]]
        avg_win_rates[model] = np.mean(rates) if rates else 0
    
    # Sort models by average win rate
    sorted_models = sorted(all_models, key=lambda x: avg_win_rates[x], reverse=True)
    
    # Create a heatmap of win rates
    win_rate_matrix = np.zeros((len(games_data), len(sorted_models)))
    
    for i, game in enumerate(games_data.keys()):
        for j, model in enumerate(sorted_models):
            win_rate_matrix[i, j] = win_rates[game].get(model, 0)
    
    plt.figure(figsize=(16, 10))
    plt.imshow(win_rate_matrix, cmap='viridis', aspect='auto')
    
    plt.colorbar(label='Win Rate (%)')
    plt.xlabel('Models')
    plt.ylabel('Games')
    plt.title('Win Rates Across Games and Models')
    
    plt.xticks(np.arange(len(sorted_models)), sorted_models, rotation=45, ha='right')
    plt.yticks(np.arange(len(games_data)), list(games_data.keys()))
    
    # Add win rate values to the heatmap
    for i in range(len(games_data)):
        for j in range(len(sorted_models)):
            plt.text(j, i, f"{win_rate_matrix[i, j]:.1f}%", 
                     ha="center", va="center", color="white" if win_rate_matrix[i, j] < 50 else "black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_heatmap.png'))
    plt.close()

def plot_chess_moves_histogram(chess_data, output_dir):
    """Plot histograms of the number of moves in chess games for each model."""
    models = list(chess_data.keys())
    
    # Determine the common x-axis range
    all_moves = [move for model in models for move in chess_data[model]['num_moves']]
    min_moves, max_moves = min(all_moves), max(all_moves)
    
    # Create subplots
    num_models = len(models)
    fig, axes = plt.subplots(nrows=num_models, ncols=1, figsize=(10, 2.5 * num_models), dpi=300)
    
    # Set background color
    fig.patch.set_facecolor('white')
    
    for ax, model in zip(axes, models):
        num_moves = chess_data[model]['num_moves']
        
        # Plot histogram
        ax.hist(num_moves, bins=20, range=(min_moves, max_moves), color=CUSTOM_COLORS[0], alpha=0.7)
        
        ax.set_title(f'{model} - Number of Moves', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Number of Moves', fontsize=12, labelpad=10)
        ax.set_ylabel('Frequency', fontsize=12, labelpad=10)
        
        # Set consistent x-axis limits
        ax.set_xlim(min_moves, max_moves)
        
        # Customize the grid
        ax.grid(axis='y', linestyle='--', alpha=0.3, color='#333333')
        ax.set_axisbelow(True)  # Put grid behind bars
        
        # Add a subtle box around the plot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#DDDDDD')
            spine.set_linewidth(0.5)
        
        # Add model logo to the background
        if model in LOGO_MAPPING:
            logo_path = LOGO_MAPPING[model]
            logo_image = get_logo(logo_path, size=0.3)
            if logo_image:
                ab = AnnotationBbox(
                    logo_image, 
                    (0.5, 0.5),  # Center of the subplot
                    xycoords='axes fraction',
                    boxcoords="axes fraction",
                    frameon=False,
                    zorder=-1  # Ensure the logo is behind the bars
                )
                ax.add_artist(ab)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'chess_moves_histogram.png'))
    plt.close()

def main():
    results_path = "/Users/haidark/Google Drive/My Drive/Zero Sum Eval/rankings-3-9-25"
    # Define output directory
    output_dir = "paper/figures/"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the imported stats directly
    games_data = get_all_stats(results_path)
    
    # Generate visualizations
    plot_chess_performance(games_data['chess'], output_dir)
    plot_gandalf_performance(games_data['gandalf'], output_dir)
    plot_mathquiz_performance(games_data['mathquiz'], output_dir)
    plot_poker_performance(games_data['poker'], output_dir)
    plot_model_comparison(games_data, output_dir)
    plot_chess_moves_histogram(games_data['chess'], output_dir)
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
