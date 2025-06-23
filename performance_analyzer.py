import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import seaborn as sns

def load_specific_json_file(file_path):
    """Load a specific JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            print(f"Loaded: {file_path}")
            return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_performance_data(json_data):
    """Extract performance data organized by suit count from an array of StrategyPerformanceData."""
    suit_data = {}
    if json_data and isinstance(json_data, list):
        for record in json_data:
            suit_count = record['suit_count']
            if suit_count not in suit_data:
                suit_data[suit_count] = {
                    'deck_sizes': [],
                    'avg_options_per_turn': [],
                    'min_options': [],
                    'max_options': [],
                    'total_min_options': [],
                    'total_max_options': []
                }
            suit_data[suit_count]['deck_sizes'].append(record['deck_size'])
            suit_data[suit_count]['avg_options_per_turn'].append(record['avg_options_per_turn'])
            suit_data[suit_count]['min_options'].append(record['min_options'])
            suit_data[suit_count]['max_options'].append(record['max_options'])
            suit_data[suit_count]['total_min_options'].append(record['total_min_options'])
            suit_data[suit_count]['total_max_options'].append(record['total_max_options'])
    return suit_data

def get_deck_size_normalization_params():
    """Get deck size normalization parameters from the Rust configuration."""
    suit_counts = [4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 48, 48, 48, 48, 48, 64, 64, 64, 64, 64]
    suit_n_size = [13, 27, 55, 83, 111, 13, 27, 55, 83, 111, 13, 27, 55, 83, 111, 13, 27, 55, 83, 111, 13, 27, 55, 83, 111, 13, 27, 55, 83, 111]
    # Calculate deck sizes
    deck_sizes = [sc * sns for sc, sns in zip(suit_counts, suit_n_size)]
    # Group by suit count to find min/max for each
    suit_deck_ranges = {}
    for sc, ds in zip(suit_counts, deck_sizes):
        if sc not in suit_deck_ranges:
            suit_deck_ranges[sc] = {'min': ds, 'max': ds}
        else:
            suit_deck_ranges[sc]['min'] = min(suit_deck_ranges[sc]['min'], ds)
            suit_deck_ranges[sc]['max'] = max(suit_deck_ranges[sc]['max'], ds)
    return suit_deck_ranges

def normalize_data_with_deck_size(suit_data, method='log_minmax'):
    """
    Normalize the data with each suit normalized individually between 0 and 1,
    including deck size normalization based on the suit count ranges.
    For performance metrics, normalize between 0 and max (not min and max).
    """
    normalized_data = {}
    deck_size_ranges = get_deck_size_normalization_params()

    # Apply normalization per suit (individual normalization)
    for suit_count, data in suit_data.items():
        normalized_data[suit_count] = {}

        # Normalize deck sizes based on the suit count's min/max range
        deck_sizes = np.array(data['deck_sizes'])
        if suit_count in deck_size_ranges:
            deck_min = deck_size_ranges[suit_count]['min']
            deck_max = deck_size_ranges[suit_count]['max']
            if deck_max != deck_min:
                normalized_deck_sizes = (deck_sizes - deck_min) / (deck_max - deck_min)
            else:
                normalized_deck_sizes = np.zeros_like(deck_sizes)
        else:
            # Fallback to local normalization if suit count not in expected ranges
            if deck_sizes.max() != deck_sizes.min():
                normalized_deck_sizes = (deck_sizes - deck_sizes.min()) / (deck_sizes.max() - deck_sizes.min())
            else:
                normalized_deck_sizes = np.zeros_like(deck_sizes)

        normalized_data[suit_count]['deck_sizes'] = deck_sizes.tolist()
        normalized_data[suit_count]['normalized_deck_sizes'] = normalized_deck_sizes.tolist()

        # Normalize performance metrics between 0 and max (not min and max)
        for metric in ['avg_options_per_turn', 'min_options', 'max_options',
                       'total_min_options', 'total_max_options']:
            values = np.array(data[metric])

            if method == 'log_minmax':
                # Apply log transformation first, then normalize between 0 and max
                log_values = np.log(values + 1e-6)
                max_log_value = log_values.max()
                if max_log_value > log_values.min():  # Avoid division by zero
                    # Normalize between 0 and max (subtract min to start from 0, then divide by range)
                    normalized_values = (log_values - log_values.min()) / (max_log_value - log_values.min())
                else:
                    normalized_values = np.zeros_like(log_values)
            elif method == 'zero_to_max':
                # Direct normalization between 0 and max
                max_value = values.max()
                if max_value > 0:
                    normalized_values = values / max_value
                else:
                    normalized_values = np.zeros_like(values)
            else:  # 'minmax' or default
                # Standard min-max normalization (between 0 and 1 based on min and max)
                if values.max() != values.min():
                    normalized_values = (values - values.min()) / (values.max() - values.min())
                else:
                    normalized_values = np.zeros_like(values)

            normalized_data[suit_count][metric] = normalized_values.tolist()

    return normalized_data

def create_enhanced_heatmap_analysis(all_suit_data, game_type):
    """Create enhanced heatmaps with better visualization and statistics."""
    normalized_data = normalize_data_with_deck_size(all_suit_data, 'zero_to_max')

    # Only show the 3 main metrics we care about
    metrics = ['avg_options_per_turn', 'min_options', 'max_options']
    metric_titles = ['Avg Options Per Turn', 'Min Options', 'Max Options']

    # Create a figure with 1x3 layout
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Set up style
    plt.style.use('default')

    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[i]

        suit_counts = sorted(normalized_data.keys())
        deck_size_bins = np.linspace(0, 1, 16)  # 15 bins for good resolution
        deck_size_labels = [f'{x:.2f}' for x in deck_size_bins[:-1]]

        # Initialize matrix with zeros instead of NaN to avoid white cells
        matrix = np.zeros((len(suit_counts), len(deck_size_bins)-1))
        count_matrix = np.zeros((len(suit_counts), len(deck_size_bins)-1))

        # Fill matrix with averaged values for overlapping data points
        for j, suit_count in enumerate(suit_counts):
            data = normalized_data[suit_count]
            norm_deck_sizes = np.array(data['normalized_deck_sizes'])
            values = np.array(data[metric])

            for norm_deck_size, value in zip(norm_deck_sizes, values):
                bin_idx = np.digitize(norm_deck_size, deck_size_bins) - 1
                bin_idx = max(0, min(bin_idx, len(deck_size_bins)-2))

                if count_matrix[j, bin_idx] == 0:
                    matrix[j, bin_idx] = value
                    count_matrix[j, bin_idx] = 1
                else:
                    # Average multiple values in the same bin
                    matrix[j, bin_idx] = (matrix[j, bin_idx] * count_matrix[j, bin_idx] + value) / (count_matrix[j, bin_idx] + 1)
                    count_matrix[j, bin_idx] += 1

        # Handle bins with no data - interpolate from neighboring bins
        for j in range(len(suit_counts)):
            for k in range(len(deck_size_bins)-1):
                if count_matrix[j, k] == 0:
                    # Find nearest non-zero values for interpolation
                    left_val = None
                    right_val = None

                    # Look left
                    for left_k in range(k-1, -1, -1):
                        if count_matrix[j, left_k] > 0:
                            left_val = matrix[j, left_k]
                            break

                    # Look right
                    for right_k in range(k+1, len(deck_size_bins)-1):
                        if count_matrix[j, right_k] > 0:
                            right_val = matrix[j, right_k]
                            break

                    # Interpolate or use available value
                    if left_val is not None and right_val is not None:
                        matrix[j, k] = (left_val + right_val) / 2
                    elif left_val is not None:
                        matrix[j, k] = left_val
                    elif right_val is not None:
                        matrix[j, k] = right_val
                    # If no neighbors, it stays 0

        # Create heatmap
        im = ax.imshow(matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1,
                       interpolation='bilinear')

        # Customize ticks and labels
        ax.set_xticks(np.arange(0, len(deck_size_labels), 3))
        ax.set_xticklabels([deck_size_labels[i] for i in range(0, len(deck_size_labels), 3)],
                           rotation=45, ha='right')
        ax.set_yticks(range(len(suit_counts)))
        ax.set_yticklabels(suit_counts)

        # Add title and labels
        ax.set_title(f'{title}\n(Higher values = warmer colors)',
                     fontweight='bold', fontsize=14, pad=20)
        ax.set_xlabel('Normalized Deck Size [0-1]', fontweight='bold', fontsize=12)

        # Only add ylabel to the first subplot
        if i == 0:
            ax.set_ylabel('Suit Count', fontweight='bold', fontsize=12)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Value [0-1]', fontweight='bold', fontsize=10)

        # Add grid for better readability
        ax.set_xticks(np.arange(-0.5, len(deck_size_labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(suit_counts), 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.7)

        # Add text annotations for values (only show every other cell to avoid clutter)
        for row in range(len(suit_counts)):
            for col in range(0, len(deck_size_bins)-1, 2):  # Every other column
                if matrix[row, col] > 0:  # Only annotate non-zero values
                    ax.text(col, row, f'{matrix[row, col]:.2f}',
                            ha='center', va='center', color='black',
                            fontsize=8, fontweight='bold')

    # Add main title
    fig.suptitle(f'{game_type.title()} - Enhanced Heatmap Analysis\n'
                 f'Strategy Performance: Average Options, Min Options, Max Options',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(f'{game_type}_enhanced_heatmap_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_correlation_heatmap(all_suit_data, game_type):
    """Create a correlation heatmap between different metrics."""
    normalized_data = normalize_data_with_deck_size(all_suit_data, 'zero_to_max')

    # Prepare data for correlation analysis - only the metrics we care about
    all_data = []
    for suit_count, data in normalized_data.items():
        for i in range(len(data['normalized_deck_sizes'])):
            row = {
                'suit_count': suit_count,
                'normalized_deck_size': data['normalized_deck_sizes'][i],
                'avg_options_per_turn': data['avg_options_per_turn'][i],
                'min_options': data['min_options'][i],
                'max_options': data['max_options'][i]
            }
            all_data.append(row)

    df = pd.DataFrame(all_data)

    # Calculate correlation matrix
    correlation_matrix = df.corr()

    # Create figure
    plt.figure(figsize=(10, 8))

    # Create heatmap with seaborn
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                fmt='.2f', annot_kws={'fontsize': 12, 'fontweight': 'bold'})

    plt.title(f'{game_type.title()} - Metric Correlation Analysis\n'
              f'Correlation Between Performance Metrics',
              fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'{game_type}_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_statistical_summary_heatmap(all_suit_data, game_type):
    """Create heatmaps showing statistical summaries (mean, std, etc.) for each suit count."""
    normalized_data = normalize_data_with_deck_size(all_suit_data, 'zero_to_max')

    # Only the 3 metrics we care about
    metrics = ['avg_options_per_turn', 'min_options', 'max_options']
    stat_functions = {
        'Mean': np.mean,
        'Std Dev': np.std,
        'Min': np.min,
        'Max': np.max,
        'Median': np.median
    }

    suit_counts = sorted(normalized_data.keys())

    fig, axes = plt.subplots(len(stat_functions), len(metrics), figsize=(15, 18))

    for stat_idx, (stat_name, stat_func) in enumerate(stat_functions.items()):
        for metric_idx, metric in enumerate(metrics):
            ax = axes[stat_idx, metric_idx]

            # Create matrix for this statistic and metric
            matrix = np.zeros((len(suit_counts), 1))

            for suit_idx, suit_count in enumerate(suit_counts):
                values = np.array(normalized_data[suit_count][metric])
                matrix[suit_idx, 0] = stat_func(values)

            # Create heatmap
            im = ax.imshow(matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)

            # Customize
            ax.set_yticks(range(len(suit_counts)))
            ax.set_yticklabels(suit_counts)
            ax.set_xticks([])
            ax.set_title(f'{stat_name}\n{metric.replace("_", " ").title()}',
                         fontweight='bold', fontsize=12)

            # Add value annotations
            for row in range(len(suit_counts)):
                ax.text(0, row, f'{matrix[row, 0]:.3f}',
                        ha='center', va='center',
                        color='black',
                        fontweight='bold', fontsize=10)

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f'{game_type.title()} - Statistical Summary Heatmaps\n'
                 f'Per-Suit Statistics Across All Deck Sizes',
                 fontsize=16, fontweight='bold', y=0.95)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(f'{game_type}_statistical_summary_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary(suit_data, game_type):
    """Print a summary of the loaded data."""
    print("\n" + "="*60)
    print(f"{game_type.upper()} STRATEGY PERFORMANCE DATA SUMMARY")
    print("="*60)

    deck_size_ranges = get_deck_size_normalization_params()

    for suit_count, data in suit_data.items():
        print(f"\nSuit Count: {suit_count}")
        print(f"  Configurations: {len(data['deck_sizes'])}")
        print(f"  Deck Size Range: {min(data['deck_sizes'])} - {max(data['deck_sizes'])}")
        if suit_count in deck_size_ranges:
            print(f"  Expected Deck Size Range: {deck_size_ranges[suit_count]['min']} - {deck_size_ranges[suit_count]['max']}")
        print(f"  Avg Options Range: {min(data['avg_options_per_turn']):.2f} - {max(data['avg_options_per_turn']):.2f}")

def debug_json_structure(json_data, file_path):
    """Debug function to understand JSON structure."""
    print(f"\nDEBUG: Structure analysis for {file_path}")
    print(f"Type: {type(json_data)}")
    if isinstance(json_data, list):
        print(f"Array length: {len(json_data)}")
        if len(json_data) > 0:
            print(f"First element type: {type(json_data[0])}")
            if isinstance(json_data[0], dict):
                print(f"First element keys: {list(json_data[0].keys())}")
    elif isinstance(json_data, dict):
        print(f"Object keys: {list(json_data.keys())}")

def process_game_data(file_path, game_type):
    """Process data for a specific game type."""
    print(f"\n{'='*60}")
    print(f"PROCESSING {game_type.upper()} STRATEGY DATA")
    print('='*60)

    json_data = load_specific_json_file(file_path)
    if not json_data:
        print(f"Could not load {file_path}. Please check if the file exists.")
        return None

    debug_json_structure(json_data, file_path)
    suit_data = extract_performance_data(json_data)

    if not suit_data:
        print(f"No valid performance data found in {file_path}.")
        print("Expected: Array of objects with 'suit_count', 'deck_size', 'avg_options_per_turn', etc.")
        return None

    print_summary(suit_data, game_type)

    # Create enhanced heatmap visualizations
    print(f"\nGenerating {game_type} enhanced heatmap analysis...")
    create_enhanced_heatmap_analysis(suit_data, game_type)

    print(f"\nGenerating {game_type} correlation heatmap...")
    create_correlation_heatmap(suit_data, game_type)

    print(f"\nGenerating {game_type} statistical summary heatmaps...")
    create_statistical_summary_heatmap(suit_data, game_type)

    print(f"\nAll {game_type} heatmap visualizations have been generated and saved!")
    return suit_data

def main():
    """Main function to execute the script."""
    print("Starting enhanced heatmap analysis for strategy performance...")

    deck_ranges = get_deck_size_normalization_params()
    print("\nDeck Size Normalization Ranges:")
    for suit_count, ranges in deck_ranges.items():
        print(f"  {suit_count} suits: {ranges['min']} - {ranges['max']}")

    game_files = [
        ("sevens_strategy_performance_data.json", "sevens"),
        ("spades_strategy_performance_data.json", "spades")
    ]

    all_game_data = {}
    for file_path, game_type in game_files:
        suit_data = process_game_data(file_path, game_type)
        if suit_data:
            all_game_data[game_type] = suit_data

    print("\n" + "="*60)
    print("ALL HEATMAP ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    for game_type in all_game_data.keys():
        print(f"{game_type.title()} Strategy:")
        print(f"  - {game_type}_enhanced_heatmap_analysis.png")
        print(f"  - {game_type}_correlation_heatmap.png")
        print(f"  - {game_type}_statistical_summary_heatmaps.png")

if __name__ == "__main__":
    main()