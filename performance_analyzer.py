import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob

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

def create_performance_graphs(suit_data, game_type):
    """Create graphs for each suit count with all metrics."""

    # Set up the plotting style
    plt.style.use('default')

    for suit_count, data in suit_data.items():
        fig, ax = plt.subplots(figsize=(12, 8))

        deck_sizes = np.array(data['deck_sizes'])

        # Sort data by deck size for better visualization
        sort_indices = np.argsort(deck_sizes)
        deck_sizes = deck_sizes[sort_indices]

        # Plot each metric with different styles for visibility
        ax.plot(deck_sizes, np.array(data['avg_options_per_turn'])[sort_indices],
                'o-', linewidth=2, markersize=6, label='Avg Options Per Turn', alpha=0.8)

        ax.plot(deck_sizes, np.array(data['min_options'])[sort_indices],
                's--', linewidth=2, markersize=5, label='Min Options', alpha=0.8)

        ax.plot(deck_sizes, np.array(data['max_options'])[sort_indices],
                '^-', linewidth=2, markersize=6, label='Max Options', alpha=0.8)

        ax.plot(deck_sizes, np.array(data['total_min_options'])[sort_indices],
                'v:', linewidth=2, markersize=5, label='Total Min Options', alpha=0.7)

        ax.plot(deck_sizes, np.array(data['total_max_options'])[sort_indices],
                'D-.', linewidth=2, markersize=5, label='Total Max Options', alpha=0.7)

        # Customize the plot
        ax.set_xlabel('Deck Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Option Count', fontsize=12, fontweight='bold')
        ax.set_title(f'{game_type.title()} Strategy Performance - {suit_count} Suits',
                     fontsize=14, fontweight='bold')

        # Add grid for better readability
        ax.grid(True, alpha=0.3)

        # Customize legend
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)

        # Set axis limits with some padding
        y_min = min(min(data['min_options']), min(data['total_min_options']))
        y_max = max(max(data['max_options']), max(data['total_max_options']))
        ax.set_ylim(y_min * 0.9, y_max * 1.1)

        # Improve layout
        plt.tight_layout()

        # Save the plot with game type prefix
        plt.savefig(f'{game_type}_performance_graph_suit_{suit_count}.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_continuity_chart(suit_data, game_type):
    """Create a continuity chart showing all suit counts together."""
    fig, ax = plt.subplots(figsize=(15, 10))

    colors = plt.cm.Set3(np.linspace(0, 1, len(suit_data)))

    for i, (suit_count, data) in enumerate(suit_data.items()):
        deck_sizes = np.array(data['deck_sizes'])
        color = colors[i]

        # Sort data by deck size for better visualization
        sort_indices = np.argsort(deck_sizes)
        deck_sizes = deck_sizes[sort_indices]

        # Plot with slight vertical offset for visibility
        offset = i * 0.5

        ax.plot(deck_sizes, np.array(data['avg_options_per_turn'])[sort_indices] + offset,
                'o-', color=color, linewidth=2, markersize=6,
                label=f'{suit_count} Suits - Avg Options', alpha=0.8)

        ax.fill_between(deck_sizes,
                        np.array(data['min_options'])[sort_indices] + offset,
                        np.array(data['max_options'])[sort_indices] + offset,
                        color=color, alpha=0.2)

    ax.set_xlabel('Deck Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Option Count (with vertical offset)', fontsize=12, fontweight='bold')
    ax.set_title(f'{game_type.title()} Strategy Performance Continuity Chart - All Suit Counts',
                 fontsize=14, fontweight='bold')

    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(f'{game_type}_performance_continuity_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary(suit_data, game_type):
    """Print a summary of the loaded data."""
    print("\n" + "="*60)
    print(f"{game_type.upper()} STRATEGY PERFORMANCE DATA SUMMARY")
    print("="*60)

    for suit_count, data in suit_data.items():
        print(f"\nSuit Count: {suit_count}")
        print(f"  Configurations: {len(data['deck_sizes'])}")
        print(f"  Deck Size Range: {min(data['deck_sizes'])} - {max(data['deck_sizes'])}")
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

    # Load the specific JSON file
    json_data = load_specific_json_file(file_path)

    if not json_data:
        print(f"Could not load {file_path}. Please check if the file exists.")
        return

    # Debug the structure
    debug_json_structure(json_data, file_path)

    # Extract performance data
    suit_data = extract_performance_data(json_data)

    if not suit_data:
        print(f"No valid performance data found in {file_path}.")
        print("Expected: Array of objects with 'suit_count', 'deck_size', 'avg_options_per_turn', etc.")
        return

    # Print summary
    print_summary(suit_data, game_type)

    # Create individual graphs for each suit count
    print(f"\nGenerating {game_type} individual performance graphs...")
    create_performance_graphs(suit_data, game_type)

    # Create continuity chart
    print(f"\nGenerating {game_type} continuity chart...")
    create_continuity_chart(suit_data, game_type)

    print(f"\nAll {game_type} graphs have been generated and saved!")

def main():
    """Main function to execute the script."""
    print("Starting dual game strategy performance analysis...")

    # Define the game files
    game_files = [
        ("sevens_strategy_performance_data.json", "sevens"),
        ("spades_strategy_performance_data.json", "spades")
    ]

    # Process each game's data
    for file_path, game_type in game_files:
        process_game_data(file_path, game_type)

    print("\n" + "="*60)
    print("ALL ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("Sevens Strategy:")
    print("  - sevens_performance_graph_suit_*.png")
    print("  - sevens_performance_continuity_chart.png")
    print("Spades Strategy:")
    print("  - spades_performance_graph_suit_*.png")
    print("  - spades_performance_continuity_chart.png")

if __name__ == "__main__":
    main()