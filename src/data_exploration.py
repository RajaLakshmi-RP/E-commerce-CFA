import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Resolve paths relative to the project root
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "processed" / "renttherunway_clean.csv"

# Paths
DATA_PATH = Path("data/processed/renttherunway_clean.csv")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # Load the dataset
    df = pd.read_csv(DATA_PATH)

    # Basic info
    print(f"Dataset shape: {df.shape}")
    print("\nSentiment value counts (including NaN):")
    print(df['sentiment'].value_counts(dropna=False))

    # Plot distribution
    df['sentiment'].value_counts().plot(
        kind='bar',
        title='Sentiment Distribution',
        color='skyblue',
        edgecolor='black'
    )
    
    # Save the plot
    plt.savefig(PLOTS_DIR / "sentiment_distribution.png", dpi=300, bbox_inches='tight')
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    
print(f"Plot saved to: {PLOTS_DIR / 'sentiment_distribution.png'}")
