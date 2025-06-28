import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
import os

# Set style for better visualization
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

T0_opinion_example = "Helen, why don't you break down your new novel idea into smaller, manageable tasks? That way, you can gauge its potential and mitigate potential risks. What do you say?"
T10_opinion_example = "As a digital strategist and someone who is cautious, |I think Helen should consider the risks and consequences before taking action.| She should weigh the potential benefits of writing a novel that could make a big impact against the potential risks of failure. ······"
T80_opinion_example = ""
class GroupPolarizationVisualizer:
    """
    Visualizer for group polarization analysis results
    """
    
    def __init__(self, data_dir: str = "data/results/conservative_llama38b_aligned"):
        self.data_dir = data_dir
        self.timesteps = list(range(0, 90, 10))  # 0, 10, 20, ..., 80
        
    def load_polarization_data(self) -> Dict[int, Dict[str, int]]:
        """
        Load polarization data from output CSV files
        Returns: Dictionary mapping timestep to ranking counts
        """
        polarization_data = {}
        
        for timestep in self.timesteps:
            file_path = os.path.join(self.data_dir, f"output{timestep}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                ranking_counts = df['ranking'].value_counts().to_dict()
                polarization_data[timestep] = ranking_counts
            else:
                print(f"Warning: File {file_path} not found")
                
        return polarization_data
    
    def process_polarization_data(self, polarization_data: Dict[int, Dict[str, int]]) -> pd.DataFrame:
        """
        Process polarization data into a structured DataFrame
        """
        processed_data = []
        
        for timestep, counts in polarization_data.items():
            # Map rankings to categories
            conservative_count = counts.get('2, 1', 0)  # Answer2 more extreme
            progressive_count = counts.get('1, 2', 0)   # Answer2 more progressive  
            neutral_count = counts.get('same or wrong format', 0)  # No difference
            
            total = conservative_count + progressive_count + neutral_count
            
            if total > 0:
                processed_data.append({
                    'timestep': timestep,
                    'conservative': conservative_count,
                    'progressive': progressive_count,
                    'neutral': neutral_count,
                    'total': total,
                    'conservative_pct': conservative_count / total * 100,
                    'progressive_pct': progressive_count / total * 100,
                    'neutral_pct': neutral_count / total * 100
                })
        
        return pd.DataFrame(processed_data)
    
    def create_stacked_bar_chart(self, df: pd.DataFrame, title: str = "Group Polarization Analysis"):
        """
        Create a horizontal stacked bar chart with timestep as y-axis
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define colors similar to the reference image
        colors = {
            'conservative': '#FF6B6B',  # Red for more conservative
            'progressive': '#4ECDC4',   # Teal for more progressive  
            'neutral': '#95E1D3'        # Green for neutral/draw
        }
        
        # Create stacked bars
        timesteps = np.array(df['timestep'].values)
        conservative_counts = np.array(df['conservative'].values)
        progressive_counts = np.array(df['progressive'].values)
        neutral_counts = np.array(df['neutral'].values)
        
        # Create the horizontal stacked bars
        bars1 = ax.barh(timesteps, conservative_counts, 
                      color=colors['conservative'], 
                      label='(2, 1)', 
                      alpha=0.8)
        
        bars2 = ax.barh(timesteps, progressive_counts, 
                      left=conservative_counts,
                      color=colors['progressive'], 
                      label='(1, 2)', 
                      alpha=0.8)
        
        bars3 = ax.barh(timesteps, neutral_counts, 
                      left=conservative_counts + progressive_counts,
                      color=colors['neutral'], 
                      label='Neutral/Draw', 
                      alpha=0.8)
        
        # Customize the plot
        ax.set_ylabel('Timestep', fontsize=12)
        ax.set_xlabel('Number of Opinions', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (t, c, p, n) in enumerate(zip(timesteps, conservative_counts, progressive_counts, neutral_counts)):
            if c > 0:
                ax.text(c/2, t, str(c), ha='center', va='center', fontweight='bold')
            if p > 0:
                ax.text(c + p/2, t, str(p), ha='center', va='center', fontweight='bold')
            if n > 0:
                ax.text(c + p + n/2, t, str(n), ha='center', va='center', fontweight='bold')
        
        plt.tight_layout()
        return fig, ax
    
    def create_polarization_trend_chart(self, df: pd.DataFrame, title: str = "Polarization Trend Over Time"):
        """
        Create a line chart showing polarization trends over time
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate polarization index (conservative - progressive) / total
        df['polarization_index'] = (df['conservative'] - df['progressive']) / df['total']
        df['extreme_ratio'] = (df['conservative'] + df['progressive']) / df['total']
        
        # Plot trends
        ax.plot(df['timestep'], df['polarization_index'], 
                marker='o', linewidth=2, markersize=8, 
                label='Polarization Index (Conservative - Progressive)', 
                color='#FF6B6B')
        
        ax.plot(df['timestep'], df['extreme_ratio'], 
                marker='s', linewidth=2, markersize=8, 
                label='Extreme Opinion Ratio', 
                color='#4ECDC4')
        
        # Add horizontal line at y=0 for reference
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Customize the plot
        ax.set_xlabel('Timestep', fontsize=12)
        ax.set_ylabel('Ratio', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        for i, row in df.iterrows():
            ax.annotate(f'{row["polarization_index"]:.2f}', 
                       (row['timestep'], row['polarization_index']),
                       textcoords="offset points", 
                       xytext=(0,10), 
                       ha='center', fontsize=8)
        
        plt.tight_layout()
        return fig, ax
    
    def create_percentage_stacked_chart(self, df: pd.DataFrame, title: str = "Polarization Percentage Distribution"):
        """
        Create a horizontal percentage-based stacked bar chart with timestep as y-axis
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define colors
        colors = {
            'conservative': '#FF6B6B',
            'progressive': '#4ECDC4',
            'neutral': '#95E1D3'
        }
        
        timesteps = np.array(df['timestep'].values)
        conservative_pct = np.array(df['conservative_pct'].values)
        progressive_pct = np.array(df['progressive_pct'].values)
        neutral_pct = np.array(df['neutral_pct'].values)
        
        # Create horizontal stacked bars with percentages
        bars1 = ax.barh(timesteps, conservative_pct, 
                      color=colors['conservative'], 
                      label='(2, 1) (%)', 
                      alpha=0.8)
        
        bars2 = ax.barh(timesteps, progressive_pct, 
                      left=conservative_pct,
                      color=colors['progressive'], 
                      label='(1, 2) (%)', 
                      alpha=0.8)
        
        bars3 = ax.barh(timesteps, neutral_pct, 
                      left=conservative_pct + progressive_pct,
                      color=colors['neutral'], 
                      label='Neutral/Draw (%)', 
                      alpha=0.8)
        
        # Customize the plot
        ax.set_ylabel('Timestep', fontsize=12)
        ax.set_xlabel('Percentage (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add percentage labels
        for i, (t, c, p, n) in enumerate(zip(timesteps, conservative_pct, progressive_pct, neutral_pct)):
            if c > 5:  # Only show label if percentage is significant
                ax.text(c/2, t, f'{c:.1f}%', ha='center', va='center', fontweight='bold')
            if p > 5:
                ax.text(c + p/2, t, f'{p:.1f}%', ha='center', va='center', fontweight='bold')
            if n > 5:
                ax.text(c + p + n/2, t, f'{n:.1f}%', ha='center', va='center', fontweight='bold')
        
        plt.tight_layout()
        return fig, ax
        
    def generate_example_text_analysis(self, timestep: int = 0) -> str:
        """
        Generate example text analysis for a specific timestep
        """
        test_file = os.path.join(self.data_dir, f"test_{timestep}.csv")
        if os.path.exists(test_file):
            df = pd.read_csv(test_file)
            if len(df) > 0:
                # Get a sample response
                sample = df.iloc[0]
                return f"Timestep {timestep} Example:\n{sample['content'][:200]}..."
        return f"No example available for timestep {timestep}"
    
    def create_comprehensive_analysis(self, save_plots: bool = True):
        """
        Create comprehensive polarization analysis with all visualizations
        """
        print("Loading polarization data...")
        polarization_data = self.load_polarization_data()
        
        if not polarization_data:
            print("No data found. Please check the data directory.")
            return
        
        print("Processing data...")
        df = self.process_polarization_data(polarization_data)
        
        if df.empty:
            print("No valid data to process.")
            return
        
        print("\nPolarization Data Summary:")
        print(df.to_string(index=False))
        
        # Create visualizations
        print("\nCreating visualizations...")
        
        # 1. Stacked bar chart (count-based)
        fig1, ax1 = self.create_stacked_bar_chart(df, "Group Polarization Analysis - Absolute Counts")
        if save_plots:
            fig1.savefig('group_polarization_counts.png', dpi=300, bbox_inches='tight')
        
        # 2. Percentage stacked bar chart
        fig2, ax2 = self.create_percentage_stacked_chart(df, "Group Polarization Analysis - Percentage Distribution")
        if save_plots:
            fig2.savefig('group_polarization_percentages.png', dpi=300, bbox_inches='tight')
        
        # 3. Trend line chart
        fig3, ax3 = self.create_polarization_trend_chart(df, "Polarization Trend Over Time")
        if save_plots:
            fig3.savefig('group_polarization_trends.png', dpi=300, bbox_inches='tight')
        
        # Show examples
        print("\nExample responses:")
        for timestep in [0, 20, 40, 60, 80]:
            if timestep in polarization_data:
                example = self.generate_example_text_analysis(timestep)
                print(f"\n{example}")
        
        plt.show()
        
        return df, [fig1, fig2, fig3]

# Main execution
if __name__ == "__main__":
    # Create visualizer instance
    visualizer = GroupPolarizationVisualizer()
    
    # Generate comprehensive analysis
    results = visualizer.create_comprehensive_analysis(save_plots=True)
    
    if results:
        df, figures = results
        print(f"\nAnalysis complete! Generated {len(figures)} visualizations.")
        print("Charts saved as PNG files in the current directory.")