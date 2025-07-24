#!/usr/bin/env python3
"""
HypotheSAEs Correlation Analysis Visualization Dashboard - UMAP Focus
This script creates an interactive HTML dashboard focusing on UMAP visualization
"""

import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
from pathlib import Path
import argparse
import textwrap

class CorrelationVisualizationDashboard:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.data = {}
        self.load_data()
    
    def load_data(self):
        """Load all the output files from the correlation analysis"""
        print(f"Loading data from: {self.output_dir}")
        
        # Load main results with interpretations and examples
        results_path = self.output_dir / "correlation_analysis_results.csv"
        if results_path.exists():
            self.data['results'] = pd.read_csv(results_path)
            print(f"✓ Loaded {len(self.data['results'])} interpreted neurons")
        else:
            raise FileNotFoundError(f"Results file not found: {results_path}")
        
        # Load visualization data if available
        viz_path = self.output_dir / "neuron_visualization_data.csv"
        if viz_path.exists():
            self.data['visualization'] = pd.read_csv(viz_path)
            print("✓ Loaded UMAP visualization data")
        else:
            print("⚠ UMAP visualization data not found - you may need to run the analysis with visualization enabled")
    
    def create_interactive_umap(self):
        """Create interactive UMAP visualization with neuron details"""
        if 'visualization' not in self.data or 'results' not in self.data:
            return None, None
        
        viz_df = self.data['visualization']
        results_df = self.data['results']
        
        # Merge visualization data with interpretations and examples
        merged_df = viz_df.merge(
            results_df[['neuron_idx', 'target_correlation', 'interpretation', 'top_example_1', 'top_example_2', 'top_example_3']],
            on='neuron_idx',
            how='left'
        )
        
        # Create hover text with wrapped interpretations
        hover_texts = []
        for _, row in merged_df.iterrows():
            interpretation = row['interpretation'] if pd.notna(row['interpretation']) else 'No interpretation available'
            wrapped_interp = '<br>'.join(textwrap.wrap(interpretation, width=60))
            
            hover_text = f"""
            <b>Neuron {int(row['neuron_idx'])}</b><br>
            <b>SAE:</b> {int(row['sae_idx'])}<br>
            <b>Correlation:</b> {row['correlation']:.4f}<br>
            <b>Interpretation:</b><br>{wrapped_interp}
            """
            hover_texts.append(hover_text)
        
        # Create the UMAP plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=merged_df['umap_x'],
            y=merged_df['umap_y'],
            mode='markers',
            marker=dict(
                size=10,
                color=merged_df['correlation'],
                colorscale='RdBu_r',
                colorbar=dict(
                    title='Target<br>Correlation',
                    thickness=20,
                    len=0.7
                ),
                line=dict(width=0.5, color='black'),
                opacity=0.8
            ),
            text=hover_texts,
            hovertemplate='%{text}<extra></extra>',
            customdata=merged_df[['neuron_idx', 'interpretation', 'top_example_1', 'top_example_2', 'top_example_3']].values,
            name='Neurons'
        ))
        
        fig.update_layout(
            title=dict(
                text='Neuron UMAP Embedding - Click on neurons to see details',
                font=dict(size=20)
            ),
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            height=700,
            hovermode='closest',
            plot_bgcolor='white',
            xaxis=dict(
                gridcolor='lightgray',
                zeroline=True,
                zerolinecolor='gray'
            ),
            yaxis=dict(
                gridcolor='lightgray',
                zeroline=True,
                zerolinecolor='gray'
            )
        )
        
        return fig, merged_df
    
    def create_neuron_details_table(self, df):
        """Create HTML for scrollable neuron details table"""
        html = """
        <div id="neuron-details" style="height: 700px; overflow-y: auto; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h2 style="margin-top: 0; position: sticky; top: 0; background: white; padding: 10px 0; border-bottom: 2px solid #ecf0f1;">Neuron Details</h2>
            <p style="color: #666; margin-bottom: 20px;">Click on a neuron in the plot to highlight it here, or scroll through all neurons below.</p>
            <div id="selected-neuron" style="display: none; padding: 20px; background: #f0f8ff; border-radius: 8px; margin-bottom: 20px; border: 2px solid #3498db;">
                <!-- Selected neuron details will be inserted here -->
            </div>
            <h3>All Neurons</h3>
            <div id="all-neurons">
        """
        
        # Sort by absolute correlation
        df_sorted = df.sort_values('abs_correlation', ascending=False)
        
        for idx, row in df_sorted.iterrows():
            neuron_id = int(row['neuron_idx'])
            correlation = row['target_correlation']
            interpretation = row['interpretation'] if pd.notna(row['interpretation']) else 'No interpretation available'
            
            # Format examples
            examples_html = ""
            for i in range(1, 4):
                example_col = f'top_example_{i}'
                if example_col in row and pd.notna(row[example_col]):
                    example_text = str(row[example_col])[:500] + "..." if len(str(row[example_col])) > 500 else str(row[example_col])
                    examples_html += f"""
                    <div style="margin: 10px 0; padding: 10px; background: #f8f9fa; border-left: 3px solid #95a5a6; font-size: 0.9em;">
                        <strong>Example {i}:</strong><br>
                        <span style="color: #555;">{example_text}</span>
                    </div>
                    """
            
            correlation_color = "#c0392b" if correlation > 0 else "#2980b9"
            
            html += f"""
            <div class="neuron-entry" id="neuron-{neuron_id}" style="margin: 20px 0; padding: 20px; background: #f5f5f5; border-radius: 8px; border: 1px solid #ddd;">
                <h4 style="margin-top: 0; color: #2c3e50;">
                    Neuron {neuron_id} 
                    <span style="color: {correlation_color}; margin-left: 10px;">
                        (Correlation: {correlation:.4f})
                    </span>
                </h4>
                <p style="font-style: italic; color: #555; margin: 10px 0;">
                    <strong>Interpretation:</strong> {interpretation}
                </p>
                <details style="margin-top: 10px;">
                    <summary style="cursor: pointer; color: #3498db; font-weight: bold;">View Activating Examples</summary>
                    {examples_html}
                </details>
            </div>
            """
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def generate_html(self):
        """Generate the complete HTML dashboard"""
        fig, merged_df = self.create_interactive_umap()
        
        if fig is None:
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>UMAP Visualization Not Available</title>
            </head>
            <body>
                <h1>UMAP visualization data not available</h1>
                <p>Please run the correlation analysis with visualization enabled.</p>
            </body>
            </html>
            """
        
        umap_html = fig.to_html(full_html=False, include_plotlyjs='cdn', div_id="umap-plot")
        details_html = self.create_neuron_details_table(merged_df)
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>HypotheSAEs UMAP Visualization</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                    color: #333;
                }}
                .container {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    padding: 20px;
                    max-width: 1800px;
                    margin: 0 auto;
                }}
                @media (max-width: 1200px) {{
                    .container {{
                        grid-template-columns: 1fr;
                    }}
                }}
                .plot-section {{
                    background: white;
                    border-radius: 8px;
                    padding: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .highlight {{
                    background: #fff3cd !important;
                    border: 2px solid #ffc107 !important;
                    transition: all 0.3s ease;
                }}
                header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px 0;
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                header h1 {{
                    margin: 0;
                    font-size: 2em;
                }}
            </style>
        </head>
        <body>
            <header>
                <h1>HypotheSAEs UMAP Visualization</h1>
                <p>Interactive exploration of neuron activations and interpretations</p>
            </header>
            
            <div class="container">
                <div class="plot-section">
                    {umap_html}
                </div>
                
                {details_html}
            </div>
            
            <script>
                // Get the plot element
                var plot = document.getElementById('umap-plot');
                
                // Add click event listener to the plot
                plot.on('plotly_click', function(data) {{
                    var point = data.points[0];
                    var neuronIdx = point.customdata[0];
                    var interpretation = point.customdata[1];
                    var example1 = point.customdata[2];
                    var example2 = point.customdata[3];
                    var example3 = point.customdata[4];
                    
                    // Remove previous highlights
                    var allEntries = document.querySelectorAll('.neuron-entry');
                    allEntries.forEach(function(entry) {{
                        entry.classList.remove('highlight');
                    }});
                    
                    // Highlight the clicked neuron
                    var neuronEntry = document.getElementById('neuron-' + neuronIdx);
                    if (neuronEntry) {{
                        neuronEntry.classList.add('highlight');
                        neuronEntry.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                    }}
                    
                    // Update selected neuron display
                    var selectedDiv = document.getElementById('selected-neuron');
                    var correlationColor = point.marker.color > 0 ? '#c0392b' : '#2980b9';
                    
                    var examplesHtml = '';
                    if (example1 && example1 !== 'null') {{
                        examplesHtml += '<div style="margin: 10px 0; padding: 10px; background: #f8f9fa; border-left: 3px solid #95a5a6; font-size: 0.9em;">';
                        examplesHtml += '<strong>Example 1:</strong><br>';
                        examplesHtml += '<span style="color: #555;">' + (example1.length > 500 ? example1.substring(0, 500) + '...' : example1) + '</span></div>';
                    }}
                    if (example2 && example2 !== 'null') {{
                        examplesHtml += '<div style="margin: 10px 0; padding: 10px; background: #f8f9fa; border-left: 3px solid #95a5a6; font-size: 0.9em;">';
                        examplesHtml += '<strong>Example 2:</strong><br>';
                        examplesHtml += '<span style="color: #555;">' + (example2.length > 500 ? example2.substring(0, 500) + '...' : example2) + '</span></div>';
                    }}
                    if (example3 && example3 !== 'null') {{
                        examplesHtml += '<div style="margin: 10px 0; padding: 10px; background: #f8f9fa; border-left: 3px solid #95a5a6; font-size: 0.9em;">';
                        examplesHtml += '<strong>Example 3:</strong><br>';
                        examplesHtml += '<span style="color: #555;">' + (example3.length > 500 ? example3.substring(0, 500) + '...' : example3) + '</span></div>';
                    }}
                    
                    selectedDiv.innerHTML = `
                        <h3 style="margin-top: 0; color: #2c3e50;">
                            Selected: Neuron ${{neuronIdx}}
                            <span style="color: ${{correlationColor}}; margin-left: 10px;">
                                (Correlation: ${{point.marker.color.toFixed(4)}})
                            </span>
                        </h3>
                        <p style="font-style: italic; color: #555;">
                            <strong>Interpretation:</strong> ${{interpretation || 'No interpretation available'}}
                        </p>
                        <details open style="margin-top: 10px;">
                            <summary style="cursor: pointer; color: #3498db; font-weight: bold;">Activating Examples</summary>
                            ${{examplesHtml || '<p style="color: #999;">No examples available</p>'}}
                        </details>
                    `;
                    selectedDiv.style.display = 'block';
                }});
            </script>
        </body>
        </html>
        """
        
        return html
    
    def save_and_open(self, output_file='umap_dashboard.html'):
        """Save the HTML dashboard and open it in browser"""
        output_path = self.output_dir / output_file
        
        html = self.generate_html()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✓ Dashboard saved to: {output_path}")
        
        # Open in browser
        webbrowser.open(f'file://{output_path.absolute()}')
        print("✓ Opening dashboard in browser...")


def main():
    parser = argparse.ArgumentParser(description='Generate UMAP visualization dashboard for HypotheSAEs correlation analysis')
    parser.add_argument('output_dir', nargs='?', default=None,
                       help='Path to the output directory containing analysis results')
    parser.add_argument('--output-file', default='umap_dashboard.html',
                       help='Output HTML filename (default: umap_dashboard.html)')
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Set default output directory
        output_dir = '/Users/ivanculo/Desktop/Projects/SAEs/HypotheSAEs/outputs/Multiple_SAEs_depressiontext-embedding-3-small'
    
    if not os.path.exists(output_dir):
        print(f"Error: Output directory not found: {output_dir}")
        print("Please provide the correct output directory path.")
        return
    
    # Create and run dashboard
    dashboard = CorrelationVisualizationDashboard(output_dir)
    dashboard.save_and_open(args.output_file)


if __name__ == "__main__":
    main()