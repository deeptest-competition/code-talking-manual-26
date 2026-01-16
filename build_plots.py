import json
import os
from typing import Dict, List
from collections import defaultdict
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from io import BytesIO
import zipfile


Results = Dict[str, Dict[str, List[float]]]


def read_results_folder(root: str) -> Results:
    results: Results = defaultdict(lambda: defaultdict(list))
    for seed in os.listdir(root):
        path = os.path.join(root, seed)
        for tool in os.listdir(path):
            tool_path = os.path.join(path, tool)
            for experiment in os.listdir(tool_path):
                file = os.path.join(tool_path, experiment, "evaluation_summary.json")
                if not os.path.exists(file):
                    continue
                with open(file, "r") as f:
                    data = json.load(f)
                    for metric, value in data.items():
                        results[tool][metric].append(value)

    return results


def create_box_plot(results: Results, metric: str):
    """Create a box and whisker plot for a specific metric across all tools."""
    fig = go.Figure()
    
    # Define a color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for idx, tool in enumerate(sorted(results.keys())):
        if metric in results[tool]:
            fig.add_trace(go.Box(
                y=results[tool][metric],
                name=tool,
                boxmean='sd',  # Show mean and standard deviation
                marker_color=colors[idx % len(colors)],
                line=dict(color=colors[idx % len(colors)])
            ))
    
    fig.update_layout(
        title=f"Tool Comparison: {metric}",
        yaxis_title=metric,
        xaxis_title="Tool",
        showlegend=False,
        height=500,
        template="plotly_white",
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black', size=12),
        margin=dict(l=60, r=40, t=60, b=60)
    )
    
    return fig


def create_zip_archive(results: Results, all_metrics: List[str]) -> BytesIO:
    """Create a zip archive containing all plots and raw data."""
    zip_buffer = BytesIO()
    
    print(f"Starting zip archive creation for {len(all_metrics)} metrics...")
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Save plots in multiple formats
        for metric in all_metrics:
            print(f"\nProcessing metric: {metric}")
            fig = create_box_plot(results, metric)
            
            # Save as HTML (interactive)
            print(f"  - Saving HTML...")
            html_content = fig.to_html()
            zip_file.writestr(f"plots/html/{metric}_plot.html", html_content.encode())
            print(f"  ✓ HTML saved")
            
            # Save as PNG (static image)
            try:
                print(f"  - Saving PNG...")
                png_bytes = fig.to_image(format="png", width=1200, height=800)
                zip_file.writestr(f"plots/png/{metric}_plot.png", png_bytes)
                print(f"  ✓ PNG saved")
            except Exception as e:
                print(f"  ✗ PNG failed: {e}")
            
            # Save as PDF (static, vector format)
            try:
                print(f"  - Saving PDF...")
                pdf_bytes = fig.to_image(format="pdf", width=1200, height=800)
                zip_file.writestr(f"plots/pdf/{metric}_plot.pdf", pdf_bytes)
                print(f"  ✓ PDF saved")
            except Exception as e:
                print(f"  ✗ PDF failed: {e}")
        
        # Save raw data as CSV
        print(f"\nSaving summary statistics CSV...")
        summary_data = []
        for tool in sorted(results.keys()):
            for metric in all_metrics:
                if metric in results[tool]:
                    values = results[tool][metric]
                    summary_data.append({
                        'Tool': tool,
                        'Metric': metric,
                        'Count': len(values),
                        'Mean': sum(values) / len(values) if values else 0,
                        'Min': min(values) if values else 0,
                        'Max': max(values) if values else 0
                    })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_buffer = BytesIO()
            df.to_csv(csv_buffer, index=False)
            zip_file.writestr("summary_statistics.csv", csv_buffer.getvalue())
            print(f"✓ Summary statistics CSV saved")
        
        # Save detailed raw data as JSON
        print(f"Saving raw data JSON...")
        detailed_data = {}
        for tool in sorted(results.keys()):
            detailed_data[tool] = {}
            for metric in all_metrics:
                if metric in results[tool]:
                    detailed_data[tool][metric] = results[tool][metric]
        
        json_content = json.dumps(detailed_data, indent=2)
        zip_file.writestr("raw_data.json", json_content)
        print(f"✓ Raw data JSON saved")
    
    print(f"\n✓ Zip archive creation complete!")
    zip_buffer.seek(0)
    return zip_buffer


def main():
    st.set_page_config(page_title="Tool Comparison Dashboard", layout="wide")
    st.title("Tool Comparison Dashboard")
    st.markdown("Box and whisker plots comparing different tools across all metrics")
    
    # Sidebar for folder selection
    st.sidebar.header("Settings")
    default_folder = "./result_ranking"
    results_folder = st.sidebar.text_input("Results Folder Path", value=default_folder)
    
    if not os.path.exists(results_folder):
        st.error(f"Folder '{results_folder}' does not exist!")
        st.info("Please enter a valid results folder path in the sidebar.")
        return
    
    # Load results
    with st.spinner("Loading results..."):
        results = read_results_folder(results_folder)
    
    if not results:
        st.warning("No results found in the specified folder.")
        return
    
    # Display summary
    st.success(f"Loaded results for {len(results)} tools")
    
    # Get all unique metrics
    all_metrics = set()
    for tool_data in results.values():
        all_metrics.update(tool_data.keys())
    all_metrics = sorted(all_metrics)
    
    # Download button for all plots and data
    st.sidebar.header("Export")
    if st.sidebar.button("📦 Generate Download Package", type="primary"):
        with st.spinner("Creating zip archive..."):
            zip_buffer = create_zip_archive(results, all_metrics)
        st.sidebar.download_button(
            label="⬇️ Download ZIP Archive",
            data=zip_buffer,
            file_name="tool_comparison_results.zip",
            mime="application/zip"
        )
        st.sidebar.success("✅ Package ready for download!")
    
    # Display tool information
    with st.expander("Tool Information", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Tools:**")
            for tool in sorted(results.keys()):
                st.write(f"- {tool}")
        with col2:
            st.write("**Metrics:**")
            for metric in all_metrics:
                st.write(f"- {metric}")
    
    # Create plots for all metrics
    st.header("Metric Comparisons")
    
    # Allow user to select which metrics to display
    selected_metrics = st.multiselect(
        "Select metrics to display",
        all_metrics,
        default=all_metrics
    )
    
    if not selected_metrics:
        st.info("Please select at least one metric to display.")
        return
    
    # Create plots in a grid layout
    cols_per_row = 2
    for i in range(0, len(selected_metrics), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(selected_metrics):
                metric = selected_metrics[i + j]
                with col:
                    fig = create_box_plot(results, metric)
                    st.plotly_chart(fig)
    
    # Show raw data table
    with st.expander("View Raw Data", expanded=False):
        st.subheader("Summary Statistics")
        
        # Create a summary table
        summary_data = []
        for tool in sorted(results.keys()):
            for metric in all_metrics:
                if metric in results[tool]:
                    values = results[tool][metric]
                    summary_data.append({
                        'Tool': tool,
                        'Metric': metric,
                        'Count': len(values),
                        'Mean': sum(values) / len(values) if values else 0,
                        'Min': min(values) if values else 0,
                        'Max': max(values) if values else 0
                    })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            st.dataframe(df)


if __name__ == "__main__":
    main()
