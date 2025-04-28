import streamlit as st
import os
import time
import pandas as pd
import numpy as np
import re
from datetime import datetime
import string
from collections import defaultdict
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import uuid
import glob
import json

# Set page configuration
st.set_page_config(
    page_title="Chemical Name Extractor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .file-uploader {
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .result-container {
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Create directories for uploads and results
upload_dir = 'uploads'
results_dir = 'results'
for folder in [upload_dir, results_dir]:
    os.makedirs(folder, exist_ok=True)

# Core functions from the original code
def preprocess_name(name):
    """Preprocess chemical names for better matching"""
    if not isinstance(name, str) or pd.isna(name):
        return ""
    
    # Convert to lowercase
    name = name.lower()
    
    # Replace common punctuation that might interfere with matching
    name = name.replace('-', ' ').replace('/', ' ').replace(',', ' ')
    
    # Remove parentheses and their contents as they often contain non-essential information
    name = re.sub(r'\([^)]*\)', ' ', name)
    
    # Remove excessive whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

def create_efficient_lookup(chemical_df_list, chemical_df_old):
    """
    Create an efficient lookup structure for chemical names
    Returns three lookup structures:
    1. exact_lookup: For direct string matches
    2. token_lookup: For matching based on tokens/words in the name
    3. standard_names: Set of all standard chemical names
    """
    # Initialize the dictionaries
    exact_lookup = {}
    token_lookup = defaultdict(set)
    standard_names = set()
    
    # Combine both dataframes for processing
    combined_df = pd.concat([chemical_df_list, chemical_df_old], ignore_index=True)
    
    # Process all rows
    for _, row in combined_df.iterrows():
        standard_name = row['PRODUCTS']
        subname = row['SUBNAMES']
        
        if pd.notna(standard_name):
            standard_names.add(standard_name)
            
            # Add standard name to lookups
            standard_processed = preprocess_name(standard_name)
            exact_lookup[standard_name.lower()] = standard_name
            exact_lookup[standard_processed] = standard_name
            
            # Add tokens from standard name to token lookup
            if standard_processed:
                tokens = standard_processed.split()
                if len(tokens) > 1:  # Only add multi-word tokens to avoid too many false positives
                    for token in tokens:
                        if len(token) > 3:  # Only consider tokens longer than 3 chars
                            token_lookup[token].add(standard_name)
            
            # Process subnames
            if pd.notna(subname):
                subname_processed = preprocess_name(subname)
                
                # Add to exact lookup
                exact_lookup[subname.lower()] = standard_name
                exact_lookup[subname_processed] = standard_name
                
                # Add variations
                exact_lookup[subname.lower().replace(' ', '')] = standard_name  # Without spaces
                exact_lookup[subname.lower().replace(' ', '-')] = standard_name  # With hyphens
                
                # Add tokens from subname to token lookup
                if subname_processed:
                    tokens = subname_processed.split()
                    if len(tokens) > 1:  # Only add multi-word tokens to avoid too many false positives
                        for token in tokens:
                            if len(token) > 3:  # Only consider tokens longer than 3 chars
                                token_lookup[token].add(standard_name)
    
    return exact_lookup, token_lookup, standard_names

def extract_chemical_name_optimized(product_name, exact_lookup, token_lookup):
    """
    Optimized function to extract chemical names using pre-built lookup structures
    with special handling for Hexyl Acetate
    """
    if not isinstance(product_name, str) or pd.isna(product_name):
        return None, None
    
    # Preprocess product name
    product_processed = preprocess_name(product_name)
    if not product_processed:
        return None, None
    
    # Special case handling for Hexyl Acetate
    # Check if the string is exactly "Hexyl Acetate" (ignoring case, spaces)
    is_exact_hexyl_acetate = re.match(r'^\s*hexyl\s+acetate\s*$', product_processed, re.IGNORECASE) is not None
    
    # If it's not an exact match but contains "hexyl acetate" at the end, we'll flag it
    contains_hexyl_acetate_at_end = re.search(r'hexyl\s+acetate\s*$', product_processed, re.IGNORECASE) is not None
    
    # Handle the special case for Hexyl Acetate
    if contains_hexyl_acetate_at_end and not is_exact_hexyl_acetate:
        # If hexyl acetate is at the end of a longer name, don't match it as Hexyl Acetate
        # We'll continue with the lookup, but we'll exclude Hexyl Acetate from the possible matches
        hexyl_acetate_standard = None
        for key, value in exact_lookup.items():
            if value.lower() == "hexyl acetate":
                hexyl_acetate_standard = value
                break
    
    # Try direct exact matching first
    if product_processed in exact_lookup:
        proposed_match = exact_lookup[product_processed]
        
        # If it's not an exact "Hexyl Acetate" but contains it at the end, don't return Hexyl Acetate
        if contains_hexyl_acetate_at_end and not is_exact_hexyl_acetate and proposed_match.lower() == "hexyl acetate":
            pass  # Skip this match
        else:
            return proposed_match, "Exact"
    
    # Try matching without spaces or with hyphens
    no_spaces = product_processed.replace(' ', '')
    if no_spaces in exact_lookup:
        proposed_match = exact_lookup[no_spaces]
        
        # Same check for Hexyl Acetate
        if contains_hexyl_acetate_at_end and not is_exact_hexyl_acetate and proposed_match.lower() == "hexyl acetate":
            pass  # Skip this match
        else:
            return proposed_match, "Variation"
    
    with_hyphens = product_processed.replace(' ', '-')
    if with_hyphens in exact_lookup:
        proposed_match = exact_lookup[with_hyphens]
        
        # Same check for Hexyl Acetate
        if contains_hexyl_acetate_at_end and not is_exact_hexyl_acetate and proposed_match.lower() == "hexyl acetate":
            pass  # Skip this match
        else:
            return proposed_match, "Variation"
    
    # Try substring matching for longer product names
    if len(product_processed) > 10:
        for exact_key, standard_name in exact_lookup.items():
            if len(exact_key) > 5 and exact_key in product_processed:
                # Check if the standard name is Hexyl Acetate and we have our special case
                if contains_hexyl_acetate_at_end and not is_exact_hexyl_acetate and standard_name.lower() == "hexyl acetate":
                    continue  # Skip this match
                return standard_name, "Substring"
    
    # Try token-based matching
    tokens = product_processed.split()
    if len(tokens) > 0:
        # Count token matches for each chemical
        chemical_matches = defaultdict(int)
        for token in tokens:
            if len(token) > 3 and token in token_lookup:
                for chem in token_lookup[token]:
                    # Skip Hexyl Acetate in our special case
                    if contains_hexyl_acetate_at_end and not is_exact_hexyl_acetate and chem.lower() == "hexyl acetate":
                        continue
                    chemical_matches[chem] += 1
        
        # Find the chemical with the most token matches
        if chemical_matches:
            best_match = max(chemical_matches.items(), key=lambda x: x[1])
            # Only consider it a match if we have at least 2 token matches or 
            # the chemical name has very few tokens
            if best_match[1] >= 2:
                return best_match[0], "Token"
    
    return None, None

def process_chemical_data(file_path, progress_callback=None, status_callback=None):
    """Process a chemical data Excel file and extract standardized names"""
    if status_callback:
        status_callback(f"Reading Excel file: {file_path}")
    
    try:
        # Read all sheets from the Excel file
        product_df = pd.read_excel(file_path, sheet_name="Product", header=0)
        chemical_df_list = pd.read_excel(file_path, sheet_name="List", header=0)
        chemical_df_old = pd.read_excel(file_path, sheet_name="old", header=0)
        
        if status_callback:
            status_callback(f"Product sheet has {len(product_df)} rows")
            status_callback(f"List sheet has {len(chemical_df_list)} rows")
            status_callback(f"Old sheet has {len(chemical_df_old)} rows")
        
        # Create efficient lookup structures
        if status_callback:
            status_callback("Building efficient lookup structures...")
        
        exact_lookup, token_lookup, standard_names = create_efficient_lookup(chemical_df_list, chemical_df_old)
        
        if status_callback:
            status_callback(f"Created exact lookup with {len(exact_lookup)} entries")
            status_callback(f"Created token lookup with {len(token_lookup)} tokens")
            status_callback(f"Found {len(standard_names)} unique standard chemical names")
            status_callback(f"Processing {len(product_df)} product names...")
        
        # Create a list to store results
        results = []
        
        # Set up a simple progress counter
        total = len(product_df)
        processed = 0
        
        # Extract chemical names for all products
        for i, (_, row) in enumerate(product_df.iterrows()):
            product_name = row.iloc[0]
            chemical_name, match_type = extract_chemical_name_optimized(product_name, exact_lookup, token_lookup)
            
            if chemical_name:
                results.append({
                    'Product Name': product_name,
                    'Chemical Name': chemical_name,
                    'Match Type': match_type
                })
            
            # Update progress
            processed += 1
            if progress_callback:
                progress_callback(processed, total)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results) if results else pd.DataFrame(columns=['Product Name', 'Chemical Name', 'Match Type'])
        
        if status_callback:
            status_callback(f"Completed processing. Total matches found: {len(results_df)}")
        
        # Save the results to a new Excel file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.dirname(file_path)
        filename_base = os.path.basename(file_path).split('.')[0]
        output_filename = f"chemical_matches_{filename_base}_{timestamp}.xlsx"
        output_path = os.path.join(output_dir, output_filename)
        
        if not results_df.empty:
            results_df.to_excel(output_path, index=False)
            if status_callback:
                status_callback(f"Results saved to {output_path}")
            
            # Create analytics
            analytics = {}
            
            # Match type distribution
            match_type_counts = results_df['Match Type'].value_counts().to_dict()
            analytics['match_type_distribution'] = match_type_counts
            
            # Most frequent chemicals
            top_chemicals = results_df['Chemical Name'].value_counts().head(10).to_dict()
            analytics['top_chemicals'] = top_chemicals
            
            return True, output_path, results_df, analytics
        else:
            if status_callback:
                status_callback("No matches found, no output file created.")
            return False, None, None, None
    
    except Exception as e:
        if status_callback:
            status_callback(f"Error processing file {file_path}: {str(e)}")
        return False, None, None, None

# Chart creation functions
def create_match_type_chart(analytics):
    """Create a pie chart for match type distribution"""
    if 'match_type_distribution' not in analytics:
        return None
    
    data = analytics['match_type_distribution']
    labels = list(data.keys())
    values = list(data.values())
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values,
        hole=.4,
        marker_colors=['#2c3e50', '#3498db', '#1abc9c', '#f39c12']
    )])
    
    fig.update_layout(
        title_text='Match Type Distribution',
        showlegend=True,
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    
    return fig

def create_top_chemicals_chart(analytics):
    """Create a bar chart for top chemicals"""
    if 'top_chemicals' not in analytics:
        return None
    
    data = analytics['top_chemicals']
    chemicals = list(data.keys())
    counts = list(data.values())
    
    # Truncate long chemical names for better display
    chemicals = [chem[:30] + '...' if len(chem) > 30 else chem for chem in chemicals]
    
    fig = go.Figure(data=[go.Bar(
        x=counts,
        y=chemicals,
        orientation='h',
        marker_color='#3498db'
    )])
    
    fig.update_layout(
        title_text='Top 10 Most Frequent Chemicals',
        xaxis_title='Count',
        yaxis_title='Chemical Name',
        height=500,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    
    return fig

def batch_process_files(files, status_container):
    """Process multiple Excel files"""
    results_summary = []
    
    for idx, file in enumerate(files):
        file_path = save_uploaded_file(file)
        status_container.write(f"Processing file {idx+1}/{len(files)}: {file.name}")
        
        # Status updates will be sent to the Streamlit container
        def status_update(message):
            status_container.write(message)
        
        # Simple progress display
        progress_bar = st.progress(0)
        
        def progress_update(current, total):
            progress_bar.progress(current / total)
        
        success, output_path, results_df, analytics = process_chemical_data(
            file_path, 
            progress_callback=progress_update,
            status_callback=status_update
        )
        
        results_summary.append({
            'File': file.name,
            'Success': success,
            'Output': output_path if success else 'Failed',
            'Results': results_df,
            'Analytics': analytics
        })
        
        # Reset progress bar for next file
        progress_bar.empty()
    
    return results_summary

def save_uploaded_file(uploaded_file):
    """Save an uploaded file to disk and return the file path"""
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def extract_from_text(text, exact_lookup, token_lookup):
    """Extract chemical names from text input"""
    if not text:
        return []
    
    # Split text into potential product names (by line breaks and semicolons)
    product_names = []
    for line in text.split('\n'):
        for item in line.split(';'):
            if item.strip():
                product_names.append(item.strip())
    
    # Extract chemicals from each product name
    results = []
    for product_name in product_names:
        chemical_name, match_type = extract_chemical_name_optimized(product_name, exact_lookup, token_lookup)
        if chemical_name:
            results.append({
                'Product Name': product_name,
                'Chemical Name': chemical_name,
                'Match Type': match_type
            })
    
    return results

def main():
    """Main Streamlit application"""
    st.title("🧪 Chemical Name Extractor")
    
    # Add a sidebar with information
    st.sidebar.title("About")
    st.sidebar.info(
        "This tool helps standardize chemical names by matching product names "
        "against a database of known chemical names. Upload an Excel file with "
        "the required sheets or enter product names directly for matching."
    )
    
    st.sidebar.title("Features")
    st.sidebar.markdown(
        """
        - Standardize chemical names from product descriptions
        - Process single or multiple Excel files
        - Generate analytical reports
        - Extract chemicals from text input
        - Visualize match distributions and frequencies
        """
    )
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Process Files", "Quick Extract", "Batch Processing"])
    
    # TAB 1: Single File Processing
    with tab1:
        st.header("Process Excel File")
        st.markdown(
            """
            Upload an Excel file with the following sheets:
            - **Product**: Contains product names to standardize (first column)
            - **List**: Contains standard chemical names and their alternate names
            - **old**: Contains additional chemical name mappings
            """
        )
        
        uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"], key="single_file")
        
        if uploaded_file is not None:
            # Display file info
            st.write(f"File: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
            
            # Process button
            if st.button("Process File"):
                # Save the uploaded file
                file_path = save_uploaded_file(uploaded_file)
                
                # Create containers for status messages and progress
                status_container = st.empty()
                progress_container = st.empty()
                
                # Define callback functions for status updates and progress tracking
                def status_update(message):
                    status_container.write(message)
                
                progress_bar = st.progress(0)
                
                def progress_update(current, total):
                    progress_bar.progress(current / total)
                
                # Process the file
                try:
                    with st.spinner("Processing file..."):
                        success, output_path, results_df, analytics = process_chemical_data(
                            file_path, 
                            progress_callback=progress_update,
                            status_callback=status_update
                        )
                    
                    if success and results_df is not None:
                        # Display results in tabs
                        result_tab1, result_tab2, result_tab3 = st.tabs(["Results Table", "Match Types", "Top Chemicals"])
                        
                        with result_tab1:
                            st.write(f"Total matches: {len(results_df)}")
                            st.dataframe(results_df)
                            
                            # Create a download button for the results
                            csv = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Results (CSV)",
                                data=csv,
                                file_name=f"chemical_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                            )
                        
                        with result_tab2:
                            match_type_fig = create_match_type_chart(analytics)
                            if match_type_fig:
                                st.plotly_chart(match_type_fig, use_container_width=True)
                            else:
                                st.write("No match type data available.")
                        
                        with result_tab3:
                            top_chemicals_fig = create_top_chemicals_chart(analytics)
                            if top_chemicals_fig:
                                st.plotly_chart(top_chemicals_fig, use_container_width=True)
                            else:
                                st.write("No top chemicals data available.")
                    else:
                        st.error("No matches found or processing failed. Please check the status messages above.")
                
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    # TAB 2: Quick Text Extraction
    with tab2:
        st.header("Quick Extract from Text")
        st.markdown(
            """
            Enter product names to extract standard chemical names.
            First, upload a reference Excel file with standard chemical names.
            """
        )
        
        # Reference file upload
        ref_file = st.file_uploader("Upload reference Excel file with chemical names", type=["xlsx", "xls"], key="ref_file")
        
        if ref_file is not None:
            # Save the reference file
            ref_file_path = save_uploaded_file(ref_file)
            
            try:
                # Load the reference data
                chemical_df_list = pd.read_excel(ref_file_path, sheet_name="List", header=0)
                chemical_df_old = pd.read_excel(ref_file_path, sheet_name="old", header=0)
                
                # Create lookup structures
                exact_lookup, token_lookup, standard_names = create_efficient_lookup(chemical_df_list, chemical_df_old)
                
                st.success(f"Reference data loaded: {len(standard_names)} standard chemical names available.")
                
                # Text input area
                st.subheader("Enter Product Names")
                text_input = st.text_area(
                    "Enter product names (one per line or separated by semicolons):",
                    height=150,
                    placeholder="Enter product names here..."
                )
                
                if st.button("Extract Chemicals"):
                    if text_input:
                        with st.spinner("Extracting chemicals..."):
                            results = extract_from_text(text_input, exact_lookup, token_lookup)
                        
                        if results:
                            results_df = pd.DataFrame(results)
                            st.write(f"Found {len(results)} matches:")
                            st.dataframe(results_df)
                            
                            # Download option
                            csv = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Results (CSV)",
                                data=csv,
                                file_name=f"text_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                            )
                        else:
                            st.warning("No chemical names found in the provided text.")
                    else:
                        st.warning("Please enter some text to process.")
            
            except Exception as e:
                st.error(f"Error loading reference file: {str(e)}")
                st.info("Make sure the file has 'List' and 'old' sheets with the required format.")
    
    # TAB 3: Batch Processing
    with tab3:
        st.header("Batch Process Multiple Files")
        st.markdown(
            """
            Upload multiple Excel files for batch processing.
            Each file should have the required sheets (Product, List, old).
            """
        )
        
        uploaded_files = st.file_uploader(
            "Choose Excel files",
            type=["xlsx", "xls"],
            accept_multiple_files=True,
            key="batch_files"
        )
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} files")
            
            if st.button("Process All Files"):
                # Create a container for status messages
                status_container = st.empty()
                status_container.write("Starting batch processing...")
                
                # Process all files
                with st.spinner("Processing files..."):
                    results_summary = batch_process_files(uploaded_files, status_container)
                
                # Display summary results
                st.subheader("Processing Summary")
                summary_table = []
                for result in results_summary:
                    summary_table.append({
                        "File": result["File"],
                        "Status": "Success" if result["Success"] else "Failed",
                        "Matches Found": len(result["Results"]) if result["Results"] is not None else 0
                    })
                
                st.table(pd.DataFrame(summary_table))
                
                # Create consolidated results
                all_results = []
                for result in results_summary:
                    if result["Success"] and result["Results"] is not None:
                        # Add a source file column
                        result["Results"]["Source File"] = result["File"]
                        all_results.append(result["Results"])
                
                if all_results:
                    combined_results = pd.concat(all_results, ignore_index=True)
                    
                    st.subheader("Combined Results")
                    st.dataframe(combined_results)
                    
                    # Download combined results
                    csv = combined_results.to_excel(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Combined Results (CSV)",
                        data=csv,
                        file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/xlsx",
                    )

if __name__ == "__main__":
    main()
