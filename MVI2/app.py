import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import time
import os
import json
import google.generativeai as genai
import re

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import scipy.stats as stats

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configure Google AI Client
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    embed_model = genai.embed_content
# Connect to Pinecone
if PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "dataset-imputation"

# Create Pinecone index if not exists
def setup_pinecone():
    if not PINECONE_API_KEY:
        st.error("❌ Pinecone API key not found!")
        return None
    
    existing_indexes = pc.list_indexes().names()
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=768,  # dimension for text-embedding-004 = 768
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        # wait for index creation
        while True:
            description = pc.describe_index(index_name)
            if description.status['ready']:
                break
            time.sleep(1)
    return pc.Index(index_name)

# Initialize Pinecone index
try:
    if PINECONE_API_KEY:
        index = setup_pinecone()
    else:
        index = None
except Exception as e:
    st.error(f"❌ Error setting up Pinecone: {e}")
    index = None


gemini_model = genai.GenerativeModel('gemini-2.0-flash-lite')
embed_model = genai.GenerativeModel('models/text-embedding-004')

def embed_text(text_list):
    vectors = []
    for text in text_list:
        try:
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="semantic_similarity"
            )
            vectors.append(response["embedding"])
        except Exception as e:
            st.error(f"Embedding Error: {e}")
            vectors.append([0.0] * 768)  # fallback vector
    return vectors

def store_vectors(ids, vectors, metadata_list):
    if not index:
        st.error("❌ Pinecone index not available!")
        return
    
    vectors_to_upsert = [(str(id_), vector, meta) for id_, vector, meta in zip(ids, vectors, metadata_list)]
    index.upsert(vectors=vectors_to_upsert)

def query_similar_vectors(vector, top_k=5):
    if not index:
        st.error("❌ Pinecone index not available!")
        return []
    
    try:
        response = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True
        )
        return response['matches']
    except Exception as e:
        st.error(f"Pinecone Query Error: {e}")
        return []

def convert_row_to_text(row):
    text = []
    for col, val in row.items():
        if pd.isnull(val):
            text.append(f"{col}: [MISSING]")
        else:
            text.append(f"{col}: {val}")
    return ", ".join(text)

def generate_embeddings_for_complete_rows(df):
    complete_rows = df.dropna()
    texts = [convert_row_to_text(row) for idx, row in complete_rows.iterrows()]
    ids = [f"row-{idx}" for idx in complete_rows.index]
    vectors = embed_text(texts)
    metadata_list = [{"text": text} for text in texts]
    store_vectors(ids, vectors, metadata_list)
    return complete_rows.index.tolist()

def extract_json(text):
    """Extract and clean JSON dictionary from AI response"""
    try:
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            return None
        json_str = json_match.group(0).strip()
        return json.loads(json_str)
    except Exception as e:
        return None

def fill_missing_values(df, batch_size=5, model_choice="Gemini"):
    filled_df = df.copy()
    rows_to_fill = []

    for idx, row in filled_df.iterrows():
        if row.isnull().sum() == 0:
            continue
        text = convert_row_to_text(row)
        rows_to_fill.append((idx, text))

    for batch_start in range(0, len(rows_to_fill), batch_size):
        batch = rows_to_fill[batch_start:batch_start+batch_size]
        texts = [text for _, text in batch]
        embeddings = embed_text(texts)

        for (idx, incomplete_text), embed_vector in zip(batch, embeddings):
            if all(val == 0.0 for val in embed_vector):
                st.warning(f"⚠️ Skipping row {idx} due to invalid embedding.")
                continue

            similar_rows = query_similar_vectors(embed_vector, top_k=5)
            if not similar_rows:
                st.warning(f"⚠️ No similar rows found for row {idx}.")
                continue

            similar_row = similar_rows[0]['metadata']['text']

            prompt = f"""
You are a helpful AI assistant helping to impute missing values in structured data.

Given a complete example row:
{similar_row}

And an incomplete row (with [MISSING] tokens):
{incomplete_text}

Predict only the missing values. 

Return your answer strictly as a JSON dictionary.
Example output:
{{"Column1": "value1", "Column2": "value2", "Column3": "value3"}}

Do not add anything else except the JSON output.
"""
            generated_text = None
            if model_choice == "Gemini":
                if not GOOGLE_API_KEY:
                    st.error("❌ Google API key required for Gemini model!")
                    continue
                
                for attempt in range(3):
                    try:
                        response = gemini_model.generate_content(prompt)
                        generated_text = response.text.strip()
                        break
                    except Exception as e:
                        if "429" in str(e):
                            st.warning("Cooling down for 60 seconds...")
                            time.sleep(60)
                        else:
                            st.error(f"❌ Gemini Error at row {idx}: {e}")
                            break

                    if not generated_text:
                        continue
            
            elif model_choice == "Groq":
                if not GROQ_API_KEY:
                    st.error("❌ Groq API key required for Groq model!")
                    continue
                
                try:
                    from groq import Groq
                    groq_client = Groq(api_key=GROQ_API_KEY)
                    completion = groq_client.chat.completions.create(
                        model="llama3-8b-8192",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that returns valid JSON only."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=1024,
                        top_p=1,
                        stream=False
                    )
                    generated_text = completion.choices[0].message.content.strip()
                except Exception as e:
                    st.error(f"❌ Groq Error at row {idx}: {e}")

            if not generated_text:
                continue
            
            predicted_dict = extract_json(generated_text)
            if predicted_dict is None:
                st.warning(f"⚠️ Parsing Error at row {idx}: Could not extract JSON.")
                continue

            row = filled_df.loc[idx]
            for column, value in predicted_dict.items():
                if pd.isnull(row[column]):
                    row[column] = value

            filled_df.loc[idx] = row

    return filled_df



def compare_correlation_matrices(original_df, imputed_df):
    """Compare correlation matrices between original and imputed datasets"""
    
    # Convert to numeric where possible
    original_numeric = original_df.copy()
    imputed_numeric = imputed_df.copy()
    
    for col in original_df.columns:
        try:
            original_numeric[col] = pd.to_numeric(original_numeric[col], errors='coerce')
            imputed_numeric[col] = pd.to_numeric(imputed_numeric[col], errors='coerce')
        except Exception:
            continue
    
    # Get only numeric columns
    numeric_cols = original_numeric.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        st.warning("⚠️ Need at least 2 numeric columns for correlation analysis.")
        return None, None, None
    
    # Calculate correlation matrices
    original_corr = original_numeric[numeric_cols].corr()
    imputed_corr = imputed_numeric[numeric_cols].corr()
    
    # Calculate correlation matrix difference
    corr_diff = imputed_corr - original_corr
    
    return original_corr, imputed_corr, corr_diff

def plot_correlation_comparison(original_corr, imputed_corr, corr_diff):
    """Plot correlation matrix comparison"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Original correlation matrix
    sns.heatmap(original_corr, annot=True, cmap='coolwarm', center=0, 
                ax=axes[0], fmt='.2f', square=True)
    axes[0].set_title('Original Data Correlation Matrix', fontsize=14, fontweight='bold')
    
    # Imputed correlation matrix
    sns.heatmap(imputed_corr, annot=True, cmap='coolwarm', center=0, 
                ax=axes[1], fmt='.2f', square=True)
    axes[1].set_title('Imputed Data Correlation Matrix', fontsize=14, fontweight='bold')
    
    # Difference in correlation matrices
    sns.heatmap(corr_diff, annot=True, cmap='RdBu_r', center=0, 
                ax=axes[2], fmt='.3f', square=True)
    axes[2].set_title('Correlation Difference (Imputed - Original)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def calculate_correlation_metrics(original_corr, imputed_corr):
    """Calculate correlation preservation metrics"""
    
    # Flatten correlation matrices (excluding diagonal)
    mask = np.triu(np.ones_like(original_corr), k=1).astype(bool)
    original_flat = original_corr.values[mask]
    imputed_flat = imputed_corr.values[mask]
    
    # Remove NaN values
    valid_mask = ~(np.isnan(original_flat) | np.isnan(imputed_flat))
    original_flat = original_flat[valid_mask]
    imputed_flat = imputed_flat[valid_mask]
    
    if len(original_flat) == 0:
        return None
    
    # Calculate metrics
    metrics = {
        'correlation_preservation': np.corrcoef(original_flat, imputed_flat)[0, 1],
        'mean_absolute_difference': np.mean(np.abs(original_flat - imputed_flat)),
        'max_absolute_difference': np.max(np.abs(original_flat - imputed_flat)),
        'rmse': np.sqrt(np.mean((original_flat - imputed_flat) ** 2))
    }
    
    return metrics


def compare_distributions_streamlit(original_df, filled_df):
    # Convert numeric columns properly
    original_df = original_df.copy()
    filled_df = filled_df.copy()

    # Try to convert all columns to numeric where possible
    for col in original_df.columns:
        try:
            original_df[col] = pd.to_numeric(original_df[col])
            filled_df[col] = pd.to_numeric(filled_df[col])
        except Exception:
            continue  # skip non-numeric columns

    numeric_cols = original_df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        st.markdown(f"### 🔍 Column: `{col}`")

        fig, ax = plt.subplots()
        original_df[col].plot(kind='kde', label='Original (with NaNs)', ax=ax)
        filled_df[col].plot(kind='kde', label='Imputed', ax=ax)
        plt.legend()
        st.pyplot(fig)

        # KS Test
        ks_stat, ks_p = stats.ks_2samp(
            original_df[col].dropna(),
            filled_df[col].dropna()
        )
        st.write(f"📌 KS Test Statistic: `{ks_stat:.4f}`, p-value: `{ks_p:.4f}`")

        # KL Divergence
        def safe_kl(p, q):
            p += 1e-10
            q += 1e-10
            return stats.entropy(p, q)

        # Normalize histograms
        hist_orig, bins = np.histogram(original_df[col].dropna(), bins=50, density=True)
        hist_imputed, _ = np.histogram(filled_df[col].dropna(), bins=bins, density=True)
        kl_div = safe_kl(hist_orig, hist_imputed)
        st.write(f"📌 KL Divergence: `{kl_div:.4f}`")

        st.markdown("---")

# Set page configuration
st.set_page_config(
    page_title="🧠 AI-Powered Data Imputation",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠Context-Aware MV Imputer</h1>', unsafe_allow_html=True)
    st.markdown("**Leverage AI to intelligently fill missing values in your datasets using semantic similarity and advanced language models.**")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model Selection
        model_options = ["Gemini 2.0 Flash", "Groq LLaMA3"]
        model_choice = st.selectbox("🤖 Choose AI Model", model_options, index=0)
        
        # Advanced Settings
        st.subheader("🔧 Advanced Settings")
        batch_size = st.slider("Batch Size", min_value=1, max_value=20, value=5, 
                              help="Number of rows to process simultaneously")
        
        similarity_threshold = st.slider("Similarity Threshold", min_value=0.1, max_value=1.0, value=0.7,
                                       help="Minimum similarity score for reference rows")
        
        top_k_similar = st.slider("Top K Similar Rows", min_value=1, max_value=10, value=5,
                                help="Number of similar rows to consider")
        
        # API Status Check
        st.subheader("🔌 API Status")
        if st.button("Check API Connections"):
            check_api_status()

    # Main Interface
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload Data", "🔍 Data Analysis", "🚀 Imputation", "📊 Results"])
    
    with tab1:
        upload_data_section()
    
    with tab2:
        if 'original_df' in st.session_state:
            data_analysis_section()
        else:
            st.info("👆 Please upload a dataset first in the 'Upload Data' tab.")
    
    with tab3:
        if 'original_df' in st.session_state:
            imputation_section(model_choice, batch_size, top_k_similar)
        else:
            st.info("👆 Please upload a dataset first in the 'Upload Data' tab.")
    
    with tab4:
        if 'imputed_df' in st.session_state:
            results_section()
        else:
            st.info("👆 Please complete the imputation process first.")

def upload_data_section():
    st.markdown('<h2 class="sub-header">📁 Upload Your Dataset</h2>', unsafe_allow_html=True)
    
    # File upload options
    upload_method = st.radio("Choose upload method:", ["Upload CSV file", "Use sample data", "Paste CSV data"])
    
    if upload_method == "Upload CSV file":
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.original_df = df
                st.session_state.filename = uploaded_file.name
                
                st.markdown('<div class="success-box">✅ File uploaded successfully!</div>', unsafe_allow_html=True)
                display_dataset_preview(df)
                
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    elif upload_method == "Use sample data":
        if st.button("Generate Sample Dataset"):
            df = generate_sample_data()
            st.session_state.original_df = df
            st.session_state.filename = "sample_data.csv"
            
            st.markdown('<div class="success-box">✅ Sample dataset generated!</div>', unsafe_allow_html=True)
            display_dataset_preview(df)
    
    elif upload_method == "Paste CSV data":
        csv_text = st.text_area("Paste your CSV data here:", height=200)
        
        if st.button("Parse CSV Data") and csv_text:
            try:
                df = pd.read_csv(StringIO(csv_text))
                st.session_state.original_df = df
                st.session_state.filename = "pasted_data.csv"
                
                st.markdown('<div class="success-box">✅ CSV data parsed successfully!</div>', unsafe_allow_html=True)
                display_dataset_preview(df)
                
            except Exception as e:
                st.error(f"❌ Error parsing CSV: {str(e)}")

def display_dataset_preview(df):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Total Rows", len(df))
    with col2:
        st.metric("📋 Total Columns", len(df.columns))
    with col3:
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("❌ Missing Values", f"{missing_percentage:.1f}%")
    
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Download option
    csv_data = df.to_csv(index=False)
    st.download_button(
        label="💾 Download Original Dataset",
        data=csv_data,
        file_name=f"original_{st.session_state.get('filename', 'dataset.csv')}",
        mime="text/csv"
    )

def data_analysis_section():
    st.markdown('<h2 class="sub-header">🔍 Data Analysis</h2>', unsafe_allow_html=True)
    
    df = st.session_state.original_df
    
    # Missing value analysis
    st.subheader("❌ Missing Values Analysis")
    
    missing_stats = df.isnull().sum()
    missing_stats = missing_stats[missing_stats > 0].sort_values(ascending=False)
    
    if len(missing_stats) > 0:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            missing_stats.plot(kind='bar', ax=ax, color='lightcoral')
            ax.set_title('Missing Values by Column')
            ax.set_ylabel('Number of Missing Values')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("📊 Missing Value Summary")
            for col, count in missing_stats.items():
                percentage = (count / len(df)) * 100
                st.write(f"**{col}**: {count} ({percentage:.1f}%)")
    else:
        st.success("🎉 No missing values found in the dataset!")
    
    # Data types and statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count()
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    with col2:
        st.subheader("📊 Statistical Summary")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        else:
            st.info("No numeric columns found for statistical summary.")

def imputation_section(model_choice, batch_size, top_k_similar):
    st.markdown('<h2 class="sub-header">🚀 AI-Powered Imputation</h2>', unsafe_allow_html=True)
    
    df = st.session_state.original_df
    
    # Check API keys first
    if not GOOGLE_API_KEY:
        st.error("❌ Google API Key is required but not found. Please check your .env file.")
        return
    
    if not PINECONE_API_KEY:
        st.error("❌ Pinecone API Key is required but not found. Please check your .env file.")
        return
    
    if not index:
        st.error("❌ Pinecone index is not available. Please check your connection.")
        return
    
    # Check if imputation is needed
    missing_count = df.isnull().sum().sum()
    if missing_count == 0:
        st.warning("⚠️ No missing values detected. Imputation is not needed.")
        return
    
    # Imputation configuration
    st.subheader("🔧 Imputation Configuration")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Model**: {model_choice}")
    with col2:
        st.info(f"**Batch Size**: {batch_size}")
    with col3:
        st.info(f"**Missing Values**: {missing_count}")
    
    # Pre-imputation checks
    complete_rows = df.dropna()
    if len(complete_rows) == 0:
        st.error("❌ No complete rows found. Cannot perform similarity-based imputation.")
        return
    
    st.success(f"✅ Found {len(complete_rows)} complete rows for reference.")
    
    # Start imputation
    if st.button("🚀 Start Imputation Process", type="primary"):
        with st.spinner("🔄 Preparing embeddings for complete rows..."):
            try:
                # Generate embeddings for complete rows
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔍 Generating embeddings for complete rows...")
                complete_indices = generate_embeddings_for_complete_rows(df)
                progress_bar.progress(25)
                
                status_text.text("🤖 Starting AI-powered imputation...")
                progress_bar.progress(50)
                
                # Perform imputation
                model_name = "Gemini" if "Gemini" in model_choice else "Groq"
                imputed_df = fill_missing_values(df, batch_size=batch_size, model_choice=model_name)
                
                progress_bar.progress(100)
                status_text.text("✅ Imputation completed successfully!")
                
                # Store results
                st.session_state.imputed_df = imputed_df
                st.session_state.imputation_config = {
                    'model': model_choice,
                    'batch_size': batch_size,
                    'top_k': top_k_similar,
                    'complete_rows': len(complete_rows),
                    'imputed_values': missing_count
                }
                
                time.sleep(1)
                st.success("🎉 Imputation process completed! Check the 'Results' tab.")
                
            except Exception as e:
                st.error(f"❌ Error during imputation: {str(e)}")
                st.exception(e)

def results_section():
    st.markdown('<h2 class="sub-header">📊 Imputation Results</h2>', unsafe_allow_html=True)
    
    original_df = st.session_state.original_df
    imputed_df = st.session_state.imputed_df
    config = st.session_state.imputation_config
    
    # Results summary
    st.subheader("📋 Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🤖 Model Used", config['model'])
    with col2:
        st.metric("📦 Batch Size", config['batch_size'])
    with col3:
        st.metric("✅ Complete Rows", config['complete_rows'])
    with col4:
        st.metric("🔧 Values Imputed", config['imputed_values'])
    
    # Before/After comparison
    st.subheader("🔄 Before vs After Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original Dataset (with missing values)**")
        st.dataframe(original_df.head(10), use_container_width=True)
        
        original_missing = original_df.isnull().sum().sum()
        st.metric("❌ Missing Values", original_missing)
    
    with col2:
        st.write("**Imputed Dataset (complete)**")
        st.dataframe(imputed_df.head(10), use_container_width=True)
        
        imputed_missing = imputed_df.isnull().sum().sum()
        st.metric("✅ Missing Values", imputed_missing)
    
    # Correlation Matrix Analysis
    st.subheader("📈 Correlation Matrix Analysis")
    
    try:
        original_corr, imputed_corr, corr_diff = compare_correlation_matrices(original_df, imputed_df)
        
        if original_corr is not None:
            # Plot correlation comparison
            correlation_fig = plot_correlation_comparison(original_corr, imputed_corr, corr_diff)
            st.pyplot(correlation_fig)
            
            # Calculate correlation metrics
            corr_metrics = calculate_correlation_metrics(original_corr, imputed_corr)
            
            if corr_metrics:
                st.subheader("🎯 Correlation Preservation Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f'''
                    <div class="correlation-metric">
                        <h4>🔗 Correlation Preservation</h4>
                        <h2>{corr_metrics["correlation_preservation"]:.3f}</h2>
                        <p>Higher is better (max: 1.0)</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f'''
                    <div class="correlation-metric">
                        <h4>📏 Mean Abs. Difference</h4>
                        <h2>{corr_metrics["mean_absolute_difference"]:.3f}</h2>
                        <p>Lower is better (min: 0.0)</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f'''
                    <div class="correlation-metric">
                        <h4>📐 Max Abs. Difference</h4>
                        <h2>{corr_metrics["max_absolute_difference"]:.3f}</h2>
                        <p>Lower is better (min: 0.0)</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f'''
                    <div class="correlation-metric">
                        <h4>📊 RMSE</h4>
                        <h2>{corr_metrics["rmse"]:.3f}</h2>
                        <p>Lower is better (min: 0.0)</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Interpretation of correlation metrics
                st.subheader("🔍 Interpretation")
                
                corr_pres = corr_metrics["correlation_preservation"]
                mean_diff = corr_metrics["mean_absolute_difference"]
                
                if corr_pres > 0.9 and mean_diff < 0.1:
                    st.success("🎉 **Excellent**: Correlation structure is very well preserved!")
                elif corr_pres > 0.8 and mean_diff < 0.2:
                    st.success("✅ **Good**: Correlation structure is well preserved.")
                elif corr_pres > 0.6 and mean_diff < 0.3:
                    st.warning("⚠️ **Fair**: Correlation structure is moderately preserved.")
                else:
                    st.error("❌ **Poor**: Correlation structure preservation needs improvement.")
                
                # Detailed correlation analysis
                with st.expander("📋 Detailed Correlation Analysis"):
                    st.write("**Original Correlation Matrix:**")
                    st.dataframe(original_corr.round(3))
                    
                    st.write("**Imputed Correlation Matrix:**")
                    st.dataframe(imputed_corr.round(3))
                    
                    st.write("**Correlation Difference Matrix:**")
                    st.dataframe(corr_diff.round(3))
        
        else:
            st.info("ℹ️ Correlation analysis requires at least 2 numeric columns.")
    
    except Exception as e:
        st.warning(f"⚠️ Could not generate correlation analysis: {str(e)}")
    
    # Statistical comparison
    st.subheader("📈 Statistical Distribution Comparison")
    
    try:
        compare_distributions_streamlit(original_df, imputed_df)
    except Exception as e:
        st.warning(f"⚠️ Could not generate statistical comparison: {str(e)}")
    
    # Download options
    st.subheader("💾 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = imputed_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Imputed Dataset",
            data=csv_data,
            file_name=f"imputed_{st.session_state.get('filename', 'dataset.csv')}",
            mime="text/csv",
            type="primary"
        )
    
    with col2:
        # Create comparison report
        report = generate_imputation_report(original_df, imputed_df, config)
        st.download_button(
            label="📊 Download Report",
            data=report,
            file_name="imputation_report.txt",
            mime="text/plain"
        )
    
    with col3:
        # Export both datasets
        combined_data = f"Original Dataset:\n{original_df.to_csv(index=False)}\n\nImputed Dataset:\n{imputed_df.to_csv(index=False)}"
        st.download_button(
            label="📋 Download Both Datasets",
            data=combined_data,
            file_name="combined_datasets.csv",
            mime="text/plain"
        )

def generate_sample_data():
    """Generate sample dataset with missing values for testing"""
    np.random.seed(42)
    n_rows = 100
    
    data = {
        'Name': [f'Person_{i}' for i in range(n_rows)],
        'Age': np.random.randint(18, 80, n_rows),
        'Salary': np.random.normal(50000, 15000, n_rows),
        'Department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing'], n_rows),
        'Experience': np.random.randint(0, 20, n_rows),
        'City': np.random.choice(['New York', 'San Francisco', 'Chicago', 'Boston'], n_rows)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce missing values
    missing_indices = np.random.choice(df.index, size=int(0.15 * len(df)), replace=False)
    for idx in missing_indices:
        col = np.random.choice(df.columns[1:])  # Don't make Name missing
        df.loc[idx, col] = np.nan
    
    return df

def generate_imputation_report(original_df, imputed_df, config):
    """Generate a detailed imputation report"""
    
    # Calculate correlation metrics if possible
    correlation_metrics_text = ""
    try:
        original_corr, imputed_corr, corr_diff = compare_correlation_matrices(original_df, imputed_df)
        if original_corr is not None:
            corr_metrics = calculate_correlation_metrics(original_corr, imputed_corr)
            if corr_metrics:
                correlation_metrics_text = f"""
Correlation Preservation Metrics:
- Correlation Preservation Score: {corr_metrics['correlation_preservation']:.3f}
- Mean Absolute Difference: {corr_metrics['mean_absolute_difference']:.3f}
- Max Absolute Difference: {corr_metrics['max_absolute_difference']:.3f}
- Root Mean Square Error: {corr_metrics['rmse']:.3f}
"""
    except Exception:
        correlation_metrics_text = "Correlation analysis could not be performed."
    
    report = f"""
AI-POWERED DATA IMPUTATION REPORT
================================

Configuration:
- Model: {config['model']}
- Batch Size: {config['batch_size']}
- Top K Similar Rows: {config['top_k']}

Dataset Information:
- Total Rows: {len(original_df)}
- Total Columns: {len(original_df.columns)}
- Complete Reference Rows: {config['complete_rows']}

Missing Values Analysis:
- Original Missing Values: {original_df.isnull().sum().sum()}
- After Imputation: {imputed_df.isnull().sum().sum()}
- Values Imputed: {config['imputed_values']}

Column-wise Missing Values (Original):
{original_df.isnull().sum().to_string()}

Imputation Success Rate: {((config['imputed_values'] - imputed_df.isnull().sum().sum()) / config['imputed_values'] * 100):.1f}%

{correlation_metrics_text}

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    return report

def check_api_status():
    """Check the status of API connections"""
    status_container = st.empty()
    
    with status_container.container():
        st.write("🔍 Checking API connections...")
        
        # Check Google API
        if GOOGLE_API_KEY:
            try:
                # Test Google API connection
                test_response = genai.embed_content(
                    model="models/text-embedding-004",
                    content=["test"]
                )
                st.success("✅ Google API Key working")
            except Exception as e:
                st.error(f"❌ Google API Error: {str(e)}")
        else:
            st.error("❌ Google API Key not found")
        
        # Check Pinecone API
        if PINECONE_API_KEY:
            try:
                if index:
                    st.success("✅ Pinecone connection working")
                else:
                    st.error("❌ Pinecone index not initialized")
            except Exception as e:
                st.error(f"❌ Pinecone Error: {str(e)}")
        else:
            st.error("❌ Pinecone API Key not found")
        
        # Check Groq API
        if GROQ_API_KEY:
            try:
                from groq import Groq
                st.success("✅ Groq API Key found")
            except ImportError:
                st.warning("⚠️ Groq library not installed")
            except Exception as e:
                st.error(f"❌ Groq Error: {str(e)}")
        else:
            st.warning("⚠️ Groq API Key not found (optional)")

if __name__ == "__main__":
    main()
