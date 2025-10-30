import streamlit as st
from transformers import pipeline
from keybert import KeyBERT
import matplotlib.pyplot as plt

# --------------------------------
# APP CONFIG
# --------------------------------
st.set_page_config(
    page_title="🧠 Smart NLP Summarizer + Keyword Extractor",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Smart NLP Summarizer + Keyword Extractor")
st.markdown(
    "This app uses advanced **Transformer** models for summarization and **BERT embeddings** for keyword extraction. "
    "Adjust parameters in the sidebar to experiment in real time."
)

# --------------------------------
# SIDEBAR CONFIG
# --------------------------------
st.sidebar.header("⚙️ Model Settings")

# Summary length controls
min_len = st.sidebar.slider("Minimum summary length (words)", 20, 150, 30, step=5)
max_len = st.sidebar.slider("Maximum summary length (words)", 50, 300, 120, step=10)

# Keyword extraction controls
num_keywords = st.sidebar.slider("Number of keywords to extract", 3, 15, 7)
ngram_range = st.sidebar.selectbox(
    "Keyword phrase size (n-gram range)",
    options=["(1,1)", "(1,2)", "(1,3)"],
    index=1
)
ngram_tuple = eval(ngram_range)

# Extraction source
extract_from = st.sidebar.radio(
    "Extract keywords from:",
    ("Original Text", "Summary")
)

# About section
st.sidebar.markdown("---")
st.sidebar.info(
    "🧠 **Tech Stack:**\n"
    "- facebook/bart-large-cnn (Summarization)\n"
    "- KeyBERT (Keyword Extraction)\n"
    "- Streamlit UI\n"
    "- CPU-Optimized Torch"
)
st.sidebar.markdown("---")
st.sidebar.caption("Created with ❤️ for NLP demos and quick experiments")

# --------------------------------
# MAIN INPUT AREA
# --------------------------------
text = st.text_area(
    "📝 Paste your text here:",
    height=250,
    placeholder="Type or paste any long text, paragraph, or article here..."
)

# --------------------------------
# MAIN LOGIC
# --------------------------------
if st.button("🚀 Generate Summary and Keywords"):
    if text.strip():
        # -------------------------------
        # SUMMARIZATION
        # -------------------------------
        with st.spinner("Summarizing text... ⏳"):
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            summary_output = summarizer(
                text,
                max_length=max_len,
                min_length=min_len,
                do_sample=False
            )
            summary = summary_output[0]['summary_text']

        st.subheader("📃 Generated Summary")
        st.success(summary)

        # -------------------------------
        # KEYWORD EXTRACTION
        # -------------------------------
        st.markdown("---")
        st.subheader("🔑 Keyword Extraction")

        source_text = text if extract_from == "Original Text" else summary

        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(
            source_text,
            keyphrase_ngram_range=ngram_tuple,
            stop_words='english',
            top_n=num_keywords
        )

        st.write(f"**Extracted from:** {extract_from}")
        for word, score in keywords:
            st.write(f"• **{word}** — relevance: {score:.2f}")

        # -------------------------------
        # VISUALIZATION
        # -------------------------------
        st.markdown("### 📊 Keyword Importance Visualization")

        labels = [kw[0] for kw in keywords]
        scores = [kw[1] for kw in keywords]

        fig, ax = plt.subplots()
        ax.barh(labels, scores, color='cornflowerblue')
        ax.set_xlabel("Relevance Score")
        ax.set_ylabel("Keyword / Phrase")
        ax.invert_yaxis()
        ax.set_title("Keyword Importance")

        st.pyplot(fig)

        # -------------------------------
        # DOWNLOAD SECTION
        # -------------------------------
        st.markdown("---")
        st.download_button(
            label="📥 Download Summary + Keywords",
            data=f"SUMMARY:\n{summary}\n\nKEYWORDS:\n" + "\n".join([w for w, _ in keywords]),
            file_name="summary_keywords.txt",
            mime="text/plain"
        )

    else:
        st.warning("⚠️ Please enter some text before generating a summary!")

