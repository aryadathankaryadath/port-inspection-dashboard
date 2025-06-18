import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
from rake_nltk import Rake
import nltk

# ✅ Download required NLTK data (punkt & stopwords)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download('punkt_tab')  # Add this line

# ✅ Configure page settings – MUST be first Streamlit command
st.set_page_config(
    page_title="Port Authority Inspection Analysis",
    page_icon="🚢",
    layout="wide"
)

# ✅ Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #2c3e50;
    }
    .stSelectbox {
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


# ✅ Load authority data
@st.cache_data
def load_authority_data():
    try:
        df = pd.read_excel("/Users/aryadathankaryadth/PycharmProjects/port-inspection-dashboard/.github/workflows/final.xlsx")
        return df
    except Exception as e:
        st.error(f"Error loading authority data: {str(e)}")
        return None

# ✅ Load ship deficiency data
@st.cache_data
def load_ship_data():
    try:
        ship_df = pd.read_excel("/Users/aryadathankaryadth/PycharmProjects/port-inspection-dashboard/.github/workflows/ship data.xlsx")
        return ship_df
    except Exception as e:
        st.error(f"Error loading ship data: {str(e)}")
        return None


# ✅ Main application
def main():
    st.title("🚢 Port Authority Inspection Analytics Dashboard")

    df = load_authority_data()
    ship_df = load_ship_data()
    if df is None or ship_df is None:
        return

    # Sidebar filters
    st.sidebar.header("Filters")

    if 'Date' in df.columns:
        min_date = df['Date'].min()
        max_date = df['Date'].max()
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

    ports = sorted(df["Authority"].unique())
    selected_port = st.sidebar.selectbox("Select a Port Authority", ports)

    filtered_df = df[df["Authority"] == selected_port]

    # Ship inspection match
    st.sidebar.header("Ship Check")
    if 'Vessel Name' in ship_df.columns and 'Nature of deficiency' in ship_df.columns:
        ships = sorted(ship_df['Vessel Name'].dropna().unique())
        selected_ship = st.sidebar.selectbox("Select a Ship", ships)

        ship_rows = ship_df[ship_df['Vessel Name'] == selected_ship]
        past_deficiencies = ship_rows['Nature of deficiency'].dropna().astype(str)
        combined_text = " ".join([t for t in past_deficiencies if t.lower() != "nil"])

        # ✅ RAKE keyword extraction
        rake = Rake()
        rake.extract_keywords_from_text(combined_text)
        ranked_phrases = rake.get_ranked_phrases_with_scores()
        top_keywords = [phrase for score, phrase in ranked_phrases[:10]]

        st.sidebar.markdown("### 🚨 Past Issues (Extracted Keywords)")
        if top_keywords:
            st.sidebar.write(", ".join(top_keywords))
        else:
            st.sidebar.info("No relevant deficiencies found.")

        authority_phrases = filtered_df['Phrase'].str.lower().tolist()
        matched_table = []

        for kw in top_keywords:
            for phrase in authority_phrases:
                if kw.lower() in phrase:
                    matched_table.append({"Matched Keyword": kw, "Port Phrase": phrase})

        matched_keywords = list({row['Matched Keyword'] for row in matched_table})  # Unique keywords

        if matched_keywords:
            st.sidebar.markdown("### ⚠️ Attention Required")
            st.sidebar.warning(
                f"This port inspects areas related to: **{', '.join(set(matched_keywords))}**.\n"
                f"Your ship had past issues in these areas. Please double-check before arrival."
            )
            st.sidebar.markdown("#### 🔍 Matched Deficiencies vs Port Focus")
        if matched_table:
            st.sidebar.dataframe(pd.DataFrame(matched_table), height=200)
        else:
            st.sidebar.success("No major overlaps found with this port's inspection priorities.")

    # Layout: Focus chart + Word cloud
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Top Inspection Focus Areas")
        top_n = st.slider("Number of top phrases to show", 5, 20, 10)
        top_phrases = filtered_df.nlargest(top_n, "TF-IDF Score")

        fig = px.bar(
            top_phrases,
            x="Phrase",
            y="TF-IDF Score",
            title=f"Top {top_n} Phrases by TF-IDF Score",
            labels={"Phrase": "Focus Area", "TF-IDF Score": "Importance Score"},
        )
        st.plotly_chart(fig)

    with col2:
        st.subheader("☁️ Inspection Word Cloud")
        phrase_dict = dict(zip(filtered_df['Phrase'], filtered_df['TF-IDF Score']))
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100,
            contour_width=3,
            contour_color='steelblue'
        ).generate_from_frequencies(phrase_dict)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    # Metrics
    st.subheader("📈 Key Metrics")
    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric(
            label="Average TF-IDF Score",
            value=f"{filtered_df['TF-IDF Score'].mean():.2f}",
            delta=f"{filtered_df['TF-IDF Score'].mean() - df['TF-IDF Score'].mean():.2f}"
        )

    with metric_col2:
        st.metric(
            label="Number of Unique Phrases",
            value=len(filtered_df)
        )

    with metric_col3:
        st.metric(
            label="Max TF-IDF Score",
            value=f"{filtered_df['TF-IDF Score'].max():.2f}"
        )

    # Searchable data table
    st.subheader("🔍 Detailed Data View")
    search_term = st.text_input("Search phrases...")

    if search_term:
        filtered_data = filtered_df[filtered_df['Phrase'].str.contains(search_term, case=False)]
    else:
        filtered_data = filtered_df

    st.dataframe(
        filtered_data.style.background_gradient(subset=['TF-IDF Score'], cmap='YlOrRd'),
        height=400
    )

    # Download button
    st.download_button(
        label="Download Data as CSV",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name=f"port_inspection_data_{selected_port}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime='text/csv'
    )

    # Footer
    st.markdown("---")
    st.markdown("### About This Dashboard")
    st.markdown("""
        This dashboard provides insights into port authority inspection trends using TF-IDF analysis.
        - Use the sidebar filters to analyze specific ports
        - Review ship deficiency overlaps
        - Download the data for offline analysis
    """)


if __name__ == "__main__":
    main()

