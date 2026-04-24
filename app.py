import streamlit as st
import pickle
import re
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Genre keywords for similarity matching
GENRE_KEYWORDS = {
    "Drama": ["emotional", "deep", "touching", "powerful", "meaningful", "character", "story", "moving", "intense"],
    "Action": ["exciting", "thrilling", "explosion", "fight", "adventure", "fast", "intense", "explosive", "dynamic"],
    "Horror": ["scary", "frightening", "creepy", "terrifying", "suspenseful", "dark", "terrifying", "horror", "chilling"],
    "Comedy": ["funny", "hilarious", "laugh", "comedy", "entertaining", "witty", "humor", "amusing", "comical"],
    "Romance": ["love", "romantic", "sweet", "beautiful", "emotional", "touching", "lovely", "tender", "affection"],
    "Family": ["family", "kid", "fun", "children", "wholesome", "entertainment", "enjoyable", "light", "friendly"],
    "Thriller": ["suspenseful", "intense", "mysterious", "dark", "tension", "thriller", "plot twist", "unexpected", "gripping"],
    "Fantasy": ["magical", "fantasy", "adventure", "wonder", "extraordinary", "imaginative", "epic", "mystical", "otherworldly"],
}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return " ".join(words)

def predict_sentiment(text):
    processed = preprocess_text(text)
    vectorized = tfidf.transform([processed])
    prediction = model.predict(vectorized)[0]
    
    return prediction

def calculate_genre_similarity(review_text, selected_genres):
    """Calculate how well the review matches the user's preferred genres"""
    if not selected_genres:
        return 0
    
    review_lower = review_text.lower()
    genre_texts = " ".join([" ".join(GENRE_KEYWORDS[genre]) for genre in selected_genres])
    
    # Count keyword matches
    matches = 0
    total_keywords = sum(len(GENRE_KEYWORDS[genre]) for genre in selected_genres)
    
    for genre in selected_genres:
        for keyword in GENRE_KEYWORDS[genre]:
            if keyword in review_lower:
                matches += 1
    
    similarity_score = matches / total_keywords if total_keywords > 0 else 0
    return similarity_score

# Custom CSS for creative movie theme with interactive background
st.markdown("""
    <style>
        * {
            transition: all 0.3s ease;
        }
        
        /* Animated background with multiple layers */
        .main {
            background: linear-gradient(-45deg, #ff6b6b, #ffd93d, #6c5ce7, #00b894, #ff6b6b);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            color: #eee;
            position: relative;
            min-height: 100vh;
        }
        
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            25% { background-position: 100% 50%; }
            50% { background-position: 0% 50%; }
            75% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Film reel decorative elements */
        .main::before {
            content: '';
            position: fixed;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: 
                radial-gradient(circle at 20% 50%, rgba(255,255,255,0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(255,255,255,0.1) 0%, transparent 50%),
                radial-gradient(circle at 40% 20%, rgba(255,255,255,0.1) 0%, transparent 50%);
            animation: float 20s ease-in-out infinite;
            pointer-events: none;
            z-index: 0;
        }
        
        @keyframes float {
            0%, 100% { transform: translate(0, 0) rotate(0deg); }
            25% { transform: translate(30px, -30px) rotate(90deg); }
            50% { transform: translate(-20px, 20px) rotate(180deg); }
            75% { transform: translate(40px, 10px) rotate(270deg); }
        }
        
        /* Wrapper for content */
        .main > div {
            position: relative;
            z-index: 1;
        }
        
        /* Title with enhanced animation */
        .title-container {
            text-align: center;
            padding: 3rem 2rem;
            background: linear-gradient(135deg, rgba(255, 107, 107, 0.95), rgba(255, 217, 61, 0.95));
            border-radius: 25px;
            margin-bottom: 3rem;
            box-shadow: 0 0 50px rgba(255, 107, 107, 0.6), 0 0 100px rgba(255, 217, 61, 0.3);
            position: relative;
            overflow: hidden;
            transform: perspective(1000px) rotateX(0deg);
            animation: titlePulse 3s ease-in-out infinite;
        }
        
        @keyframes titlePulse {
            0%, 100% { box-shadow: 0 0 50px rgba(255, 107, 107, 0.6), 0 0 100px rgba(255, 217, 61, 0.3); }
            50% { box-shadow: 0 0 70px rgba(255, 107, 107, 0.8), 0 0 120px rgba(255, 217, 61, 0.5); }
        }
        
        .title-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
            animation: shimmer 2s infinite;
            z-index: 2;
        }
        
        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        
        .title-container::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6c5ce7, #00b894, #ff6b6b);
            background-size: 200% 100%;
            animation: colorFlow 3s ease infinite;
        }
        
        @keyframes colorFlow {
            0% { background-position: 0% 0%; }
            100% { background-position: 200% 0%; }
        }
        
        .title-container h1 {
            color: white;
            font-size: 3.8em;
            margin: 0;
            text-shadow: 3px 3px 8px rgba(0,0,0,0.4), 0 0 25px rgba(255,255,255,0.3);
            position: relative;
            z-index: 3;
            letter-spacing: 3px;
            font-weight: 900;
        }
        
        /* Input styling - enhanced glow */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            background: rgba(255, 255, 255, 0.1) !important;
            color: #fff !important;
            border: 2px solid #ffd93d !important;
            border-radius: 15px !important;
            font-size: 1.08em !important;
            box-shadow: 0 0 20px rgba(255, 217, 61, 0.3), inset 0 0 10px rgba(255,255,255,0.05) !important;
            backdrop-filter: blur(10px) !important;
        }
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: #ff6b6b !important;
            box-shadow: 0 0 35px rgba(255, 107, 107, 0.6), inset 0 0 10px rgba(255,255,255,0.1) !important;
        }
        
        /* Button styling - super interactive */
        .stButton > button {
            background: linear-gradient(135deg, #ff6b6b, #ff8c42, #ffd93d);
            background-size: 200% 200%;
            color: white;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 15px;
            padding: 18px 50px;
            font-size: 1.25em;
            font-weight: 900;
            transition: all 0.4s ease;
            box-shadow: 0 0 30px rgba(255, 107, 107, 0.5), inset 0 0 20px rgba(255,255,255,0.1);
            position: relative;
            overflow: hidden;
            animation: buttonGlow 2s ease-in-out infinite;
        }
        
        @keyframes buttonGlow {
            0%, 100% { box-shadow: 0 0 30px rgba(255, 107, 107, 0.5), inset 0 0 20px rgba(255,255,255,0.1); }
            50% { box-shadow: 0 0 50px rgba(255, 107, 107, 0.8), inset 0 0 20px rgba(255,255,255,0.2); }
        }
        
        .stButton > button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
            transition: left 0.7s;
            z-index: 1;
        }
        
        .stButton > button:hover::before {
            left: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-5px) scale(1.08);
            box-shadow: 0 0 60px rgba(255, 107, 107, 0.9), 0 10px 30px rgba(0,0,0,0.3);
        }
        
        .stButton > button:active {
            transform: translateY(-2px);
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background: linear-gradient(135deg, rgba(108, 92, 231, 0.2), rgba(0, 184, 148, 0.2)) !important;
            backdrop-filter: blur(10px) !important;
            border: 2px solid rgba(255, 217, 61, 0.3) !important;
            border-radius: 20px !important;
        }
        
        /* Result container - dramatic reveal */
        .result-container {
            animation: slideUp 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(50px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Positive result - neon green */
        .positive-result {
            background: linear-gradient(135deg, rgba(0, 184, 148, 0.25), rgba(168, 224, 99, 0.25));
            border: 3px solid #00b894;
            color: #a8e063;
            padding: 30px;
            border-radius: 18px;
            font-size: 1.4em;
            margin: 20px 0;
            box-shadow: 0 0 40px rgba(0, 184, 148, 0.5), inset 0 0 20px rgba(0, 184, 148, 0.15);
            text-align: center;
            font-weight: bold;
            animation: resultGlow 2s ease-in-out infinite;
        }
        
        @keyframes resultGlow {
            0%, 100% { box-shadow: 0 0 40px rgba(0, 184, 148, 0.5), inset 0 0 20px rgba(0, 184, 148, 0.15); }
            50% { box-shadow: 0 0 60px rgba(0, 184, 148, 0.8), inset 0 0 20px rgba(0, 184, 148, 0.25); }
        }
        
        /* Negative result - neon red */
        .negative-result {
            background: linear-gradient(135deg, rgba(255, 107, 107, 0.25), rgba(255, 68, 68, 0.25));
            border: 3px solid #ff6b6b;
            color: #ff9999;
            padding: 30px;
            border-radius: 18px;
            font-size: 1.4em;
            margin: 20px 0;
            box-shadow: 0 0 40px rgba(255, 107, 107, 0.5), inset 0 0 20px rgba(255, 107, 107, 0.15);
            text-align: center;
            font-weight: bold;
            animation: resultGlow 2s ease-in-out infinite;
        }
        
        /* Recommendation text */
        .recommendation-text {
            font-size: 1.2em;
            color: #fff;
            margin: 15px 0;
            line-height: 1.9;
            text-shadow: 0 0 10px rgba(0,0,0,0.3);
        }
        
        /* Label styling */
        .step-label {
            font-size: 1.4em;
            font-weight: bold;
            background: linear-gradient(135deg, #ffd93d, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 25px 0 15px 0;
            text-shadow: none;
            filter: drop-shadow(0 0 5px rgba(255, 217, 61, 0.3));
        }
    </style>
""", unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="🎬 Movie Sentiment Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom title
st.markdown("""
    <div class="title-container">
        <h2>🎬 MOVIE REVIEW SENTIMENT </h2>
    </div>
""", unsafe_allow_html=True)

# Sidebar for genre preferences
st.sidebar.markdown("### Your Movie Preferences")
st.sidebar.markdown("---")

selected_genres = st.sidebar.multiselect(
    "What genres do you love?",
    options=list(GENRE_KEYWORDS.keys()),
    default=["Drama", "Action"],
    help="Select your preferred movie genres"
)

st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)

# Movie name input
st.markdown("<h2 class='step-label'> Enter Movie Name</h2>", unsafe_allow_html=True)
movie_name = st.text_input(
    "Movie Name",
    placeholder="e.g., Inception, The Dark Knight, Barbie...",
    label_visibility="collapsed"
)

st.markdown("<div style='margin: 25px 0;'></div>", unsafe_allow_html=True)

# Review input
st.markdown("<h2 class='step-label'> Write Your Honest Review</h2>", unsafe_allow_html=True)
review_text = st.text_area(
    "Your Review",
    placeholder="Share what you really think about this movie... Don't hold back! 🎬",
    height=160,
    label_visibility="collapsed"
)

st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)

# Analyze button - centered
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_btn = st.button("🎬 ANALYZE MY REVIEW 🎬", use_container_width=True)

# Analysis results
if analyze_btn:
    if movie_name.strip() == "":
        st.warning("⚠️ Please enter a movie name!")
    elif review_text.strip() == "":
        st.warning("⚠️ Please write a review!")
    else:
        st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
        
        # Predict sentiment
        sentiment = predict_sentiment(review_text)
        
        # Calculate genre similarity (but don't display the percentage)
        genre_similarity = calculate_genre_similarity(review_text, selected_genres)
        
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        
        # Movie name display
        st.markdown(f"""
            <div style='text-align: center; margin: 30px 0; font-size: 2.5em; color: #ffd93d; font-weight: bold; text-shadow: 0 0 15px rgba(255, 217, 61, 0.4);'>
                🎬 {movie_name} 🎬
            </div>
        """, unsafe_allow_html=True)
        
        # Sentiment result
        if sentiment == 1:
            st.markdown('<div class="positive-result"> POSITIVE VIBES! 🎉</div>', unsafe_allow_html=True)
            
            st.markdown("""
                <div class="recommendation-text" style='text-align: center;'>
                    <h3 style='color: #a8e063;'>🍿 YOU SHOULD DEFINITELY WATCH THIS! 🍿</h3>
                    <p>The reviewer absolutely loved this movie! Time to add it to your watchlist! </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="negative-result">❌ NEGATIVE VIBES 😞</div>', unsafe_allow_html=True)
            
            st.markdown("""
                <div class="recommendation-text" style='text-align: center;'>
                    <h3 style='color: #ff9999;'>⏭️ MAYBE SKIP THIS ONE ⏭️</h3>
                    <p>The reviewer wasn't impressed. Plenty of better movies out there! 🎬</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Show selected genres
        if selected_genres:
            st.markdown(f"""
                <div style='text-align: center; margin-top: 25px; padding: 15px; background: rgba(255, 107, 107, 0.1); border: 2px solid #ff6b6b; border-radius: 10px;'>
                    <strong style='color: #ffd93d;'>Your Genres:</strong> {', '.join(selected_genres)}
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<p style='text-align: center; color: #999; margin-top: 30px;'>Start over and analyze another movie! 🎥</p>", unsafe_allow_html=True)
