"""
Styling components for CheckYourHeartHealth Streamlit application."""
import streamlit as st
from pathlib import Path
import base64


def _get_background_image_url() -> str:
    """Return a data URL for the global background image, or an empty string if unavailable."""
    img_path = (
        Path(__file__)
        .resolve()
        .parent  # src
        .parent  # backend
        / "heart-master"
        / "assets.jpg"
    )
    try:
        with img_path.open("rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return ""


def apply_global_styles():
    """Apply global CSS styles to the entire application."""
    # Enhanced dark theme with red accents and subtle heart pattern
    bg_url = _get_background_image_url()
    if bg_url:
        background_line = f"background: url('{bg_url}') center/cover fixed no-repeat #05060c;"
        text_color = "#fafafa"
    else:
        background_line = (
            "background-color: #0e1117; "
            "background-image: radial-gradient(#ff4b4b 1px, transparent 1px); "
            "background-size: 20px 20px; "
            "background-attachment: fixed;"
        )
        text_color = "#fafafa"

    css = f"""
    <style>
        /* Base styles with subtle heart pattern background */
        body, .main, .stApp {{
            {background_line}
            color: {text_color};
        }}
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {{
            color: #ff4b4b;
        }}
        
        /* Input fields */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > div,
        .stTextArea > div > div > textarea,
        .stDateInput > div > div > input {{
            background-color: #262730;
            border: 1px solid #4a4a4a;
            color: #fafafa;
        }}
        
        /* Buttons */
        .stButton > button {{
            background-color: #ff4b4b;
            color: white;
            border: none;
        }}
        
        .stButton > button:hover {{
            background-color: #ff6b6b;
        }}
        
        /* Alerts */
        .stAlert {{
            background-color: #262730;
            border: 1px solid #4a4a4a;
        }}
        
        /* Sidebar */
        .css-1d391kg .css-8lsfnb {{
            background-color: #0e1117;
        }}
        
        /* Main content area */
        .block-container {{
            background-color: rgba(14, 17, 23, 0.9);
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }}
        .nav-card {{
            background-color: rgba(14, 17, 23, 0.95);
            border-radius: 14px;
            border: 1px solid rgba(255, 75, 75, 0.35);
            padding: 0.75rem 1rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.7);
            animation: navCardFadeIn 0.4s ease-out;
        }}
        @keyframes navCardFadeIn {{
            from {{ opacity: 0; transform: translateY(8px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        /* Make each navigation option look like its own pill/box */
        .nav-card [data-testid="stRadio"] > div {{
            row-gap: 0.4rem;
        }}
        .nav-card [data-testid="stRadio"] label {{
            width: 100%;
            padding: 0.45rem 0.75rem;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: radial-gradient(circle at 0 0, rgba(255,75,75,0.16), rgba(9,12,20,0.96));
            transition: border 0.15s ease, background 0.15s ease, transform 0.1s ease;
        }}
        .nav-card [data-testid="stRadio"] label:hover {{
            border-color: rgba(255, 117, 117, 0.7);
            transform: translateY(-1px);
        }}
        .nav-card [data-testid="stRadio"] input:checked + div > label {{
            border-color: rgba(255, 75, 75, 0.95);
            background: linear-gradient(135deg, rgba(255,75,75,0.3), rgba(255,75,75,0.05));
        }}
        .admin-card {{
            background-color: rgba(8, 12, 20, 0.96);
            border-radius: 18px;
            border: 1px solid rgba(255, 75, 75, 0.35);
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 26px 60px rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(10px);
            animation: adminCardRise 0.45s ease-out;
        }}
        @keyframes adminCardRise {{
            from {{ opacity: 0; transform: translateY(12px) scale(0.98); }}
            to {{ opacity: 1; transform: translateY(0) scale(1); }}
        }}
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)


def apply_custom_styles():
    """Apply all custom styles and components."""
    apply_global_styles()
    
    # Add custom fonts (example using Google Fonts)
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Add viewport meta tag for mobile responsiveness
    st.markdown(
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
        unsafe_allow_html=True
    )