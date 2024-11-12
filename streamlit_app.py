import streamlit as st

import streamlit as st

# Center alignment for the whole app
st.markdown("""
    <style>
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    .stButton>button {
        width: 200px;  /* Optional: Set a fixed width for the buttons */
        margin: 10px;  /* Optional: Add space between buttons */
    }
    </style>
""", unsafe_allow_html=True)

# Container to center all elements
st.markdown('<div class="centered">', unsafe_allow_html=True)

# Streamlit app title
st.title("Free airdrop")

# List of links and their labels
links = {
    "Google": "https://www.google.com",
    "YouTube": "https://www.youtube.com",
    "GitHub": "https://www.github.com",
    "OpenAI": "https://www.openai.com"
}

# Create a button for each link
for label, url in links.items():
    if st.button(f"Go to {label}"):
        st.write(f"Redirecting to {label}...")
        st.experimental_rerun()  # Optional: Displaying a small text or indication before redirect
        st.markdown(f'<meta http-equiv="refresh" content="0; url={url}">', unsafe_allow_html=True)

# Close the div
st.markdown('</div>', unsafe_allow_html=True)
