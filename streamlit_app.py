import streamlit as st

# Streamlit app title
st.title("Link Redirector")

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

