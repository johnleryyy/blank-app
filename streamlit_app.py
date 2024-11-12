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

# Link and button label
link = "https://dropair.io?ref=RMF9RU"
button_label = "TwitterAgeAirdrop"

# Create a button that shows a clickable link
if st.button(button_label):
    st.markdown(f"[Click here to go to {button_label}]({link})", unsafe_allow_html=True)

# Close the div
st.markdown('</div>', unsafe_allow_html=True)
