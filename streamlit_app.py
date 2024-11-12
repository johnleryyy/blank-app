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

# Create a button for the link
if st.button(button_label):
    # Using JavaScript for redirection
    js_code = f"""
    <script type="text/javascript">
        window.location.href = "{link}";
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)

# Close the div
st.markdown('</div>', unsafe_allow_html=True)
