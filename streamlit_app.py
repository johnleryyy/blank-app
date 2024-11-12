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

# List of links and their corresponding button labels
links = {
    "TwitterAgeAirdrop": "https://dropair.io?ref=RMF9RU",
    "PAWSOG AIRDROP": "https://t.me/PAWSOG_bot/PAWS?startapp=e9IaDCrE",
    "GRASS": "https://app.getgrass.io/register?referralCode=o_Ty1Tm40K9lhSJ",
    "NODEPAY": "https://app.nodepay.ai/register?ref=4UdHVDIztwgxkRR",
    "GRADIENT": "https://app.gradient.network/signup?code=7J1JGZ",
    "OASIS": "https://r.oasis.ai/2666b9b66ee30259"
}

# Create a button for each link
for label, url in links.items():
    if st.button(label):
        st.markdown(f"[Click here to go to {label}]({url})", unsafe_allow_html=True)

# Close the div
st.markdown('</div>', unsafe_allow_html=True)
