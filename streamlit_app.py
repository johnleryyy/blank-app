import streamlit as st

# CSS for custom button styling and grid layout
st.markdown("""
    <style>
    .button-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 20px;
        justify-items: center;
    }
    .button {
        display: inline-block;
        border-radius: 25px;
        background-color: #fff;
        border: 2px solid #000;
        color: #000;
        text-align: center;
        font-size: 16px;
        padding: 10px 20px;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
    }
    .button:hover {
        background-color: #000;
        color: #fff;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app title
st.title("FREE AIRDROP")

# Start the container for the grid layout
st.markdown('<div class="button-container">', unsafe_allow_html=True)

# List of button labels and links
links = {
    "TwitterAgeAirdrop": "https://dropair.io?ref=RMF9RU",
    "PAWSOG AIRDROP": "https://t.me/PAWSOG_bot/PAWS?startapp=e9IaDCrE",
    "GRASS": "https://app.getgrass.io/register?referralCode=o_Ty1Tm40K9lhSJ",
    "NODEPAY": "https://app.nodepay.ai/register?ref=4UdHVDIztwgxkRR",
    "GRADIENT": "https://app.gradient.network/signup?code=7J1JGZ",
    "OASIS": "https://r.oasis.ai/2666b9b66ee30259"
}

# Generate the buttons in a grid layout
for label, url in links.items():
    st.markdown(f'<a href="{url}" class="button">{label}</a>', unsafe_allow_html=True)

# Close the container
st.markdown('</div>', unsafe_allow_html=True)
