import streamlit as st

# CSS for custom button styling
st.markdown("""
    <style>
    .button {
        display: inline-block;
        border-radius: 25px;
        background-color: #fff;
        border: 2px solid #000;
        color: #000;
        text-align: center;
        font-size: 16px;
        padding: 10px 20px;
        margin: 10px;
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
st.title("Stylish CSS Buttons")

# List of button labels and links
links = {
    "TwitterAgeAirdrop": "https://dropair.io?ref=RMF9RU",
    "PAWSOG AIRDROP": "https://t.me/PAWSOG_bot/PAWS?startapp=e9IaDCrE",
    "GRASS": "https://app.getgrass.io/register?referralCode=o_Ty1Tm40K9lhSJ",
    "NODEPAY": "https://app.nodepay.ai/register?ref=4UdHVDIztwgxkRR",
    "GRADIENT": "https://app.gradient.network/signup?code=7J1JGZ",
    "OASIS": "https://r.oasis.ai/2666b9b66ee30259"
}

# Generate the buttons
for label, url in links.items():
    st.markdown(f'<a href="{url}" class="button">{label}</a>', unsafe_allow_html=True)
