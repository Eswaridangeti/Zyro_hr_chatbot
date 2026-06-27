import os
import streamlit as st
...

# ── Load API key from Streamlit secrets ──────────────────────────────────────
# DELETE THIS LINE:
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# REPLACE WITH THIS:
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
elif not os.environ.get("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found. Please add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()

...rest of the code