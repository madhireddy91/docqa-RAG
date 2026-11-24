import sys
from pathlib import Path

sys.path.insert(0, "src")
import streamlit as st
from ask import ask

st.set_page_config(page_title="DocQA", page_icon="📄")
st.title("📄 DocQA: FAU exam regulations")
st.caption("Ask in German or English. Answers come only from the indexed PDFs, with sources.")

question = st.text_input("Your question")

if question:
    with st.spinner("Searching the documents..."):
        answer, hits = ask(question)
    st.markdown(answer)
    with st.expander("Sources (retrieved passages)"):
        for h in hits:
            name = Path(h.metadata.get("source", "")).name
            st.markdown(f"**{name}, page {h.metadata.get('page', 0) + 1}**")
            st.write(h.page_content)
