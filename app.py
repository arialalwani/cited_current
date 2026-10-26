"""Streamlit demo:  streamlit run app.py"""
from __future__ import annotations

import os

import streamlit as st

from src import llm
from src.pipeline import CONFIGS, FULL, RAGPipeline

st.set_page_config(page_title="Cited & Current", page_icon="📑", layout="wide")


@st.cache_resource(show_spinner="Building index...")
def load(corpus: str) -> RAGPipeline:
    return RAGPipeline(corpus)


with st.sidebar:
    st.header("Settings")
    corpus = st.selectbox("Corpus", [d for d in ("data/corpus", "data/sample_corpus") if os.path.isdir(d)],
                          index=0)
    cfg_name = st.selectbox("Pipeline configuration", [c.name for c in CONFIGS], index=len(CONFIGS) - 1)
    use_hyde = st.checkbox("HyDE query expansion (needs LLM key)", value=False,
                           disabled=llm.provider() is None)
    pipe = load(corpus)
    lam = st.slider("Recency weight (lambda)", 0.0, 1.5, float(pipe.lam), 0.05)
    pipe.lam = lam
    st.caption(f"Embedding: {pipe.backends['embedding']}")
    st.caption(f"Cross-encoder: {pipe.backends['cross_encoder']}")
    st.caption(f"Generator: {('LLM ' + llm.provider() + ' / ' + llm.model_name()) if llm.provider() else 'extractive (set an API key for LLM answers)'}")

st.title("Cited & Current")
st.write("Answers about Aadhaar and OCI procedures, drawn only from official documents, "
         "with the source circular cited for every statement.")
if "sample" in corpus:
    st.warning("Running on the SAMPLE corpus (paraphrased test documents, not official text). "
               "Run the scraper and ingest steps to use the real corpus.")

examples = ["Till when is the online document update in Aadhaar free?",
            "I am 25 and got a new passport. Do I need to re-issue my OCI card?",
            "Which documents are accepted as proof of address for Aadhaar?",
            "What size should my photograph be for the OCI application?"]
q = st.text_input("Your question", placeholder=examples[0])
cols = st.columns(len(examples))
for col, ex in zip(cols, examples):
    if col.button(ex, use_container_width=True):
        q = ex

if q:
    cfg = next(c for c in CONFIGS if c.name == cfg_name)
    with st.spinner("Retrieving and answering..."):
        res = pipe.answer(q, cfg, use_hyde=use_hyde)
    st.subheader("Answer")
    st.markdown(res["answer"])
    if res["references"]:
        st.markdown("**Sources**")
        for ref in res["references"]:
            flag = " ⚠️ superseded by a newer circular" if ref["superseded"] else ""
            st.markdown(f"[{ref['n']}] {ref['citation']}{flag}")
    st.caption(f"Service detected: {res['service'] or 'both'} | generator: {res['mode']}")

    with st.expander("Retrieved chunks (top 5)"):
        for c in res["retrieved"]:
            st.markdown(f"**{c.chunk_id}** · {c.chunk_type} · {c.section} · issued {c.issue_date or 'n/a'}")
            st.text(c.text)
    if res["removed_superseded"]:
        with st.expander(f"Removed by recency rule ({len(res['removed_superseded'])})"):
            for c in res["removed_superseded"]:
                st.markdown(f"~~{c.chunk_id}~~ · {c.title} · issued {c.issue_date}")
