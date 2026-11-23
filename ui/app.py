"""Streamlit user interface (UI layer). It only talks to the API over HTTP/JSON.

Run (the API must be running):  python -m streamlit run ui/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from ui import client as c

st.set_page_config(page_title="Software Vulnerability Prediction", layout="wide")
st.title("Software Vulnerability Prediction Prototype")
st.caption("Decision support for C and C++ functions. Prediction only: it does not find, exploit or fix vulnerabilities.")
st.warning(c.SHORT_NOTICE)

try:
    api_default = st.secrets.get("SVPM_API_URL", c.DEFAULT_API)
except st.errors.StreamlitSecretNotFoundError:
    api_default = c.DEFAULT_API

api = st.sidebar.text_input("Prediction service address", api_default)
page = st.sidebar.radio("Page", ["Predict", "Admin"])

try:
    model = c.health(api)
except c.ApiError as exc:
    st.error(str(exc))
    st.stop()

if model is None:
    st.error("No active model. Run: python -m src.ml.final_fit --promote  (or src.ml.final_evaluation, see the guide).")
    st.stop()

st.sidebar.markdown(f"**Model:** {model['model_name']} ({model['version']})")
if not model.get("sealed_test_evaluated"):
    st.sidebar.warning("Development model: not yet evaluated on the sealed test set.")
else:
    st.sidebar.success("Evaluated once on the sealed test set.")
st.sidebar.caption("Label meaning: a function changed by a CVE-fixing commit. This is not verified vulnerable code.")


def show(data: dict) -> None:
    df = c.results_frame(data)
    flagged = int((df["Review priority"] == "Higher").sum())
    st.success(f"{len(df)} function(s) scored with model {data['model_version']}; {flagged} flagged for higher review priority.")
    st.dataframe(df, width="stretch", hide_index=True)
    if any(r["warnings"] for r in data["results"]):
        st.warning("Some values are outside the range the model was trained on, so those estimates are less reliable.")
    st.info(data["notice"])


if page == "Predict":
    mode = st.radio("How do you want to provide the code?",
                    ["Upload a C/C++ file", "Paste a function", "Enter metrics"], horizontal=True)
    try:
        if mode == "Upload a C/C++ file":
            f = st.file_uploader("Choose one complete source file", type=c.UPLOAD_TYPES)
            if st.button("Analyse file") and f is not None:
                show(c.predict_source(api, f.name, f.getvalue()))
            st.caption("Every function in the file is scored and ranked. Files up to 1 MB and 2,000 functions.")
        elif mode == "Paste a function":
            lang = st.selectbox("Language", ["C", "C++"])
            code = st.text_area("Paste one or more complete functions", height=260)
            if st.button("Analyse function"):
                if not code.strip():
                    st.error("Please paste some code first.")
                else:
                    show(c.predict_source(api, "pasted.c" if lang == "C" else "pasted.cpp", code.encode("utf-8")))
            if "top_nesting_level" in model.get("feature_names", []) and lang == "C++":
                st.warning("This model uses nesting depth, which depends on the surrounding class or file. "
                           "Upload the complete file for C++ code.")
        else:
            st.caption("Enter the values a tool such as Lizard reports for one function.")
            vals = {}
            cols = st.columns(len(model["feature_names"]))
            for col, name in zip(cols, model["feature_names"]):
                vals[name] = col.number_input(name, min_value=0, value=1 if name == "complexity" else 0, step=1, key=f"m_{name}")
            if st.button("Estimate"):
                show(c.predict_metrics(api, {k: int(v) for k, v in vals.items()}))
    except c.ApiError as exc:
        st.error(str(exc))
else:
    st.subheader("Administration")
    st.caption("Needs the admin token set in the SVPM_ADMIN_TOKEN environment variable when the service was started.")
    token = st.text_input("Admin token", type="password")
    col1, col2 = st.columns(2)
    try:
        if col1.button("Show model versions") and token:
            d = c.admin_get(api, "/admin/models", token)
            st.write(f"Active version: **{d['active']}**")
            st.dataframe(d["versions"], width="stretch", hide_index=True)
        if col2.button("Show recent predictions") and token:
            d = c.admin_get(api, "/admin/log?limit=100", token)
            st.dataframe(d["rows"], width="stretch", hide_index=True)
            st.caption("The log stores metric values and results only. Uploaded source code is never stored.")
    except c.ApiError as exc:
        st.error(str(exc))
    st.markdown("**Retraining** is run from the command line by the administrator "
                "(`python -m src.ml.train ...`, then `src.ml.final_evaluation`); new models never become active by themselves.")
