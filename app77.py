import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="GuardAI Dashboard", layout="wide")

# ---------------------------
# Email Masking Function
# ---------------------------
def mask_email(email):
    try:
        local, domain = email.split("@")
        if len(local) > 1:
            local_masked = local[0] + "***"
        else:
            local_masked = "*"
        return f"{local_masked}@{domain}"
    except:
        return "Invalid Email"

# ---------------------------
# Load & Preprocess Data
# ---------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    # Map status codes
    status_map = {200: "Success", 401: "Unauthorized", 403: "Forbidden", 404: "Not Found", 500: "Server Error"}
    df["Status_Label"] = df["Status"].map(status_map).fillna("Other")

    # Mask IPs
    df["IP_Masked"] = df["IP"].apply(lambda x: ".".join(x.split(".")[:2] + ["xxx", "xxx"]))

    # Mask Emails
    if "Email" in df.columns:
        df["Email_Masked"] = df["Email"].apply(lambda x: mask_email(x))

    # Convert timestamp to datetime
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    return df

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🛡️ GuardAI – Real-Time Web Security & Anomaly Detection Dashboard")

uploaded_file = st.file_uploader("Upload Access Log CSV", type="csv")

if uploaded_file:
    df = load_data(uploaded_file)

    # ----------------------------------------
    # TOP METRICS (CARDS)
    # ----------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Requests", len(df))
    col2.metric("Unique IPs", df["IP"].nunique())
    col3.metric("Failed Requests", (df["Status"] != 200).sum())
    col4.metric("Suspicious Patterns", "Detected after modeling")

    # ----------------------------------------
    # DATA PREVIEW
    # ----------------------------------------
    st.subheader("📄 Masked Data Preview")
    st.dataframe(df.head())

    # ----------------------------------------
    # FEATURE ENGINEERING
    # ----------------------------------------
    st.subheader("⚙️ Feature Engineering (Used for AI Detection)")

    # Group by IP
    agg = df.groupby("IP").agg({
        "Status": "count",
        "Request": "nunique",
        "ResponseTime": "mean" if "ResponseTime" in df.columns else "median",
        "Timestamp": "count"
    }).rename(columns={
        "Status": "Total_Requests",
        "Request": "Unique_Endpoints",
        "Timestamp": "Request_Rate"
    })

    # Add failed login / forbidden counts
    agg["Failed_Logins"] = df[df["Status"] == 401].groupby("IP").size().reindex(agg.index, fill_value=0)
    agg["Forbidden_Count"] = df[df["Status"] == 403].groupby("IP").size().reindex(agg.index, fill_value=0)

    st.write(agg.head())

    # ----------------------------------------
    # MULTI-FEATURE ANOMALY DETECTION
    # ----------------------------------------
    st.subheader("🤖 AI-Based Anomaly Detection (Isolation Forest)")

    features = ["Total_Requests", "Unique_Endpoints", "Request_Rate",
                "Failed_Logins", "Forbidden_Count"]

    if "ResponseTime" in df.columns:
        features.append("ResponseTime")

    X = agg[features]

    iso = IsolationForest(contamination=0.07, random_state=42)
    agg["Anomaly"] = iso.fit_predict(X)
    agg["Anomaly_Label"] = agg["Anomaly"].map({1: "Normal", -1: "Suspicious"})

    # ----------------------------------------
    # ANOMALY VISUALIZATION
    # ----------------------------------------
    colA, colB = st.columns([2, 1])

    with colA:
        st.subheader("Anomaly Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x=agg["Anomaly_Label"], palette="rocket", ax=ax)
        st.pyplot(fig)

    with colB:
        st.subheader("Anomaly Summary")
        st.write(agg["Anomaly_Label"].value_counts())

    # ----------------------------------------
    # SUSPICIOUS IPs TABLE
    # ----------------------------------------
    st.subheader("🚨 Suspicious IP Addresses Detected")

    suspicious_ips = agg[agg["Anomaly"] == -1]

    if suspicious_ips.empty:
        st.success("No suspicious IPs detected.")
    else:
        df_susp = df[df["IP"].isin(suspicious_ips.index)]
        display_cols = ["IP_Masked", "Request", "Status_Label", "Timestamp"]

        if "Email_Masked" in df.columns:
            display_cols.insert(1, "Email_Masked")

        st.dataframe(df_susp[display_cols])

        st.warning(f"{len(suspicious_ips)} suspicious IPs detected.")

