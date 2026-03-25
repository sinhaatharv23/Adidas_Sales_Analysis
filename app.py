import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Adidas Dashboard", layout="wide")

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
.main {
    background-color: #f5f5f5;
}
h1 {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1>Adidas Sales Dashboard</h1>", unsafe_allow_html=True)

# ---------------- LOAD & CLEAN DATA ----------------
@st.cache_data
def load_data():
    raw = pd.read_excel("Adidas US Sales Datasets.xlsx", header=None)

    header = raw.iloc[3].values
    df = raw.iloc[4:].copy()
    df.columns = header

    df = df.iloc[:, 1:]
    df.reset_index(drop=True, inplace=True)

    df.columns = df.iloc[0]
    df = df[1:]
    df.reset_index(drop=True, inplace=True)

    df.columns = df.columns.astype(str).str.strip()

    # Convert date
    for col in df.columns:
        if 'Invoice' in col:
            df[col] = pd.to_datetime(df[col])

    # Convert numeric columns
    numeric_cols = ['Price per Unit', 'Units Sold', 'Total Sales', 'Operating Profit', 'Operating Margin']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.header("Filters")

region = st.sidebar.selectbox("Select Region", ["All"] + list(df['Region'].dropna().unique()))
product = st.sidebar.selectbox("Select Product", ["All"] + list(df['Product'].dropna().unique()))

filtered_df = df.copy()

if region != "All":
    filtered_df = filtered_df[filtered_df['Region'] == region]

if product != "All":
    filtered_df = filtered_df[filtered_df['Product'] == product]

# ---------------- METRICS ----------------
st.subheader("Overview")

total_sales = int(filtered_df['Total Sales'].sum())
total_profit = int(filtered_df['Operating Profit'].sum())
total_units = int(filtered_df['Units Sold'].sum())

col1, col2, col3 = st.columns(3)

col1.metric("Total Sales", f"${total_sales:,}")
col2.metric("Total Profit", f"${total_profit:,}")
col3.metric("Units Sold", f"{total_units:,}")

st.divider()

# ---------------- GRAPHS ----------------

# Prepare data
region_sales = filtered_df.groupby('Region')['Total Sales'].sum()
monthly_sales = filtered_df.groupby(filtered_df['Invoice Date'].dt.month)['Total Sales'].sum()
product_sales = filtered_df.groupby('Product')['Total Sales'].sum().sort_values(ascending=False)

# Row 1
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sales by Region")
    fig1, ax1 = plt.subplots()
    sns.barplot(x=region_sales.index, y=region_sales.values, ax=ax1)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

with col2:
    st.subheader("Monthly Sales Trend")
    fig2, ax2 = plt.subplots()
    ax2.plot(monthly_sales.index, monthly_sales.values, marker='o')
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Sales")
    st.pyplot(fig2)

# Row 2
col3, col4 = st.columns(2)

with col3:
    st.subheader("Sales by Product")
    fig3, ax3 = plt.subplots()
    sns.barplot(x=product_sales.values, y=product_sales.index, ax=ax3)
    st.pyplot(fig3)

with col4:
    st.subheader("Sales Method Distribution")
    method_sales = filtered_df.groupby('Sales Method')['Total Sales'].sum()
    fig4, ax4 = plt.subplots()
    ax4.pie(method_sales, labels=method_sales.index, autopct='%1.1f%%')
    st.pyplot(fig4)

# ---------------- HEATMAP ----------------
st.subheader("Correlation Heatmap")

numeric_cols = ['Price per Unit', 'Units Sold', 'Total Sales', 'Operating Profit', 'Operating Margin']

fig5, ax5 = plt.subplots()
sns.heatmap(filtered_df[numeric_cols].corr(), annot=True, ax=ax5)
st.pyplot(fig5)

# ---------------- DATA VIEW ----------------
with st.expander("View Raw Data"):
    st.dataframe(filtered_df)