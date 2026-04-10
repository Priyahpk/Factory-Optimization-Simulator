import streamlit as st
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("Nassau Candy Distributor.csv")

# ---------- PREPROCESS ----------
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)

df['Lead Time'] = (df['Ship Date'] - df['Order Date']).dt.days
df['Profit Margin'] = df['Gross Profit'] / df['Sales']

# Clean text
df['Region'] = df['Region'].str.strip()
df['Ship Mode'] = df['Ship Mode'].str.strip()

# ---------- FACTORY MAP ----------
factory_map = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory"
}

df['Factory'] = df['Product Name'].map(factory_map)

# ---------- ENCODING ----------
from sklearn.preprocessing import LabelEncoder

region_le = LabelEncoder()
ship_le = LabelEncoder()
factory_le = LabelEncoder()
product_le = LabelEncoder()

df['Region_enc'] = region_le.fit_transform(df['Region'])
df['ShipMode_enc'] = ship_le.fit_transform(df['Ship Mode'])
df['Factory_enc'] = factory_le.fit_transform(df['Factory'])
df['Product_enc'] = product_le.fit_transform(df['Product Name'])

# ---------- MODEL ----------
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X = df[['Region_enc', 'ShipMode_enc', 'Factory_enc', 'Product_enc', 'Units']]
y = df['Lead Time']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# ---------- SIMULATION FUNCTION ----------
def simulate(product, region, ship_mode, units):
    results = []

    for factory in df['Factory'].unique():
        row = {
            'Region_enc': region_le.transform([region])[0],
            'ShipMode_enc': ship_le.transform([ship_mode])[0],
            'Factory_enc': factory_le.transform([factory])[0],
            'Product_enc': product_le.transform([product])[0],
            'Units': units
        }

        pred = model.predict(pd.DataFrame([row]))[0]
        results.append((factory, round(pred, 2)))

    return sorted(results, key=lambda x: x[1])

# ---------- UI ----------
st.title("🍬 Factory Optimization Simulator")

st.sidebar.header("User Inputs")

product = st.sidebar.selectbox("Select Product", df['Product Name'].unique())
region = st.sidebar.selectbox("Select Region", df['Region'].unique())
ship_mode = st.sidebar.selectbox("Select Ship Mode", df['Ship Mode'].unique())
units = st.sidebar.slider("Units", 1, 100, 10)

# ---------- RUN SIMULATION ----------
if st.sidebar.button("Run Simulation"):

    results = simulate(product, region, ship_mode, units)

    best = results[0]
    worst = results[-1]

    st.subheader("📊 Recommendation Results")

    st.success(f"✅ Best Factory: {best[0]} (Lead Time: {best[1]} days)")
    st.error(f"❌ Worst Factory: {worst[0]} (Lead Time: {worst[1]} days)")

    # Convert to DataFrame for table
    res_df = pd.DataFrame(results, columns=["Factory", "Predicted Lead Time"])

    st.subheader("📈 All Factory Comparisons")
    st.dataframe(res_df)

    # Bar chart
    st.subheader("📊 Lead Time Comparison")
    st.bar_chart(res_df.set_index("Factory"))