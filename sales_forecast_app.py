import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px
import shap
from prophet import Prophet
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Sales Forecast App", layout="wide")
st.title("📊 Sales Forecast & Profit/Loss Analysis")

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

def detect_target_and_prepare(df):
    lowercols = {c.lower(): c for c in df.columns}
    target = None
    
    if "profit" in lowercols:
        target = lowercols["profit"]
    elif "revenue" in lowercols and "cost" in lowercols:
        df["Profit"] = df[lowercols["revenue"]] - df[lowercols["cost"]]
        target = "Profit"
    elif "sales" in lowercols:
        target = lowercols["sales"]
    return target, df

def find_date_column(df):
    for c in df.columns:
        if "date" in c.lower():
            try:
                _ = pd.to_datetime(df[c])
                return c
            except Exception:
                continue
    return None

def encode_features(X):
    
    num = X.select_dtypes(include=[np.number])
    cat = X.select_dtypes(exclude=[np.number])
    if not cat.empty:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        try:
            cat_enc = pd.DataFrame(enc.fit_transform(cat.fillna("NA")), columns=cat.columns, index=cat.index)
        except Exception:
            
            cat_enc = cat.apply(lambda s: pd.factorize(s.fillna("NA"))[0])
        Xp = pd.concat([num.fillna(num.median()), cat_enc], axis=1)
    else:
        Xp = num.fillna(num.median())
    return Xp

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    
    target, df = detect_target_and_prepare(df)

    if not target:
        st.error("Target column not found. Please include Profit, Sales, or Income/Revenue & Cost.")
        st.stop()

    st.write(f"**Detected target column:** {target}")

    
    date_col = find_date_column(df)
    if date_col:
        
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception:
            st.warning("Date column detected but could not convert to datetime — skipping Prophet.")
            date_col = None

    
    if date_col:
        st.subheader("📈 Time-series Forecast (Prophet)")
        prophet_df = df[[date_col, target]].rename(columns={date_col: "ds", target: "y"}).dropna()
        if prophet_df.shape[0] < 10:
            st.warning("Not enough date-indexed rows for reliable Prophet forecasting (need >=10).")
        else:
            try:
                m = Prophet()
                m.fit(prophet_df)
                future = m.make_future_dataframe(periods=30)
                forecast = m.predict(future)
                fig1 = px.line(forecast, x="ds", y=["yhat", "yhat_lower", "yhat_upper"], title="Prophet Forecast (yhat ± bounds)")
                st.plotly_chart(fig1, use_container_width=True)
            except Exception as e:
                st.error(f"Prophet failed: {e}")
    else:
        st.info("No Date column found → skipping time-series forecasting.")

    
    st.subheader("🤖 Machine Learning Model (RandomForest)")

    
    df_model = df.dropna(subset=[target]).reset_index(drop=True)
    if df_model.shape[0] < 10:
        st.error("Not enough rows with target values to train a model (need at least 10).")
        st.stop()

    y = df_model[target]
    X = df_model.drop(columns=[target])

    
    for c in X.columns:
        if c.lower() in ("id", "identifier", "rowid"):
            X = X.drop(columns=[c])

    Xp = encode_features(X)

    if Xp.shape[1] == 0:
        st.error("After encoding, there are no usable features for modeling. Include numeric and categorical features (Price, Quantity, Promotion, Store, Product, etc.).")
        st.stop()

    
    X_train, X_test, y_train, y_test = train_test_split(Xp, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    st.write(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    result_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    fig2 = px.scatter(result_df, x="Actual", y="Predicted", title="Actual vs Predicted")
    st.plotly_chart(fig2, use_container_width=True)

    
    importances = pd.DataFrame({"Feature": Xp.columns, "Importance": model.feature_importances_})
    importances = importances.sort_values("Importance", ascending=False).reset_index(drop=True)
    fig3 = px.bar(importances.head(12), x="Importance", y="Feature", orientation="h", title="Top Features (by importance)")
    st.plotly_chart(fig3, use_container_width=True)

    
    st.subheader("🔍 SHAP Explanations")
    try:
        explainer = shap.TreeExplainer(model)
        
        shap_values = explainer.shap_values(X_test)
        plt.figure(figsize=(8, 5))
        
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        fig_shap = plt.gcf()
        st.pyplot(fig_shap)
        plt.clf()
    except Exception as e:
        st.warning(f"Could not generate SHAP plots: {e}")

    
    st.subheader("💡 Suggestions to Improve Profit (heuristic)")
    suggestions = []
    top_feats = importances.head(8)["Feature"].tolist()
    if any("price" in f.lower() for f in top_feats):
        suggestions.append("Optimize pricing strategy for high-impact products (consider price elasticity tests).")
    if any("cost" in f.lower() or "expense" in f.lower() for f in top_feats):
        suggestions.append("Focus on reducing costs: negotiate with suppliers or optimize operations.")
    if any("promotion" in f.lower() or "discount" in f.lower() for f in top_feats):
        suggestions.append("Re-evaluate promotion ROI; shift spend to the most-converting campaigns/products.")
    if any("quantity" in f.lower() or "units" in f.lower() for f in top_feats):
        suggestions.append("Improve inventory turnover and bundle slow-moving SKUs.")
    if any("store" in f.lower() or "location" in f.lower() for f in top_feats):
        suggestions.append("Identify underperforming locations and run local promotions or re-allocation.")
    if not suggestions:
        suggestions.append("Investigate top features and run A/B tests on pricing, promotions, and cost changes.")

    for s in suggestions:
        st.write("- ", s)

   
    st.subheader("📉 Groups with Largest Average Loss")
    loss_groups = []
    group_col = None
    for candidate in ["Product", "product", "Store", "store", "Category", "category"]:
        if candidate in df.columns:
            group_col = candidate
            break
    if group_col:
        profit_col = target if target in df.columns else None
        if profit_col:
            grp = df[[group_col, profit_col]].groupby(group_col).agg(['mean', 'count'])
            grp.columns = ['mean', 'count']
            neg = grp[grp['mean'] < 0].sort_values('mean').head(10)
            if not neg.empty:
                neg = neg.reset_index()
                st.dataframe(neg)
            else:
                st.write("No groups with negative mean profit found.")
        else:
            st.write("Profit column not available to compute group losses.")
    else:
        st.write("No Product/Store/Category column found to group by.")
    st.subheader("⬇ Download Predictions")
    try:
        full_preds = model.predict(Xp)
        df_out = df_model.copy()
        df_out["Predicted_" + str(target)] = full_preds
        csv = df_out.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV with Predictions", csv, "predictions.csv", "text/csv")
    except Exception as e:
        st.warning(f"Could not generate full predictions: {e}")
else:
    st.info("Awaiting file upload. Please upload a CSV or Excel file containing your sales data.")