import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sns.set(style="whitegrid")

# ==========================================
# UI HEADER
# ==========================================
st.set_page_config(page_title="E-Commerce Analysis", layout="wide")
st.title("E-Commerce EDA & Predictive Analysis")

# ==========================================
# FILE UPLOAD
# ==========================================
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # ==========================================
    # TABS
    # ==========================================
    tab1, tab2, tab3 = st.tabs(["EDA", "Visualization", "ML Models"])

    # ==========================================
    # TAB 1 - EDA
    # ==========================================
    with tab1:
        st.subheader("Dataset Overview")

        st.write("Shape:", df.shape)
        st.write("Columns:", df.columns.tolist())
        st.dataframe(df.head())

        st.subheader("Statistical Summary")
        st.write(df.describe(include="all"))

        st.subheader("Missing Values")
        missing = df.isnull().sum()
        st.dataframe(missing)

        st.subheader("Column Analysis")
        col = st.selectbox("Select Column", df.columns)
        st.write(df[col].value_counts())

    with tab2:
        st.subheader("Data Visualization")

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select Numeric Column", numeric_cols)

            fig, ax = plt.subplots()
            sns.histplot(df[selected_col], kde=True, ax=ax)
            st.pyplot(fig)

        if len(numeric_cols) > 1:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

    def extract_purchase_features(purchase_list):
        try:
            purchases = json.loads(purchase_list.replace("'", '"'))
            total_spent = sum([item['Price'] for item in purchases])
            total_items = len(purchases)
        except:
            total_spent = 0
            total_items = 0
        return pd.Series([total_spent, total_items])

    def extract_browsing_features(browsing_list):
        try:
            views = json.loads(browsing_list.replace("'", '"'))
            return len(views)
        except:
            return 0

    def extract_review_rating(review_list):
        try:
            review = json.loads(review_list.replace("'", '"'))
            if isinstance(review, dict) and 'Rating' in review:
                return review['Rating']
            elif isinstance(review, dict):
                return list(review.values())[0]['Rating']
        except:
            return 0

    with tab3:
        st.subheader("Machine Learning Models")

        try:
            # Encoding
            le_gender = LabelEncoder()
            df['Gender_encoded'] = le_gender.fit_transform(df['Gender'])

            le_location = LabelEncoder()
            df['Location_encoded'] = le_location.fit_transform(df['Location'])

            # Feature Engineering
            df[['Total_Spent', 'Total_Purchases']] = df['Purchase History'].apply(extract_purchase_features)
            df['Total_Views'] = df['Browsing History'].apply(extract_browsing_features)
            df['Review_Rating'] = df['Product Reviews'].apply(extract_review_rating)

            df['Made_Purchase'] = (df['Total_Purchases'] > 0).astype(int)

            features = ['Age', 'Gender_encoded', 'Location_encoded',
                        'Annual Income', 'Time on Site',
                        'Total_Views', 'Review_Rating']

            X = df[features]

            if st.button("Run Purchase Prediction Model"):
                y = df['Made_Purchase']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                model = RandomForestClassifier()
                model.fit(X_train, y_train)

                preds = model.predict(X_test)

                st.write("Accuracy:", accuracy_score(y_test, preds))
                st.text(classification_report(y_test, preds))

            if st.button("Run CLV Prediction"):
                y = df['Total_Spent']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                model = RandomForestRegressor()
                model.fit(X_train, y_train)

                preds = model.predict(X_test)

                st.write("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
                st.write("R2 Score:", r2_score(y_test, preds))

            if st.button("Run Customer Segmentation"):
                seg_features = ['Age', 'Annual Income', 'Time on Site',
                                'Total_Spent', 'Total_Purchases', 'Total_Views']

                kmeans = KMeans(n_clusters=3)
                df['Segment'] = kmeans.fit_predict(df[seg_features])

                fig, ax = plt.subplots()
                sns.countplot(x='Segment', data=df, ax=ax)
                st.pyplot(fig)

                # PCA Visualization
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(df[seg_features])

                fig, ax = plt.subplots()
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Segment'])
                st.pyplot(fig)

        except Exception as e:
            st.error("Dataset format issue. Make sure columns like Gender, Location, Purchase History exist.")
            st.write(e)

else:
    st.info("Upload a CSV file to start")