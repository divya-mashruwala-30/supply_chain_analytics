import streamlit as st
import pandas as pd
import pickle
import joblib

# ======================================================
# PAGE CONFIGURATION
# ======================================================

st.set_page_config(
    page_title="Supply Chain Analytics",
    page_icon="📦",
    layout="wide"
)

# ======================================================
# LOAD MODELS
# ======================================================

@st.cache_resource
def load_models():

    with open("classifier.pkl", "rb") as file:
        classifier = pickle.load(file)

    kmeans = joblib.load("customer_segmentation_kmeans.pkl")
    scaler = joblib.load("customer_segmentation_scaler.pkl")
    categories = joblib.load("categories.pkl")

    return classifier, kmeans, scaler, categories


classifier, kmeans, scaler, categories = load_models()

# ======================================================
# CUSTOMER SEGMENT NAMES
# ======================================================

segment_names = {
    0: "Regular Customers",
    1: "Premium Customers",
    2: "Frequent Customers",
    3: "Occasional Customers"
}

# ======================================================
# PAGE TITLE
# ======================================================

st.title("📦 Supply Chain Analytics Dashboard")

st.write(
"""
This application demonstrates two Machine Learning solutions developed on the
**DataCo Smart Supply Chain Dataset**:

- 🚚 Late Delivery Risk Prediction
- 👥 Customer Segmentation

The project covers data preprocessing, feature engineering,
classification, clustering and deployment using Streamlit.
"""
)

# ======================================================
# TABS
# ======================================================

home_tab, prediction_tab, segment_tab = st.tabs([
    "🏠 Home",
    "🚚 Late Delivery Prediction",
    "👥 Customer Segmentation"
])

# ======================================================
# HOME PAGE
# ======================================================

with home_tab:

    st.header("Project Overview")

    st.write("""
The objective of this project is to solve two important supply chain problems using Machine Learning.

### 🚚 Late Delivery Risk Prediction
Predict whether an order is likely to be delivered late based on order, customer and shipping information.

### 👥 Customer Segmentation
Group customers based on their purchasing behaviour to help businesses create targeted marketing strategies.
""")

    st.divider()

    st.header("Dataset")

    st.write("""
The project uses the **DataCo Smart Supply Chain Dataset** containing:

- Customer Information
- Product Information
- Order Details
- Shipping Details
- Sales & Discounts
- Delivery Status
""")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Orders", "180,519")
        st.metric("Customers", "20,652")

    with col2:
        st.metric("Classification Model", "Random Forest")
        st.metric("Clustering Model", "K-Means (K=4)")

    st.divider()

    st.header("Machine Learning Workflow")

    st.markdown("""
1. Data Cleaning

2. Feature Engineering

3. Late Delivery Risk Prediction (Random Forest)

4. Customer Segmentation (K-Means)

5. Streamlit Deployment
""")

    st.divider()

    st.header("Technologies Used")

    st.write("""
- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Joblib
""")

# ======================================================
# PREDICTION TAB
# ======================================================

# ======================================================
# LATE DELIVERY PREDICTION
# ======================================================

with prediction_tab:

    st.header("🚚 Late Delivery Risk Prediction")

    st.write("Enter the order details below.")

    col1, col2 = st.columns(2)

    # ---------------- LEFT COLUMN ---------------- #

    with col1:

        order_type = st.selectbox(
            "Type",
            categories["Type"],
            key= "pred_type"
        )

        category = st.selectbox(
            "Category Name",
            categories["Category Name"],
            key="pred_category"
        )

        customer_segment = st.selectbox(
            "Customer Segment",
            categories["Customer Segment"],
            key="pred_segment"
        )

        customer_state = st.selectbox(
            "Customer State",
            categories["Customer State"],
            key="pred_state"
        )

        department = st.selectbox(
            "Department Name",
            categories["Department Name"],
            key="pred_dept"
        )

        market = st.selectbox(
            "Market",
            categories["Market"],
            key="pred_market"
        )

        shipping_mode = st.selectbox(
            "Shipping Mode",
            categories["Shipping Mode"],
            key="pred_shipping"
        )

        scheduled_days = st.number_input(
            "Days for shipment (scheduled)",
            min_value=0,
            max_value=10,
            value=4,
            key="pred_scheduled_days"
        )

        order_month = st.selectbox(
            "Order Month",
            list(range(1, 13)),
            key="pred_order_month"
        )

        order_day = st.selectbox(
            "Order Day Of Week",
            list(range(7)),
            key="pred_order_day"
        )

    # ---------------- RIGHT COLUMN ---------------- #

    with col2:

        order_country = st.selectbox(
            "Order Country",
            categories["Order Country"],
            key="pred_order_country"
        )

        order_region = st.selectbox(
            "Order Region",
            categories["Order Region"],
            key="pred_order_region"
        )

        order_state = st.selectbox(
            "Order State",
            categories["Order State"],
            key="pred_order_state"
        )

        discount = st.number_input(
            "Order Item Discount",
            min_value=0.0,
            value=20.0,
            key="pred_discount"
        )

        product_price = st.number_input(
            "Order Item Product Price",
            min_value=1.0,
            value=100.0,
            key="pred_product_price"
        )

        quantity = st.number_input(
            "Order Item Quantity",
            min_value=1,
            value=2,
            key="pred_quantity"
        )

        order_total = st.number_input(
            "Order Item Total",
            min_value=1.0,
            value=200.0,
            key="pred_order_total"
        )

        customer_orders = st.number_input(
            "Customer Total Orders",
            min_value=1,
            value=5,
            key="pred_customer_orders"
        )

        weekend = st.selectbox(
            "Weekend Order?",
            ["No", "Yes"],
            key="pred_weekend"
        )

    # ---------------- FEATURE ENGINEERING ---------------- #

    is_weekend = 1 if weekend == "Yes" else 0

    discount_to_sales = discount / order_total

    # ---------------- PREDICT BUTTON ---------------- #

    if st.button("Predict Late Delivery Risk"):

        input_df = pd.DataFrame({

            "Type":[order_type],

            "Days for shipment (scheduled)":[scheduled_days],

            "Category Name":[category],

            "Customer Segment":[customer_segment],

            "Customer State":[customer_state],

            "Department Name":[department],

            "Market":[market],

            "Order Country":[order_country],

            "Order Item Discount":[discount],

            "Order Item Product Price":[product_price],

            "Order Item Quantity":[quantity],

            "Order Item Total":[order_total],

            "Order Region":[order_region],

            "Order State":[order_state],

            "Shipping Mode":[shipping_mode],

            "Order_Month":[order_month],

            "Order_DayOfWeek":[order_day],

            "Is_Weekend":[is_weekend],

            "Discount_to_Sales":[discount_to_sales],

            "Customer_Total_Orders":[customer_orders]

        })

        prediction = classifier.predict(input_df)[0]

        probability = classifier.predict_proba(input_df)[0][1]

        st.divider()

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error("⚠️ High Risk of Late Delivery")

        else:

            st.success("✅ Low Risk of Late Delivery")

        st.write(f"**Probability of Late Delivery:** {probability:.2%}")

# ======================================================
# CUSTOMER SEGMENTATION TAB
# ======================================================
# ======================================================
# CUSTOMER SEGMENTATION
# ======================================================

with segment_tab:

    st.header("👥 Customer Segmentation")

    st.write(
        """
Enter the customer purchase details to identify the customer segment.
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        total_orders = st.number_input(
            "Customer Total Orders",
            min_value=1,
            value=5,
            key="cluster_total_orders"
        )

        order_total = st.number_input(
            "Order Item Total",
            min_value=1.0,
            value=500.0,
            key="cluster_order_total"
        )

    with col2:

        quantity = st.number_input(
            "Order Item Quantity",
            min_value=1,
            value=5,
            key="cluster_quantity"
        )

        discount = st.number_input(
            "Order Item Discount",
            min_value=0.0,
            value=50.0,
            key="cluster_discount"
        )

    if st.button("Predict Customer Segment"):

        cluster_data = pd.DataFrame({
            "Customer_Total_Orders": [total_orders],
            "Order_Item_Total": [order_total],
            "Order_Item_Quantity": [quantity],
            "Order_Item_Discount": [discount]
        })

        # Scale the input
        cluster_scaled = scaler.transform(cluster_data)

        # Predict Cluster
        cluster = kmeans.predict(cluster_scaled)[0]

        # Get Segment Name
        segment = segment_names.get(cluster, "Unknown")

        st.divider()

        st.subheader("Prediction Result")

        st.success(f"Customer Segment: **{segment}**")

        st.subheader("Business Recommendation")

        if segment == "Premium Customers":

            st.write("""
⭐ These customers are your highest-value customers.

**Recommendation**

- Provide VIP benefits
- Offer loyalty rewards
- Give early access to new products
- Personalize premium offers
""")

        elif segment == "Frequent Customers":

            st.write("""
🛒 These customers purchase regularly and have good spending habits.

**Recommendation**

- Cross-sell related products
- Upsell premium products
- Reward repeat purchases
""")

        elif segment == "Regular Customers":

            st.write("""
👥 These customers purchase occasionally with moderate spending.

**Recommendation**

- Send personalized promotions
- Offer seasonal discounts
- Encourage repeat purchases
""")

        else:

            st.write("""
🌱 These customers purchase infrequently.

**Recommendation**

- Run re-engagement campaigns
- Send discount coupons
- Encourage another purchase
""")

