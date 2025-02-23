import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

# Load the trained pipeline/model
with open('pipeline_model.pkl', 'rb') as file:
    pipeline = pickle.load(file)

# Add CSS for background
page_bg = """
<style>
body {
    background: linear-gradient(to right, #1e3c72, #2a5298);
    color: white;
    font-family: "Arial", sans-serif;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Title of the app
st.title('🌊 Titanic Survival Prediction App :ship:')

# Create tabs for organization
tabs = st.tabs(["Passenger Details", "Prediction", "Visualizations"])

# --- Passenger Details Input ---
with tabs[0]:
    st.subheader("Provide Passenger Details")
    st.markdown("Use the sliders and dropdowns below to input passenger details:")

    Pclass = st.radio('Passenger Class (Pclass)', [1, 2, 3])
    Sex = st.selectbox('Sex', ['male', 'female'])
    Age = st.slider('Age', min_value=0, max_value=100, value=30)
    SibSp = st.slider('Siblings/Spouses Aboard (SibSp)', min_value=0, max_value=10, value=0)
    Parch = st.slider('Parents/Children Aboard (Parch)', min_value=0, max_value=10, value=0)
    Fare = st.slider('Fare', min_value=0.0, max_value=500.0, value=30.0)
    Embarked = st.selectbox('Port of Embarkation (Embarked)', ['S', 'C', 'Q'])

# Convert categorical inputs to numerical values
Sex = 1 if Sex == 'female' else 0
Embarked_mapping = {'S': 2, 'C': 0, 'Q': 1}
Embarked = Embarked_mapping[Embarked]

# Create a DataFrame for the input
user_input = pd.DataFrame({
    'Pclass': [Pclass],
    'Sex': [Sex],
    'Age': [Age],
    'SibSp': [SibSp],
    'Parch': [Parch],
    'Fare': [Fare],
    'Embarked': [Embarked]
})

# --- Prediction Section ---
with tabs[1]:
    st.subheader("Prediction Results")

    if st.button('Predict'):
        prediction = pipeline.predict(user_input)
        prediction_proba = pipeline.predict_proba(user_input)[:, 1]

        # Display the result
        if prediction[0] == 1:
            st.balloons()
            st.success(f"Passenger is likely to **Survive**. 🎉")
            st.write(f"**Survival Probability:** {prediction_proba[0] * 100:.2f}%")
        else:
            st.snow()
            st.error(f"Passenger is unlikely to **Survive**. ❄️")
            st.write(f"**Survival Probability:** {(1 - prediction_proba[0]) * 100:.2f}%")

# --- Visualization Section ---
with tabs[2]:
    st.subheader("Visualizations")

    # Pie Chart for Survival Probability
    if 'prediction_proba' in locals():
        fig = go.Figure(go.Pie(
            labels=['Survival', 'Non-Survival'],
            values=[prediction_proba[0], 1 - prediction_proba[0]],
            hole=0.4
        ))
        fig.update_layout(
            title="Survival Probability Distribution",
            showlegend=True,
            title_font_size=18,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please make a prediction to view visualizations.")

    # Bar Chart for Input Features
    st.markdown("### Passenger Details Visualization")
    feature_fig = go.Figure()
    feature_fig.add_trace(go.Bar(
        x=user_input.columns,
        y=user_input.values[0],
        marker_color='lightblue'
    ))
    feature_fig.update_layout(
        title="Passenger Input Features",
        xaxis_title="Feature",
        yaxis_title="Value",
        title_font_size=18,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    st.plotly_chart(feature_fig, use_container_width=True)
