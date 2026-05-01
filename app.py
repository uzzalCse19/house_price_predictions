import gradio as gr
import pandas as pd
import pickle

# Load trained model
with open('house_price_prediction_model (1).pkl', 'rb') as file:
    model = pickle.load(file)


def predict_price(
    area, bedrooms, bathrooms, floors, age, distance,
    garage, parking, garden, security,
    school_nearby, hospital_nearby, shopping_mall_nearby,
    public_transport, crime_rate, population_density,
    location, income_level
):

    # input dataframe
    input_data = pd.DataFrame([{
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'floors': floors,
        'age': age,
        'distance': distance,
        'garage': garage,
        'parking': parking,
        'garden': garden,
        'security': security,
        'school_nearby': school_nearby,
        'hospital_nearby': hospital_nearby,
        'shopping_mall_nearby': shopping_mall_nearby,
        'public_transport': public_transport,
        'crime_rate': crime_rate,
        'population_density': population_density,
        'location': location,
        'income_level': income_level
    }])

    prediction = model.predict(input_data)

    return f"Predicted House Price: {prediction[0]:,.2f}"


# Gradio UI
app = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Area"),
        gr.Number(label="Bedrooms"),
        gr.Number(label="Bathrooms"),
        gr.Number(label="Floors"),
        gr.Number(label="Age"),
        gr.Number(label="Distance"),
        gr.Number(label="Garage"),
        gr.Number(label="Parking"),
        gr.Number(label="Garden"),
        gr.Number(label="Security"),
        gr.Number(label="School Nearby"),
        gr.Number(label="Hospital Nearby"),
        gr.Number(label="Shopping Mall Nearby"),
        gr.Number(label="Public Transport"),
        gr.Number(label="Crime Rate"),
        gr.Number(label="Population Density"),

        gr.Dropdown(["premium", "standard", "low"], label="Location"),
        gr.Dropdown(["high", "medium", "low"], label="Income Level"),
    ],
    outputs="text",
    title=" House Price Prediction System"
)

app.launch()