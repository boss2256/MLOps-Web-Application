from flask import Flask, request, render_template
from pycaret.regression import load_model, predict_model
import pandas as pd
import os
from hydra import compose, initialize

# Initialize Hydra and load the config
initialize(config_path="configs")
cfg = compose("config.yaml")


app = Flask(__name__, static_folder='static')

# Step 1: Load the trained model
#model_path = 'models/nas_rental_prediction/best_model_new_saved'

# Use cfg for configuration in your app
#model_path = cfg.model.path  # Assuming you have model_path defined in config.yaml
#loaded_model = load_model(model_path)



# Load the Nas rental prediction model
model_path = cfg.model.nas_rental_prediction.path
loaded_model = load_model(model_path)

# Load Shaqirah's mushroom prediction model
shaqirah_model_path = cfg.model.shaqirah_mushroom.path
shaqirah_model = load_model(shaqirah_model_path)  # Assuming this is also a pycaret model

# Define the context for each feature to be displayed in the form
features_context = {
    'amenity_count': 'How many amenities are there in the establishment',
    'availability_30': 'Number of days available in the next 30 days',
    'bathrooms': 'Number of bathrooms',
    'accommodates': 'Number of people it accommodates',
    'number_of_reviews': 'Number of reviews',
    'maximum_nights': 'Maximum number of nights for stay',
    'bedrooms': 'Number of bedrooms',
    'beds': 'Number of beds',
    'guests_included': 'Number of guests included in the price'
}

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Extracting form data
        form_data = request.form
        data = {feature: float(form_data[feature]) for feature in form_data}
        user_input_df = pd.DataFrame([data])

        # Making predictions
        predictions = predict_model(loaded_model, data=user_input_df)
        prediction_value = predictions.at[0, 'prediction_label']  # column name based on prediction output

        # RMSE is known (from model evaluation)
        rmse = 273.4838
        lower_bound = prediction_value - rmse
        upper_bound = prediction_value + rmse
        margin_of_error_percentage = (rmse / prediction_value) * 100

        # Formatting for display
        prediction_str = f"${prediction_value:.2f}"
        confidence_interval_str = f"(${lower_bound:.2f} to ${upper_bound:.2f})"
        margin_of_error_str = f"±{margin_of_error_percentage:.2f}%"

        return render_template('nas_rental_prediction.html',
                               features=features_context,
                               prediction=prediction_str,
                               confidence_interval=confidence_interval_str,
                               margin_of_error=margin_of_error_str)
    # Pass features context to the template for the initial GET request as well
    return render_template('nas_rental_prediction.html', features=features_context)


@app.route('/mushroom_prediction', methods=['GET', 'POST'])
def mushroom_prediction():
    prediction_text = ""  # Initialize prediction text as empty

    if request.method == 'POST':
        # Extracting form data and creating a DataFrame
        form_data = request.form.to_dict()
        user_input_df = pd.DataFrame([form_data], columns=form_data.keys())

        # Ensure the data types match those expected by the model, you might need conversion
        # user_input_df = user_input_df.astype({'gill-size': 'type', 'spore-print-color': 'type', ...})

        # Making predictions using the Shaqirah mushroom model
        predicted_class = shaqirah_model.predict(user_input_df)[0]  # Assuming the predict method returns an array

        # Convert prediction to text
        prediction_text = "Edible" if predicted_class == '1' else "Poisonous"  # Adjust based on your model's output

    # Render the form page for both GET requests and after POST request with the prediction result
    return render_template('shaqirah_mushroom_classification.html', prediction_text=prediction_text)



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
