from flask import Flask, request, render_template
from pycaret.regression import load_model, predict_model
import pandas as pd
import os

app = Flask(__name__, static_folder='static')

# Step 1: Load the trained model
model_path = 'models/nas_rental_prediction/best_model_new_saved'
loaded_model = load_model(model_path)

# Define the context for each feature to be displayed in the form
features_context = {
    'amenity_count': 'How many amenities are there',
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
        prediction_value = predictions.at[0, 'prediction_label']  # Ensure correct column name based on your prediction output

        # Assuming RMSE is known (from model evaluation)
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
