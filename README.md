# Rental Home Price Prediction

![MLOps Web Application](https://miro.medium.com/v2/resize:fit:786/format:webp/1*AYglWbRIuKfBITlkSZj1vg.png)

This project aims to predict rental home prices based on various features using a machine learning model. The web application allows users to input features and get predictions on rental prices.

## Web Application URL

[Click here to access the Rental Price Prediction Web App](https://mlops-prediction-nas-shaqirah-eaba5152ba6d.herokuapp.com/)

## Project Structure

```plaintext
Root (Rental Home Price Prediction)
│
├── app.py                # Main Flask application file
├── dvc.yaml              # DVC configuration file
├── poetry.lock           # Poetry lock file for dependencies
├── pyproject.toml        # Poetry configuration file
│
├── configs               # Configuration files
│   └── config.yaml
│
├── data                  # Data directory for raw and processed datasets
│   ├── raw               # Raw data
│   └── processed         # Processed data
│
├── models                # ML models
│   └── nas_rental_prediction
│       └── best_model_new_saved.pkl
│
├── notebooks             # Jupyter notebooks for EDA and modeling
│   └── EDA_and_Modeling.ipynb
│
├── scripts               # Utility scripts
│   └── train_model.py
│
├── static                # Static files for the web app
│   ├── css               # CSS files
│   ├── js                # JavaScript files
│   └── loading.gif       # Loading animation GIF
│
├── templates             # Flask templates
│   ├── nas_rental_prediction.html
│   └── navbar.html
│
└── tests                 # Test cases
    └── test_app.py
```

Setup and Installation
1. Clone the repository:
git clone https://github.com/boss2256/MLOps-Web-Application.git
cd yourrepository

2. Install dependencies using Poetry:
poetry install

3. Activate the virtual environment:
poetry shell

4. Run the Flask application:
flask run

## Usage
1. Predict Rental Price: Enter the required features such as amenities, availability, bathrooms, accommodates, number of reviews, maximum nights, bedrooms, beds, and guests included to predict the rental price of budget accommodations.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

