# churn-prediction-api

# 📊 Customer Churn Prediction API

This project is a **machine learning-powered API** for predicting customer churn (the number of customers who stop using a company's products or services over a specific period of time). It uses **Flask** for API deployment and **XGBoost** for churn prediction, deployed on **Render**.

## 🚀 Features
- **Machine Learning Model:** Trained with XGBoost.
- **API:** Flask-based, serving predictions.
- **Deployment:** Hosted on Render.
- **Database:** Uses PostgreSQL for customer records.
- **Visualization:** Generates customer churn insights.

## 📂 Project Structure
```
├── crm/
│   ├── app.py              # Flask API
│   ├── train_model.py      # ML model training
│   ├── fetch_data.py       # Data extraction from PostgreSQL
│   ├── requirements.txt    # Dependencies
│   ├── render.yaml         # Render deployment config
│   ├── test_api.py         # API test script
│   ├── README.md           # Project documentation
```

## 🛠️ Installation
Clone the repository and install dependencies:
```sh
git clone https://github.com/fatima-d09/churn-prediction-api.git
cd churn-prediction-api
pip install -r requirements.txt
```

## 🔥 Running the API Locally
```sh
python app.py
```
The API will run on `http://127.0.0.1:5000`.

## 🎯 Making Predictions
Test your API with:
```sh
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [5, 1, 1, 1200.50, 1127, 453, 240.10, 0.002203]}'
```

## 🚀 Deployment on Render
1. Push your latest changes to GitHub:
   ```sh
   git add .
   git commit -m "Updated README and project files"
   git push origin main
   ```
2. Go to [Render](https://render.com/) and deploy the repository.

## 📝 Future Enhancements
- Store predictions in PostgreSQL.
- Build a frontend for visualization.
- Improve model accuracy with more features.

## 💡 Author
Created by **Fatima Dembele** 🚀

