from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from charts import bar_trees_summary, pie_tree_distribution, aqi_gauge_attractive

app = Flask(__name__)

weather_model = load_model("models/weather_lstm.keras")
tree_model = pickle.load(open("models/tree_data_model.pkl","rb"))

weather_df = pd.read_csv("data/weather_2021_2025.csv")
scaler = MinMaxScaler()
scaler.fit(weather_df[['temperature','humidity','rainfall','aqi']])

# Inside your POST request route

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        population = int(request.form['population'])
        existing = int(request.form['existing'])
        area = float(request.form['area'])
        city = request.form["city"]

        last_30 = scaler.transform(
            weather_df[['temperature','humidity','rainfall','aqi']].tail(30)
        )

        pred_weather = weather_model.predict(last_30.reshape(1,30,4))
        pred_weather = scaler.inverse_transform(pred_weather)

        predicted_aqi = pred_weather[0][3]

        def aqi_tree_factor(aqi):
            if aqi <= 50:
                return 0.9  # Good air → fewer trees needed
            elif aqi <= 100:
                return 1.0
            elif aqi <= 200:
                return 1.2
            elif aqi <= 300:
                return 1.4
            else:
                return 1.6  # Hazardous → aggressive plantation

        AREA_CATEGORY = {
            # IT / Corporate hubs → very high floating population
            "Hinjewadi": "IT",
            "Magarpatta City": "IT",
            "Kharadi": "IT",
            "Viman Nagar": "IT",
            "Baner": "IT",
            "Balewadi": "IT",
            "Senapati Bapat Road": "IT",

            # Commercial / Mixed business zones
            "Camp": "commercial",
            "Deccan Gymkhana": "commercial",
            "Koregaon Park": "commercial",
            "Kalyani Nagar": "commercial",

            # Education / Student-heavy zones
            "Karve Nagar": "education",
            "Aundh": "education",
            "Pashan": "education",

            # Industrial zones
            "Pimpri": "industrial",
            "Chinchwad": "industrial",
            "Bhosari": "industrial",

            # Residential dominant
            "Kothrud": "residential",
            "Bavdhan": "residential",
            "Wanowrie": "residential",
            "Hadapsar": "residential",
            "Yerwada": "residential",
            "Talegaon": "residential"
        }

        def population_factor_by_area(city):
            category = AREA_CATEGORY.get(city, "residential")

            if category == "IT":
                return 1.45
            elif category == "commercial":
                return 1.50
            elif category == "education":
                return 1.35
            elif category == "industrial":
                return 1.30
            else:
                return 1.20



        pop_factor = population_factor_by_area(city)
        effective_population = int(population * pop_factor)

        trees = tree_model.predict([[effective_population, existing, area, predicted_aqi]])
        base_prediction = int(trees[0])
        print("Base Tree Prediction (ML):", base_prediction)
        print("Predicted AQI:", predicted_aqi)
        factor = aqi_tree_factor(predicted_aqi)
        adjusted_prediction = int(base_prediction * factor)


        print("Area:", city)
        print("Category:", AREA_CATEGORY.get(city))
        print("Population factor:", pop_factor)
        print("Effective population:", effective_population)


        tree_deficit = max(adjusted_prediction - existing, 0)
        bar_img = bar_trees_summary(existing, adjusted_prediction)
        pie_img = pie_tree_distribution(existing, adjusted_prediction)
        aqi_img = aqi_gauge_attractive(predicted_aqi)
        return render_template(
            "dashboard.html",
            trees=adjusted_prediction,
            aqi=int(predicted_aqi),
            city=city,
            bar_img=bar_img,
            pie_img=pie_img,
            aqi_img=aqi_img,
            area=area,
            predicted=adjusted_prediction,
            tree_deficit=tree_deficit,
            aqi_factor=factor,
            area_category=AREA_CATEGORY.get(city),
            population_factor=pop_factor,
            effective_population=effective_population,


        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
