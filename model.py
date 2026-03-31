# =========================
# 📦 IMPORT LIBRARIES
# =========================
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score , classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib


# =========================
# 📂 LOAD DATASETS
# =========================
df_c = pd.read_csv("Crop_recommendation.csv")
df_f = pd.read_csv("Fertilizer Prediction.csv")

# Clean column names
df_f.columns = df_f.columns.str.strip().str.lower().str.replace(" ", "_")
df_c.columns = df_c.columns.str.strip().str.lower()

# Display columns
print(df_c.columns.tolist())
print(df_f.columns.tolist())


# =========================
# 🎯 FEATURE SELECTION
# =========================
crop_features = ['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall']

fertilizer_cols = [
    'temparature', 'humidity', 'moisture', 'nitrogen', 'potassium', 'phosphorous' ,'crop_type','soil_type'
]


# =========================
# 📊 INPUT-OUTPUT SPLIT
# =========================
x_c = df_c[crop_features]
y_c = df_c["label"]

x_f = df_f[fertilizer_cols]
y_f = df_f["fertilizer_name"]


# =========================
# 🔀 TRAIN-TEST SPLIT
# =========================
x_c_train, x_c_test, y_c_train, y_c_test = train_test_split(
    x_c, y_c, test_size=0.2, random_state=42
)

x_f_train, x_f_test, y_f_train, y_f_test = train_test_split(
    x_f, y_f, test_size=0.2, random_state=42
)


# =========================
# 🧮 COLUMN TYPES
# =========================
nums_col_c = [
   'n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall' , 
]

nums_col_f = [
   'temparature', 'humidity', 'moisture', 'nitrogen', 'potassium', 'phosphorous' 
]

text_cols_f = ['soil_type','crop_type']


# =========================
# ⚙️ PREPROCESSING
# =========================
preprocess = ColumnTransformer([
    ('num', StandardScaler(),nums_col_c)
])

preproces1s = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), text_cols_f),
    ('num', StandardScaler(),nums_col_f)
])


# =========================
# 🤖 MODEL PIPELINES
# =========================
decision_pipeline = Pipeline([
    ("scaler",preprocess),
    ("model",DecisionTreeClassifier())
])

logistic_pipeline = Pipeline([
    ("scaler",preprocess),
    ("model",LogisticRegression())
])

random_pipeline = Pipeline([
    ("scaler",preprocess),
    ("model",RandomForestClassifier(n_estimators=1000))
])


# =========================
# 🔍 HYPERPARAMETERS
# =========================
lr_params = {
    "model__C":       [0.01, 0.1, 1, 10, 100],
    "model__solver":  ["lbfgs", "saga"],
    "model__penalty": ["l2"]
}

dt_params = {
    "model__max_depth":        [3, 5, 10, None],
    "model__min_samples_split":[2, 5, 10],
    "model__criterion":        ["gini", "entropy"]
}

rf_params = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}


# =========================
# 🧠 GRID SEARCH TRAINING
# =========================
dt_grid = GridSearchCV(decision_pipeline,dt_params, cv=10, scoring="accuracy", n_jobs=-1, verbose=1)
lr_grid = GridSearchCV(logistic_pipeline, lr_params, cv=10, scoring="accuracy", n_jobs=-1, verbose=1)
rf_grid = GridSearchCV(random_pipeline,rf_params,cv=10,scoring="accuracy",n_jobs=-1,verbose=1)

dt_grid.fit(x_c_train,y_c_train)
lr_grid.fit(x_c_train,y_c_train)
rf_grid.fit(x_c_train, y_c_train)


# =========================
# 📈 MODEL EVALUATION
# =========================
print("DECISION TREE")
print(f"Best Params : {dt_grid.best_params_}")
print(f"CV Accuracy : {dt_grid.best_score_:.4f}")
print(f"Test Accuracy: {accuracy_score(y_c_test, dt_grid.predict(x_c_test)):.4f}")

print("LOGISTIC REGRESSION")
print(f"Best Params : {lr_grid.best_params_}")
print(f"CV Accuracy : {lr_grid.best_score_:.4f}")
print(f"Test Accuracy: {accuracy_score(y_c_test, lr_grid.predict(x_c_test)):.4f}")

print("RANDOM FOREST")
print(f"Best Params : {rf_grid.best_params_}")
print(f"CV Accuracy : {rf_grid.best_score_:.4f}")
print(f"Test Accuracy: {accuracy_score(y_c_test, rf_grid.predict(x_c_test)):.4f}")


# =========================
# 🌾 FINAL MODELS (CROP + FERTILIZER)
# =========================
random_pipeline_c = Pipeline([
    ("scaler",preprocess),
    ("model",RandomForestClassifier(n_estimators=1000))
])

random_pipeline_f = Pipeline([
    ("scaler",preproces1s),
    ("model",RandomForestClassifier(n_estimators=1000))
])

crop_model = GridSearchCV(random_pipeline_c,rf_params,cv=10,scoring="accuracy",n_jobs=-1,verbose=1)
fertilizer_model = GridSearchCV(random_pipeline_f,rf_params,cv=10,scoring="accuracy",n_jobs=-1,verbose=1)

crop_model.fit(x_c_train,y_c_train)
fertilizer_model.fit(x_f_train,y_f_train)


# =========================
# 🧑‍🌾 USER INPUT
# =========================
n_i = float(input("Enter Nitrogen (N): "))
p_i = float(input("Enter Phosphorus (P): "))
k_i = float(input("Enter Potassium (K): "))
temp_i = float(input("Enter Temperature: "))
hum_i = float(input("Enter Humidity: "))

if(hum_i> 100 or hum_i < 0):
    print("invalid humidity")

ph_i = float(input("Enter pH: "))

if(ph_i > 14 or ph_i< 0):
    print("wring ph entered")

rain_i = float(input("Enter Rainfall: "))
moist_i = float(input("Enter Moisture: "))

soil_i = input("Enter Soil Type (loamy/sandy/clay): ").lower()

if(soil_i not in  ['loamy','sandy','clay']):
    print("invalid soil type")


# =========================
# 📥 PREPARE INPUT DATA
# =========================
data = {
        "N_x": n_i,
        "P_x": p_i,
        "K_x": k_i,
        "temperature_x": temp_i,
        "humidity_x": hum_i,
        "ph": ph_i,
        "rainfall": rain_i,
        "moisture": moist_i,
        "soil_type": soil_i
}

df_input = pd.DataFrame([data])

crop_input = pd.DataFrame([{
    "n": n_i,
    "p": p_i,
    "k": k_i,
    "temperature": temp_i,
    "humidity": hum_i,
    "ph": ph_i,
    "rainfall": rain_i
}])


# =========================
# 🌱 PREDICTIONS
# =========================
crop_pred = crop_model.predict(crop_input)

fert_input = pd.DataFrame([{
    "temparature": temp_i,     
    "humidity": hum_i,
    "moisture": moist_i,
    "nitrogen": n_i,
    "potassium": k_i,
    "phosphorous": p_i,        
    "crop_type": crop_pred[0],
    "soil_type": soil_i
}])

fertilizer_pred = fertilizer_model.predict(fert_input)


# =========================
# 📊 OUTPUT DISPLAY
# =========================
print(" SMART AGRICULTURE RECOMMENDATION")

print("\n INPUT DETAILS")
print(f"N: {n_i}, P: {p_i}, K: {k_i}")
print(f"Temp: {temp_i}, Humidity: {hum_i}")
print(f"pH: {ph_i}, Rainfall: {rain_i}")
print(f"Soil: {soil_i}, Moisture: {moist_i}")

print("\nPREDICTIONS")
print(f"Recommended Crop       → {crop_pred[0]}")
print(f"Recommended Fertilizer → {fertilizer_pred[0]}")

print("="*50)


# =========================
# 🧪 FERTILIZER ANALYSIS
# =========================
fert = fertilizer_pred[0] 

if fert == "Urea":
    print("Fertilizer: Urea (High Nitrogen ~46%)")

elif "-" in fert:   
    parts = list(map(int, fert.split("-")))

    if len(parts) == 2:
        n, p = parts
        print(f"Fertilizer Composition → N: {n}%, P: {p}%")

    elif len(parts) == 3:
        n, p, k = parts
        print(f"Fertilizer Composition → N: {n}%, P: {p}%, K: {k}%")

else:
    print("Fertilizer:", fert)


# =========================
# 🌿 CROP ADVICE DATABASE
# =========================
crop_advice = {
    "rice": {
        "temp": "20°C to 35°C",
        "tips": [
            "Requires flooded or waterlogged conditions",
            "High nitrogen demand during vegetative stage",
            "Maintain 5cm standing water during tillering"
        ]
    },
    "maize": {
        "temp": "18°C to 27°C",
        "tips": [
            "Ensure proper nitrogen supply for good yield",
            "Avoid waterlogging, sensitive to excess water",
            "Needs well-drained fertile soil"
        ]
    },
    "chickpea": {
        "temp": "15°C to 30°C",
        "tips": [
            "Drought tolerant, avoid overwatering",
            "Fixes atmospheric nitrogen, low fertilizer needed",
            "Prefers well-drained loamy soil"
        ]
    },
    "kidneybeans": {
        "temp": "18°C to 24°C",
        "tips": [
            "Sensitive to frost, plant after last frost",
            "Avoid high nitrogen, it's a nitrogen fixer",
            "Needs consistent moisture during pod fill"
        ]
    },
    "pigeonpeas": {
        "temp": "18°C to 38°C",
        "tips": [
            "Drought resistant once established",
            "Deep rooted, improves soil structure",
            "Low water requirement compared to other legumes"
        ]
    },
    "mothbeans": {
        "temp": "25°C to 35°C",
        "tips": [
            "Extremely drought tolerant",
            "Grows well in sandy and arid soils",
            "Minimal irrigation required"
        ]
    },
    "mungbean": {
        "temp": "25°C to 35°C",
        "tips": [
            "Short duration crop (60-90 days)",
            "Sensitive to waterlogging",
            "Good green manure crop for soil health"
        ]
    },
    "blackgram": {
        "temp": "25°C to 35°C",
        "tips": [
            "Thrives in humid tropical climate",
            "Nitrogen fixing legume",
            "Avoid waterlogging during flowering"
        ]
    },
    "lentil": {
        "temp": "15°C to 25°C",
        "tips": [
            "Cool season crop, frost tolerant at early stage",
            "Low water requirement",
            "Prefers well-drained neutral pH soil"
        ]
    },
    "pomegranate": {
        "temp": "25°C to 35°C",
        "tips": [
            "Drought tolerant once established",
            "Needs dry weather during ripening",
            "Regular pruning improves yield"
        ]
    },
    "banana": {
        "temp": "20°C to 35°C",
        "tips": [
            "High water and potassium demand",
            "Wind sensitive, needs sheltered location",
            "Needs well-drained fertile soil"
        ]
    },
    "mango": {
        "temp": "24°C to 30°C",
        "tips": [
            "Dry spell before flowering improves fruit set",
            "Avoid frost, highly sensitive",
            "Deep well-drained soil preferred"
        ]
    },
    "grapes": {
        "temp": "15°C to 35°C",
        "tips": [
            "Requires distinct wet and dry seasons",
            "Pruning is critical for good yield",
            "Sensitive to excess humidity (promotes disease)"
        ]
    },
    "watermelon": {
        "temp": "22°C to 30°C",
        "tips": [
            "Needs warm soil, plant after frost",
            "High water demand during fruit development",
            "Sandy loam soil gives best results"
        ]
    },
    "muskmelon": {
        "temp": "24°C to 35°C",
        "tips": [
            "Reduce watering as fruit matures for sweetness",
            "Warm climate essential",
            "Well-drained sandy loam preferred"
        ]
    },
    "apple": {
        "temp": "10°C to 25°C",
        "tips": [
            "Requires chilling hours (cold winter) for dormancy",
            "Well-drained slightly acidic soil",
            "Regular pruning and thinning improves fruit size"
        ]
    },
    "orange": {
        "temp": "15°C to 30°C",
        "tips": [
            "Needs dry spell to improve fruit color",
            "Regular irrigation, avoid drought stress",
            "Sensitive to frost and waterlogging"
        ]
    },
    "papaya": {
        "temp": "22°C to 32°C",
        "tips": [
            "Fast growing, bears fruit in 9-12 months",
            "Cannot tolerate waterlogging",
            "High potassium demand for fruiting"
        ]
    },
    "coconut": {
        "temp": "25°C to 32°C",
        "tips": [
            "High humidity and rainfall preferred",
            "Salt tolerant, grows near coastal areas",
            "Needs 1500mm+ annual rainfall"
        ]
    },
    "cotton": {
        "temp": "20°C to 30°C",
        "tips": [
            "Requires warm climate and moderate rainfall",
            "Needs good sunlight for growth",
            "Avoid excessive moisture during flowering"
        ]
    },
    "jute": {
        "temp": "24°C to 38°C",
        "tips": [
            "Requires high humidity and rainfall",
            "Grows best in alluvial soil",
            "Needs warm and wet climate throughout growth"
        ]
    },
    "coffee": {
        "temp": "15°C to 28°C",
        "tips": [
            "Grows best at high altitudes",
            "Needs shade in early growth stages",
            "Sensitive to frost and waterlogging"
        ]
    }
}

# =========================
# 🌾 ADVICE DISPLAY
# =========================
crop_name = crop_pred[0].lower()

if crop_name in crop_advice:
    info = crop_advice[crop_name]
    print(f"\n {crop_name.upper()} CULTIVATION ADVICE")
    print(f"- Best Temperature: {info['temp']}")
    for tip in info['tips']:
        print(f"- {tip}")
else:
    print(f"\n Cultivation advice for '{crop_name}' is not available yet.")


# =========================
# 💾 SAVE MODELS
# =========================
joblib.dump(fertilizer_model,"fertilizer_model.joblib")
joblib.dump(crop_model,"crop_model.joblib")


# =========================
# 📊 FINAL MODEL ANALYSIS
# =========================
print("crop model analysis")
print("accuracy ",accuracy_score(y_c_test,crop_model.predict(x_c_test)))
print("confusion matrix  \n",confusion_matrix(y_c_test,crop_model.predict(x_c_test)))
print("classification report : \n" , classification_report(y_c_test,crop_model.predict(x_c_test)))

print("fertilizer model analysis")
print("accuracy ",accuracy_score(y_f_test,fertilizer_model.predict(x_f_test)))
print("confusion matrix  \n",confusion_matrix(y_f_test,fertilizer_model.predict(x_f_test)))
print("classification report : \n" , classification_report(y_f_test,fertilizer_model.predict(x_f_test)))