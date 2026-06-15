# 🌾 KRISHIGRAHAN – Smart Agro Intelligence

An AI-powered smart agriculture platform that helps farmers detect crop diseases, get crop and fertilizer recommendations, and access government schemes — all from a single web application.

Built using **Deep Learning + Machine Learning + Flask**.

---

## 🚀 Features

- 🌱 **Crop Recommendation** — Best crop for your soil and climate using ML
- 🧪 **Fertilizer Prediction** — Exact nutrients your field needs
- 🔬 **Plant Disease Detection** — Upload a leaf image or use your camera to detect disease instantly using Computer Vision
- 📋 **Cultivation Advice** — Step-by-step guidance per recommended crop
- 🏛️ **Government Schemes** — 12 active Indian farmer schemes with official links
- 💻 **Multi-page UI** — Clean, modern glassmorphism design with full mobile responsiveness

---

## 🧠 Tech Stack

| Layer                  | Technology                     |
| ---------------------- | ------------------------------ |
| Frontend               | HTML, CSS, JavaScript          |
| Backend                | Flask (Python)                 |
| ML — Crop & Fertilizer | Scikit-learn, Joblib           |
| ML — Disease Detection | TensorFlow, Keras, MobileNetV2 |
| Data Processing        | Pandas, NumPy, Pillow          |
| Model Format           | .joblib, .keras                |

---

## 📁 Project Structure

    project/
    │
    ├── app.py                               ← Flask routes (rendering + prediction APIs)
    ├── inference.py                         ← Disease detection inference module
    ├── model.py                             ← Crop & fertilizer model training script
    │
    ├── krishigrahan_plant_disease_v1.keras  ← Trained MobileNetV2 disease model
    ├── class_names.json                     ← 38 PlantVillage disease class labels
    ├── crop_model.joblib                    ← Trained crop recommendation model
    ├── fertilizer_model.joblib              ← Trained fertilizer prediction model
    │
    ├── Crop_recommendation.csv              ← Crop training dataset
    ├── Fertilizer Prediction.csv            ← Fertilizer training dataset
    │
    ├── templates/
    │   ├── base.html                        ← Shared layout (navbar + footer)
    │   ├── index.html                       ← Home page
    │   ├── crop.html                        ← Crop & fertilizer recommendation page
    │   ├── disease.html                     ← Plant disease detection page
    │   ├── schemes.html                     ← Government schemes page
    │   └── about.html                       ← About page
    │
    ├── static/
    │   ├── style.css
    │   └── script.js
    │
    ├── requirements.txt
    └── README.md

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

    git clone https://github.com/huzaifalokhandwala0307/KRISHIGRAHAN-smart-agro-intelligence.git
    cd krishigrahan

### 2️⃣ Install dependencies

    pip install -r requirements.txt

### 3️⃣ Run the application

    python app.py

Open your browser at `http://localhost:10000`

---

## 🌐 Pages & Routes

| Route              | Page                | Description                                  |
| ------------------ | ------------------- | -------------------------------------------- |
| `/`                | Home                | Hero, stats, features, module CTAs           |
| `/crop`            | Crop Recommendation | Soil input form → crop + fertilizer + advice |
| `/disease`         | Disease Detection   | Upload or camera → instant leaf diagnosis    |
| `/schemes`         | Government Schemes  | 12 active farmer schemes with official links |
| `/about`           | About               | Project overview and tech explanation        |
| `/predict`         | API (POST)          | Crop & fertilizer prediction endpoint        |
| `/predict_disease` | API (POST)          | Disease detection endpoint (returns JSON)    |

---

## 🔬 Disease Detection Model

- **Architecture:** MobileNetV2 Transfer Learning
- **Dataset:** PlantVillage (20,000+ images)
- **Classes:** 38 (plant species × disease type)
- **Validation Accuracy:** 93.34%
- **Input:** Leaf image (upload or webcam capture)
- **Output:** Disease name + confidence % + health status

---

## 🌱 Crop Recommendation

**Inputs:** N, P, K, Temperature, Humidity, pH, Rainfall, Moisture, Soil Type

**Outputs:**

- Recommended crop
- Suggested fertilizer
- Cultivation advice (optimal temperature + farming tips)

---

## 🏛️ Government Schemes Covered

PM-KISAN · PM Fasal Bima Yojana · Kisan Credit Card · Soil Health Card · eNAM · PMKVY · PM Krishi Sinchai Yojana · National Food Security Mission · Paramparagat Krishi Vikas Yojana · Agri Infrastructure Fund · PM Kisan Maandhan Yojana · Rashtriya Krishi Vikas Yojana

---

## 🔮 Future Improvements

- 📡 IoT sensor integration for real-time soil data
- 🌍 Live weather API for auto-filled temperature & humidity
- 🗣️ Multilingual support (Hindi, Gujarati, Marathi)
- 📱 Progressive Web App (PWA) for offline use
- 📊 Farm analytics dashboard
- 🎯 YOLO-based disease localization (highlight infected leaf region)
- 🤖 AI farming chatbot


---

## 🔗 Deployed link


https://krishigrahan-857872478296.asia-south1.run.app


---

## 📜 License

This project is open-source and available under the MIT License.

---

## 💡 Author

**Huzaifa**
B.Tech CS (AI & ML) · Karnavati University · 2025–2029
Built with ❤️ for Indian farmers · 28.3.2026
