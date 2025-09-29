from flask import Flask, render_template, request, redirect, url_for, flash
import joblib

app = Flask(__name__)
app.secret_key = 'selin_secret_key'  # Flash mesajları için şart

# Modeli yükle
try:
    model = joblib.load("gradient_boosting_model.pkl")
    print("Model başarıyla yüklendi.")
except Exception as e:
    print(f"Model yüklenirken bir hata oluştu: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        pregnancies = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bloodpressure = int(request.form['bloodpressure'])
        skinthickness = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])

        input_data = [[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]]
        prediction = model.predict(input_data)[0]

        result = "Diyabetli" if prediction == 1 else "Diyabetli Değil"

        flash(result)  # Sonucu flash mesajına kaydet
        return redirect(url_for('home'))  # Ana sayfaya yönlendir

    except Exception as e:
        flash(f"Hata: {str(e)}")
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True, port=5500)
