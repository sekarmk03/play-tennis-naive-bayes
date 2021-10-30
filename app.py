import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

ds = pd.read_csv('play_tennis.csv', usecols= ['outlook','temp','humidity','wind','play'])

# mengambil data tiap atribut
x = ds.iloc[:,:-1].values # atribut x (outlook, temp, humidity, wind)
y = ds.iloc[:,-1].values # atribut y (play)

# membuat encoder (mengubah string menjadi angka)
encoder = LabelEncoder()

# mengubah value string menjadi angka
x[:,0] = encoder.fit_transform(x[:,0]) # outlook
x[:,1] = encoder.fit_transform(x[:,1]) # temp
x[:,2] = encoder.fit_transform(x[:,2]) # humidity
x[:,3] = encoder.fit_transform(x[:,3]) # wind
y = encoder.fit_transform(y) # play

# Membuat Pengklasifikasi Multinomial Naive Bayes
model = MultinomialNB()

# train model
model.fit(x, y)

# tampilan index (halaman awal)
@app.route('/')
def index():
    # menampilkan template
    return render_template('index.html', predicted="?", outlook = "?", temp = "?", humidity = "?", wind = "?")

# tampilan setelah button prediksi di klik
# diarahkan ke halaman /prediction
# data dikirimkan menggunakan method post
@app.route('/prediction', methods=['POST'])
def prediction():
    # mengambil data masukan user berdasarkan name form input
    outlook = int(request.form['outlook'])
    temp = int(request.form['temp'])
    humidity = int(request.form['humidity'])
    wind = int(request.form['wind'])
    
    # memprediksi masukan user berdasarkan model
    predicted = model.predict([[outlook, temp, humidity, wind]])
    
    # mengubah angka encoder menjadi string
    # outlook
    if outlook == 0:
        outlook = "Overcast"
    elif outlook == 1:
        outlook = "Rain"
    elif outlook == 2:
        outlook = "Sunny"
    # temp
    if temp == 0:
        temp = "Cool"
    elif temp == 1:
        temp = "Hot"
    elif temp == 2:
        temp = "Mild"
    # humidity
    humidity = "Normal" if humidity else "High"
    # wind
    wind = "Weak" if wind else "Strong"

    # menampilkan template yang sama dengan membawa data hasil prediksi
    return render_template('index.html', predicted = "Yes" if predicted else "No", outlook = outlook, temp = temp, humidity = humidity, wind = wind)

# drive
if __name__ == '__main__':
    app.run(debug=True)