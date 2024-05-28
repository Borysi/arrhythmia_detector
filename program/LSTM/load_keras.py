import keras
import matplotlib.pyplot as plt
from pickle import load
import numpy as np
import pandas as pd
import seaborn as sns
from openedf import parse_edf

ekg,x,y,z = parse_edf("X:/final_program/viva_full_day.edf")

x = x[172500:174000]
y = y[172500:174000]
z = z[172500:174000]

ekg = ekg[4416000:4454400]
plt.figure()
plt.plot(ekg)
plt.show()

exit()


df = pd.DataFrame()
df["ecg"] = ekg
df["x"] = x
df["y"] = y
df["z"] = z
df.to_csv("high_activity_5min.csv")


LSTM = keras.saving.load_model("LSTM_NN.keras")

scaler_X = load(open('scaler_X.sav', 'rb'))

scaler_Y = load(open('scaler_Y.sav', 'rb'))

scaler_Z = load(open('scaler_Z.sav', 'rb'))

scaler_ymean = load(open('scaler_ymean.sav', 'rb'))

x = scaler_X.transform(x.reshape(-1,1)).reshape(1,-1)[0]
y = scaler_Y.transform(y.reshape(-1,1)).reshape(1,-1)[0]
z = scaler_Z.transform(z.reshape(-1,1)).reshape(1,-1)[0]


# Функция для разбиения данных на окна с перекрытием
def create_windows(data, window_size, overlap):
    step = int(window_size * (1 - overlap))
    windows = []
    for start in range(0, len(data) - window_size + 1, step):
        windows.append(data[start:start + window_size])
    return np.array(windows)

# Определение параметров окон
window_size = 150
overlap = 0

# Создание окон для каждой оси акселерометра
X_x = create_windows(x, window_size, overlap)
X_y = create_windows(y, window_size, overlap)
X_z = create_windows(z, window_size, overlap)


X = np.stack((X_x, X_y, X_z), axis=2)



y_pred_scaled = LSTM.predict(X)
y_pred = scaler_ymean.inverse_transform(y_pred_scaled).reshape(1,-1)[0]

print(y_pred)
print(len(y_pred))

print("pred_mean: ",y_pred.mean())

plt.figure()
sns.scatterplot(x=range(0,len(y_pred)),y=y_pred,alpha = 0.6)
plt.show()