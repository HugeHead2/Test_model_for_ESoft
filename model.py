# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Загрузка данных
df = pd.read_csv('sold_flats_2020-09-30.csv')

# Подготовка данных
# Замена пропущенных значений и обработка категориальных признаков

df = df.fillna({'bathroom': 'unite', 'closed_yard': 'no',
                'bathrooms_cnt': 1, 'plate': 'no_plate'})
df = df.dropna()

label_encoders = {}
for column in ['status', 'date_sold', 'type', 'two_levels', 'bathroom',
               'plate', 'windows', 'keep', 'balcon',
               'closed_yard']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le


df.to_csv('gdfd.csv')

# Разделение данных на обучающую и тестовую выборки
X = df.drop(['id', 'status', 'date_sold', 'sold_price', 'komunal_cost', 'territory', 'longitude',
             'latitude', 'area_balcony', 'metro_station_id'], axis=1)
y = df['sold_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# Обучение модели
model = RandomForestRegressor(n_estimators=1000, random_state=42)
model.fit(X_train, y_train)

# Оценка модели
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Оценка важности атрибутов
feature_importances = model.feature_importances_
feature_importance_dict = dict(zip(X.columns, feature_importances))
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
print("Feature Importance:")
for feature, importance in sorted_feature_importance:
    print(f"{feature}: {importance}")

n = len(y_test)

plt.plot(np.linspace(1, 4000, n), y_test, c='r')
plt.plot(np.linspace(1, 4000, n), y_pred, c='b')
plt.show()
