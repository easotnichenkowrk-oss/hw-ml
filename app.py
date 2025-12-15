import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.set_page_config(page_title="HOMEWORK", layout="wide")
sns.set_style("whitegrid")

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
ohe = pickle.load(open("ohe.pkl", "rb"))
numer = pickle.load(open("numer.pkl", "rb"))
categor = pickle.load(open("categor.pkl", "rb"))

colors_train = ['#4C3D19', '#354024', '#889063', '#CFBB99', '#E5D7C4']
colors_edges = ['#29220e', '#2d361f', '#5c6142', '#9e8f75', '#b6aa9b']


st.title("ДЗ №1. Предсказать стоимость автомобиля")
st.header("Хочу предсказание!")
data = st.file_uploader("Загрузите CSV-файл (для получения массива предсказаний)", type="csv")
manual = st.checkbox("Или ввести данные вручную (для единичного предсказания)")


if manual:
    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Год", min_value=1990, max_value=2025, value=2015)
        km_driven = st.number_input("Пробег", min_value=0, value=50000)
        mileage = st.number_input("Расход топлива (км/л)", min_value=0.0, value=18.0)
        engine = st.number_input("Объём двигателя", min_value=500, value=1500)
        max_power = st.number_input("Мощность", min_value=20.0, value=100.0)
        seats = st.number_input("Места", min_value=2, value=14)
    with col2:
        name = st.text_input("Модель", "Audi A4")
        fuel = st.selectbox("Вид топлива", ["Petrol", "Diesel", "LPG", "CNG"])
        seller_type = st.selectbox("Вид продавца", ["Individual", "Dealer"])
        transmission = st.selectbox("Коробка передач", ["Manual", "Automatic"])
        owner = st.selectbox("Владелец", ["First Owner", "Second Owner", "Third Owner"])
    data = pd.DataFrame([{
        "year": year, "km_driven": km_driven, "mileage": mileage,
        "engine": engine, "max_power": max_power, "seats": seats,
        "name": name, "fuel": fuel, "seller_type": seller_type,
        "transmission": transmission, "owner": owner
    }])
elif data is not None:
    data = pd.read_csv(data)
else:
    data = None



if data is not None and not manual:
    st.subheader("Данные вида:")
    st.dataframe(data.head(), use_container_width=True)

    st.header("Распределения данных")
    numer = [col for col in numer if col in data.columns]
    col_abs, col_rel, _ = st.columns([2, 2, 2])

    with col_abs:
        st.subheader("Абсолютные распределения")
        i = 0
        for col in numer[:5]:
            t = f'{col}'
            if col == 'km_driven':
                data[col] = data[col]/1000
                t = f'{col} (в тыс.)'
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.hist(data[col], edgecolor=colors_edges[i], color=colors_train[i])
            ax.set_title(t, fontsize=9)
            ax.tick_params(labelsize=8)
            st.pyplot(fig, use_container_width=True)
            i = (i + 1) % 5

    with col_rel:
        st.subheader('Относительные распределения')
        i = 0
        for col in numer[:5]:
            t = f'{col}'
            if col == 'km_driven':
                data[col] = data[col]
                t = f'{col} (в тыс.)'
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.hist(data[col], bins=20, density=True, edgecolor=colors_edges[i], color=colors_train[i])
            ax.set_title(t, fontsize=9)
            ax.tick_params(labelsize=8)
            st.pyplot(fig, use_container_width=True)
            i = (i + 1) % 5
    
    if col == 'km_driven':
                data[col] = data[col]*1000


    st.subheader('Корреляция признаков')
    col_heatmap, _ = st.columns([2, 2])
    with col_heatmap:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(data[numer].corr(), cmap="gist_earth", annot=True, vmin=-1, vmax=1, annot_kws={"size": 8}, ax=ax)
        ax.tick_params(labelsize=8)
        st.pyplot(fig, use_container_width=True)


if manual:
    with st.chat_message('Елена Андреевна', avatar='https://img.freepik.com/free-vector/exclamation-mark-message-bubble_78370-7231.jpg?semt=ais_hybrid&w=740&q=80'):
        st.write ('К сожалению, статистика данных доступна только для csv-таблиц')


if data is not None:
    st.header("Предсказание")
    try:
        num_scaled = pd.DataFrame(scaler.transform(data[numer]), columns=numer, index=data.index)
        cat_encoded = pd.DataFrame(ohe.transform(data[categor]).toarray(), columns=ohe.get_feature_names_out(), index=data.index)
        X = pd.concat([num_scaled, cat_encoded], axis = 1)
        preds = model.predict(X)
        results = data[data.columns[:5]]
        results["Predicted Price"] = preds

        styled_df = results.style.set_properties(
            subset=["Predicted Price"],
            **{"font-weight": "bold"}
        )
        st.write("Предсказанные значения:")
        st.dataframe(styled_df, use_container_width=True)

    except Exception as e:
        st.error(f"Ошибка при предсказании: {e}")


st.header("Распределение весов модели")
col_weights, _ = st.columns([4, 2])
with col_weights:
    features = numer + list(ohe.get_feature_names_out(categor))
    df = pd.DataFrame({"feature": features, "coef": model.best_estimator_.coef_/1000})
    fig, ax = plt.subplots(figsize=(4,4))
    sns.barplot(data=df.head(20), x="coef", y="feature", ax=ax, color=colors_train[2], edgecolor = colors_edges[2])
    ax.set_title("Топ-20 коэффициентов по важности (значения в тыс.)")
    st.pyplot(fig, use_container_width=True)