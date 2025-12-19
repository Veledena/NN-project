import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
from torchvision import transforms as T
import torchvision.models as models
import time
import requests
from io import BytesIO



def page_about_model2():

    # Заголовок
    st.title('📊 Анализ обучения модели')

    st.header('📂 Состав датасета')

    train_class_counts = {'sunflower': 17,
    'vigna-radiati(Mung)': 20,
    'jowar': 22,
    'almond': 15,
    'papaya': 16,
    'mustard-oil': 20,
    'rice': 23,
    'pineapple': 18,
    'tomato': 20,
    'Tobacco-plant': 27,
    'cotton': 24,
    'gram': 17,
    'banana': 25,
    'coconut': 18,
    'maize': 25,
    'Olive-tree': 23,
    'soyabean': 24,
    'sugarcane': 19,
    'jute': 16,
    'clove': 23,
    'wheat': 21,
    'chilli': 17,
    'Fox_nut(Makhana)': 16,
    'cardamom': 16,
    'Lemon': 21,
    'tea': 17,
    'Pearl_millet(bajra)': 39,
    'Cucumber': 24,
    'Cherry': 25,
    'Coffee-plant': 22}

    valid_class_counts = {'banana': 6,
    'clove': 7,
    'almond': 6,
    'chilli': 6,
    'cardamom': 6,
    'Cherry': 7,
    'coconut': 7,
    'Coffee-plant': 7,
    'Cucumber': 7,
    'Fox_nut(Makhana)': 7,
    'jute': 7,
    'jowar': 8,
    'Lemon': 7,
    'maize': 6,
    'papaya': 7,
    'mustard-oil': 8,
    'pineapple': 7,
    'gram': 8,
    'soyabean': 6,
    'sunflower': 7,
    'rice': 6,
    'Olive-tree': 7,
    'sugarcane': 6,
    'Pearl_millet(bajra)': 8,
    'cotton': 8,
    'tea': 6,
    'tomato': 6,
    'Tobacco-plant': 6,
    'vigna-radiati(Mung)': 7,
    'wheat': 10}

    plt.style.use('seaborn-v0_8')

    # Создаем два столбца
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Train Dataset")
        fig1, ax1 = plt.subplots(figsize=(16, 10))
        
        # Сортируем классы для упорядоченного отображения
        sorted_train = dict(sorted(train_class_counts.items(), key=lambda x: x[1], reverse=True))
        
        # Создаем красивый градиент цветов
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_train)))
        
        bars = ax1.bar(range(len(sorted_train)), sorted_train.values(), color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.set_ylabel('Количество изображений', fontsize=12, fontweight='bold')
        ax1.set_title(f'📦 Train Dataset: {sum(train_class_counts.values())} изображений', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Настраиваем метки на оси X
        plt.xticks(range(len(sorted_train)), list(sorted_train.keys()), 
                rotation=90, fontsize=9, ha='center')
        
        # Добавляем значения на столбцы
        for i, (bar, count) in enumerate(zip(bars, sorted_train.values())):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    f'{count}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Улучшаем внешний вид
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_axisbelow(True)
    # ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        
        st.pyplot(fig1)

    with col2:
        st.subheader("Valid Dataset")
        fig2, ax2 = plt.subplots(figsize=(16, 10))
        
        # Сортируем классы для упорядоченного отображения
        sorted_valid = dict(sorted(valid_class_counts.items(), key=lambda x: x[1], reverse=True))
        
        # Создаем красивый градиент цветов (другой цветовой схемы)
        colors = plt.cm.plasma(np.linspace(0, 1, len(sorted_valid)))
        
        bars = ax2.bar(range(len(sorted_valid)), sorted_valid.values(), color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax2.set_ylabel('Количество изображений', fontsize=12, fontweight='bold')
        ax2.set_title(f'✅ Valid Dataset: {sum(valid_class_counts.values())} изображений', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Настраиваем метки на оси X
        plt.xticks(range(len(sorted_valid)), list(sorted_valid.keys()), 
                rotation=90, fontsize=9, ha='center')
        
        # Добавляем значения на столбцы
        for i, (bar, count) in enumerate(zip(bars, sorted_valid.values())):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                    f'{count}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Улучшаем внешний вид
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.set_axisbelow(True)
        # ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        
        st.pyplot(fig2)

    # Красивая итоговая статистика
    st.divider()

    # Создаем красивый контейнер для статистики
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;'>
        <h3 style='color: #2E86AB; margin-bottom: 20px;'>📈 Итоговая статистика данных</h3>
    """, unsafe_allow_html=True)

    # Создаем колонки для статистики
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

    with stat_col1:
        st.metric("🎯 Всего классов", len(train_class_counts), delta=None)

    with stat_col2:
        total_train = sum(train_class_counts.values())
        st.metric("📦 Train изображений", total_train, delta=None)

    with stat_col3:
        total_valid = sum(valid_class_counts.values())
        st.metric("✅ Valid изображений", total_valid, delta=None)

    with stat_col4:
        total_all = total_train + total_valid
        st.metric("📊 Всего изображений", total_all, delta=None)

    # Дополнительная статистика
    st.markdown("---")
    col5, col6 = st.columns(2)

    with col5:
        st.write("**📊 Распределение по наборам:**")
        train_percent = round((total_train / total_all) * 100, 1)
        valid_percent = round((total_valid / total_all) * 100, 1)
        st.write(f"- Train: {train_percent}% ({total_train} изображений)")
        st.write(f"- Valid: {valid_percent}% ({total_valid} изображений)")

    with col6:
        st.write("**📋 Статистика по классам:**")
        avg_per_class = round(total_all / len(train_class_counts), 1)
        st.write(f"- Среднее количество: {avg_per_class} изображений/класс")
        st.write(f"- Максимальное: {max(train_class_counts.values())} (Train), {max(valid_class_counts.values())} (Valid)")
        st.write(f"- Минимальное: {min(train_class_counts.values())} (Train), {min(valid_class_counts.values())} (Valid)")

    st.markdown("</div>", unsafe_allow_html=True)

    ### Данные с кривыми и метриками

    st.set_page_config(layout="wide")

    # Заголовок
    st.title("📈 Кривые обучения модели")

    learning_curves = "images/learning_curves.png"
    metrics_file = "images/metrics.png"

    if os.path.exists(learning_curves) and os.path.exists(metrics_file):
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(Image.open(learning_curves), caption="Кривые обучения")
            
        with col2:
            st.image(Image.open(metrics_file), caption="Метрики модели")
            
            # Итоговые метрики
            st.subheader("📊 Итоговые метрики")
            st.metric("Final Train Accuracy", "0.884")
            st.metric("Final Valid Accuracy", "0.761")
        
    else:
        st.error(f"Файлы не найдены! Проверьте пути:")
        st.write(f"Путь к кривым: {os.path.abspath(learning_curves)}")
        st.write(f"Путь к метрикам: {os.path.abspath(metrics_file)}")
        st.write(f"Текущая директория: {os.getcwd()}")

    ### Время обучения

    st.title("🕔 Время обучения")
    st.write(f"**Общее время:** 10 минут")
    st.write(f"📊 Эпох: 10 | Батчей: 128")

    # Матрица ошибок


    st.set_page_config(layout="wide")
    st.title("💼 Матрица ошибок")

    matrix = 'images/confusion_matrix.png'
    st.image(Image.open(matrix), caption="Матрица ошибок")


def page_pred_model2():
    CLASSES = ['Cherry', 'Coffee-plant', 'Cucumber', 'Fox_nut(Makhana)', 
               'Lemon', 'Olive-tree', 'Pearl_millet(bajra)', 'Tobacco-plant', 
               'almond', 'banana', 'cardamom', 'chilli', 'clove', 'coconut', 
               'cotton', 'gram', 'jowar', 'jute', 'maize', 'mustard-oil', 'papaya', 
               'pineapple', 'rice', 'soyabean', 'sugarcane', 'sunflower', 'tea', 'tomato', 
               'vigna-radiati(Mung)', 'wheat']

    @st.cache_resource

    
    def load_model():
        torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential])
        return torch.load('models/my_torch_model_full.pth', map_location='cpu', weights_only=False)

    model = load_model()

    st.title("Загрузка изображения культуры и предсказание класса")
    uploaded_file = st.file_uploader("Загрузите картинку", type=["png", "jpg", "jpeg"])

    url = st.text_input("Или введите URL изображения:", "")

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Загруженная картинка', use_container_width=True)

        # Предварительная обработка изображения
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        input_tensor = transform(image).unsqueeze(0)  # Размер (1,3,224,224)

    if url:
        response = requests.get(url)
        response.raise_for_status()  # Проверка на ошибки HTTP
        image = Image.open(BytesIO(response.content))

        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # проверьте, подходит ли этот размер
            transforms.ToTensor(),
        ])
        input_tensor = transform(image).unsqueeze(0)  # Размер (1,3,224,224)
        
        # Показываем изображение
        st.image(image, caption="Загруженное изображение", use_container_width=True)

        # Предсказание
        try:
            with torch.no_grad():
                outputs = model(input_tensor)
                
                # Вычисляем вероятности
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Получаем топ-3 предсказания
                top3_probs, top3_indices = torch.topk(probabilities, 3)
                
                st.write(f"Предсказанный класс: {CLASSES[top3_indices[0][0].item()]}")
                st.write("Топ-3 предсказания:")
                
                for i in range(3):
                    class_idx = top3_indices[0][i].item()
                    prob = top3_probs[0][i].item()
                    st.write(f"{i+1}. {CLASSES[class_idx]}: {prob:.4f} ({prob*100:.2f}%)")
        
        except Exception as e:
            st.write(f"Ошибка при предсказании: {e}")

def page_pred_model1():
    CLASSES1 = ["Dark", "Green", "Light", "Medium"]

    @st.cache_resource
    def load_model1():
        return torch.load('models/coffee-beans_model.pt', map_location='cpu', weights_only=False)

    model1 = load_model1()

    st.title("Загрузка изображения зерна и предсказание класса")
    uploaded_file = st.file_uploader("Загрузите картинку", type=["png", "jpg", "jpeg"])

    url = st.text_input("Или введите URL изображения:", "")

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Загруженная картинка', use_container_width=True)

        # Предварительная обработка изображения
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # проверьте, подходит ли этот размер
            transforms.ToTensor(),
        ])
        input_tensor = transform(image).unsqueeze(0)  # Размер (1,3,224,224)

    if url:
        response = requests.get(url)
        response.raise_for_status()  # Проверка на ошибки HTTP
        image = Image.open(BytesIO(response.content))

        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # проверьте, подходит ли этот размер
            transforms.ToTensor(),
        ])
        input_tensor = transform(image).unsqueeze(0)  # Размер (1,3,224,224)
        
        # Показываем изображение
        st.image(image, caption="Загруженное изображение", use_container_width=True)



        try:
            with torch.no_grad():
                outputs = model1(input_tensor)
                
                # Вычисляем вероятности
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Получаем топ-3 предсказания
                top3_probs, top3_indices = torch.topk(probabilities, 3)
                
                st.write(f"Предсказанный класс: {CLASSES1[top3_indices[0][0].item()]}")
                st.write("Топ-3 предсказания:")
                
                for i in range(3):
                    class_idx = top3_indices[0][i].item()
                    prob = top3_probs[0][i].item()
                    st.write(f"{i+1}. {CLASSES1[class_idx]}: {prob:.4f} ({prob*100:.2f}%)")
        
        except Exception as e:
            st.write(f"Ошибка при предсказании: {e}")

def about_model1():
    # ===== КОНФИГУРАЦИЯ СТРАНИЦЫ =====
    st.set_page_config(
        page_title="Coffee Beans Model Training Report", page_icon="☕", layout="wide"
    )

    # ===== ЗАГОЛОВОК =====
    st.title("🗂️ Отчёт об обучении модели классификации кофейных зёрен")
    st.markdown("---")

    # ===== РАЗДЕЛ 1: ИНФОРМАЦИЯ О МОДЕЛИ =====
    st.header("🔍 Информация о модели")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Архитектура", "ShuffleNetV2 x1.0")
        st.caption("Pretrained на ImageNet")
    with col2:
        st.metric("Всего параметров", "~1.2M")
        st.caption("Обучаемых: 4,100 (только fc)")
    with col3:
        st.metric("Размер модели", "4.82 MB")
        st.caption("+ 1.3 GB активаций")

    st.markdown(
        "**Стратегия обучения:** Transfer Learning (заморожены все слои кроме финального)"
    )

    # Архитектура модели
    with st.expander("📐 Детальная архитектура модели"):
        st.code(
            """
    ShuffleNetV2(
    (conv1): Conv2d(3 → 24, kernel=3×3, stride=2)
    (maxpool): MaxPool2d(kernel=3, stride=2)
    (stage2): 4× InvertedResidual (24 → 58 каналов)
    (stage3): 8× InvertedResidual (58 → 232 каналов)
    (stage4): 4× InvertedResidual (232 → 464 каналов)
    (conv5): Conv2d(464 → 1024, kernel=1×1)
    (fc): Linear(1024 → 4 класса) ← ОБУЧАЕМЫЙ
    )
        """,
            language="text",
        )

    st.markdown("---")

    # ===== РАЗДЕЛ 2: СОСТАВ ДАТАСЕТА =====
    st.header("📦 Состав датасета")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Всего изображений", "1,600")
    with col2:
        st.metric("Train", "1,200 (75%)")
    with col3:
        st.metric("Validation", "400 (25%)")
    with col4:
        st.metric("Размер изображений", "224×224")

    # Распределение по классам
    st.subheader("📊 Распределение по классам")
    class_distribution = pd.DataFrame(
        {
            "Класс": ["Dark", "Green", "Light", "Medium"],
            "Количество объектов": [400, 400, 400, 400],
            "Процент": ["25%", "25%", "25%", "25%"],
        }
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#8B4513", "#228B22", "#F4A460", "#D2691E"]
        ax.bar(
            class_distribution["Класс"],
            class_distribution["Количество объектов"],
            color=colors,
        )
        ax.set_ylabel("Количество изображений")
        ax.set_title("Распределение объектов по классам")
        ax.set_ylim(0, 500)
        for i, v in enumerate(class_distribution["Количество объектов"]):
            ax.text(i, v + 10, str(v), ha="center", fontweight="bold")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.dataframe(class_distribution, hide_index=True, use_container_width=True)

    # Параметры обработки данных
    with st.expander("⚙️ Параметры обработки данных"):
        st.markdown(
            """
        **Трансформации для обучения:**
        - `Resize(224×224)`
        - `RandomRotation(45°)`
        - `RandomHorizontalFlip()`
        - `ToTensor()`
        - `Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])`
        
        **Трансформации для валидации:**
        - `Resize(224×224)`
        - `ToTensor()`
        - `Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])`
        
        **DataLoader:**
        - Batch size: 64
        - Num_classes: 4
        """
        )

    st.markdown("---")

    # ===== РАЗДЕЛ 3: ПАРАМЕТРЫ ОБУЧЕНИЯ =====
    st.header("🎓 Параметры обучения")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Эпохи", "5 / 5")
    with col2:
        st.metric("Learning Rate", "0.005")
        st.caption("Оптимизатор: Adam")
    with col3:
        st.metric("Loss Function", "CrossEntropyLoss")
    with col4:
        st.metric("Batch Size", "64")

    st.markdown("---")

    # ===== РАЗДЕЛ 4: КРИВЫЕ ОБУЧЕНИЯ =====
    st.header("📈 Кривые обучения")

    # Данные из обучения (из скриншота №8)
    train_losses = [1.2666, 1.0626, 0.9233, 0.7460, 0.3386]
    valid_losses = [1.2313, 1.0458, 0.9217, 0.7668, 0.6890]
    train_accs = [0.7023, 0.8460, 0.8913, 0.9518, 0.9055]
    valid_accs = [0.7746, 0.8584, 0.9001, 0.9665, 0.9598]
    epochs = list(range(5))

    col1, col2 = st.columns(2)

    with col1:
        # График Loss
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            epochs,
            train_losses,
            marker="o",
            linewidth=2,
            label="Train Loss",
            color="#1f77b4",
        )
        ax.plot(
            epochs,
            valid_losses,
            marker="s",
            linewidth=2,
            label="Valid Loss",
            color="#ff7f0e",
        )
        ax.set_xlabel("Эпоха", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Loss History", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col2:
        # График Accuracy
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            epochs,
            train_accs,
            marker="o",
            linewidth=2,
            label="Train Accuracy",
            color="#1f77b4",
        )
        ax.plot(
            epochs,
            valid_accs,
            marker="s",
            linewidth=2,
            label="Valid Accuracy",
            color="#ff7f0e",
        )
        ax.set_xlabel("Эпоха", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Accuracy History", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.6, 1.0)
        st.pyplot(fig)
        plt.close()

    # Метрики в табличном виде
    st.subheader("📊 Метрики по эпохам")
    metrics_df = pd.DataFrame(
        {
            "Эпоха": [f"Epoch {i:02d}" for i in epochs],
            "Train Loss": train_losses,
            "Valid Loss": valid_losses,
            "Train Accuracy": [f"{acc:.2%}" for acc in train_accs],
            "Valid Accuracy": [f"{acc:.2%}" for acc in valid_accs],
        }
    )
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    # Финальные метрики
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Final Train Loss", f"{train_losses[-1]:.4f}")
    with col2:
        st.metric("Final Valid Loss", f"{valid_losses[-1]:.4f}")
    with col3:
        st.metric("Final Train Accuracy", f"{train_accs[-1]:.2%}")
    with col4:
        st.metric("Final Valid Accuracy", f"{valid_accs[-1]:.2%}", delta="+19.52%")

    st.info(
        "💡 **Наблюдение:** Validation accuracy выше train accuracy — признак хорошей генерализации модели!"
    )

    st.markdown("---")

    # ===== РАЗДЕЛ 5: МЕТРИКИ КЛАССИФИКАЦИИ =====
    st.header("🎯 Метрики классификации")

    st.subheader("📋 Classification Report")

    # Данные без стилизации
    report_data = {
        "Class": ["Dark", "Green", "Light", "Medium", "", "Weighted Avg"],
        "Precision": [0.97, 0.95, 0.94, 0.98, None, 0.96],
        "Recall": [0.96, 0.97, 0.93, 0.96, None, 0.96],
        "F1-Score": [0.96, 0.96, 0.94, 0.97, None, 0.96],
        "Support": [100, 100, 100, 100, None, 400],
    }
    report_df = pd.DataFrame(report_data)

    # Простое отображение без стилей
    st.dataframe(report_df, hide_index=True, use_container_width=True)

    # Визуализация метрик через график (альтернатива)
    fig, ax = plt.subplots(figsize=(10, 5))
    classes_only = report_df[report_df["Class"].isin(["Dark", "Green", "Light", "Medium"])]
    x = np.arange(len(classes_only))
    width = 0.25

    ax.bar(x - width, classes_only["Precision"], width, label="Precision", color="#1f77b4")
    ax.bar(x, classes_only["Recall"], width, label="Recall", color="#ff7f0e")
    ax.bar(x + width, classes_only["F1-Score"], width, label="F1-Score", color="#2ca02c")

    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Метрики по классам")
    ax.set_xticks(x)
    ax.set_xticklabels(classes_only["Class"])
    ax.legend()
    ax.set_ylim(0.9, 1.0)
    ax.grid(axis="y", alpha=0.3)

    st.pyplot(fig)
    plt.close()


    # Итоговые метрики
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Weighted Precision", "0.9600")
    with col2:
        st.metric("Weighted Recall", "0.9598")
    with col3:
        st.metric("Weighted F1-Score", "0.9599")

    st.markdown("---")

    # ===== РАЗДЕЛ 7: ВРЕМЯ ОБУЧЕНИЯ =====
    st.header("⏱️ Время обучения")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Общее время", "~93 секунд")
        st.caption("≈ 1 минута 33 секунды")
    with col2:
        st.metric("Время на эпоху", "~18.6 сек")
        st.caption("Среднее по 5 эпохам")
    with col3:
        st.metric("Устройство", "CUDA")
        st.caption("GPU-ускорение")

    st.markdown("---")

    # ===== FOOTER =====
    st.success("✅ Модель готова к использованию! Validation Accuracy: **95.98%**")

    with st.expander("💾 Сохранение модели"):
        st.code(
            """
    # Сохранение весов модели
    torch.save(model.state_dict(), "coffee-beans_model.pt")

    # Загрузка модели
    model = models.shufflenet_v2_x1_0(pretrained=False)
    model.fc = nn.Linear(1024, 4)
    model.load_state_dict(torch.load("coffee-beans_model.pt"))
    model.eval()
        """,
            language="python",
        )   



page = st.sidebar.radio("Выберите страницу", [ 'стр.1- Про модель "Степень прожарки зерён"', 'стр.2- Про модель "Сельскохозяйственные культуры"', 
                                              'стр.3- Предсказание модели "Степень прожарки зерён"', 'стр.4- Предсказание модели "Сельскохозяйственные культуры"'])

# Вызов нужной функции по выбору
if page == 'стр.2- Про модель "Сельскохозяйственные культуры"':
    page_about_model2()
elif page == 'стр.4- Предсказание модели "Сельскохозяйственные культуры"':
    page_pred_model2()
elif page == 'стр.3- Предсказание модели "Степень прожарки зерён"':
    page_pred_model1()
elif page == 'стр.1- Про модель "Степень прожарки зерён"':
    about_model1()

