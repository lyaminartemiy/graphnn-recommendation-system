import streamlit as st
import redis
import requests
import json
from io import BytesIO
import base64
from PIL import Image

# Подключение к Redis
r = redis.Redis(
    host="localhost",
    password="redis123",
    port=6379,
    decode_responses=True,
)

# Функция для отображения изображений
def display_images_base64(images_data, max_images=5, image_width=200):
    """Отображает изображения в горизонтальной ленте с прокруткой"""
    if not images_data:
        st.warning("Нет данных для отображения")
        return
    
    # Создаем контейнер с горизонтальным скроллом
    st.markdown(
        """
        <style>
        .horizontal-scroll-container {
            display: flex;
            overflow-x: auto;
            gap: 16px;
            padding: 10px 0;
            margin-bottom: 20px;
        }
        .image-wrapper {
            flex: 0 0 auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Начало горизонтального контейнера
    st.markdown('<div class="horizontal-scroll-container">', unsafe_allow_html=True)
    
    for i, img_data in enumerate(images_data[:max_images], 1):
        try:
            if img_data is None:
                continue
            
            img_bytes = base64.b64decode(img_data.split(',')[-1])
            img = Image.open(BytesIO(img_bytes))
            
            # Обертка для каждого изображения
            st.markdown('<div class="image-wrapper">', unsafe_allow_html=True)
            st.image(
                img,
                width=image_width,
                caption=f"Товар {i}",
                output_format='JPEG'
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Ошибка загрузки изображения {i}: {e}")
    
    # Закрываем контейнер
    st.markdown('</div>', unsafe_allow_html=True)


# Настройка страницы
st.set_page_config(page_title="Рекомендательная система", layout="wide")
st.title("Рекомендательная система")

# Ввод ID пользователя
user_id = st.text_input("Введите ID пользователя", key="user_id")

# Разделы для истории и рекомендаций
tab1, tab2 = st.tabs(["История просмотров", "Рекомендации"])

with tab1:
    if st.button("Получить историю просмотров"):
        if not user_id:
            st.warning("Пожалуйста, введите ID пользователя")
        else:
            with st.spinner("Загрузка истории..."):
                try:
                    # Получаем историю из Redis
                    data = r.get(f"user:{user_id}")
                    if data:
                        history = json.loads(data)
                        article_ids = ["0" + str(article_id) for article_id in history[-10:]]  # Последние 3 просмотренных
                        
                        # Получаем URL изображений
                        response = requests.get(
                            "http://localhost:8000/recommendations/product_images/",
                            params={"article_ids": article_ids}
                        )
                        
                        if response.status_code == 200:
                            images_data = response.json()
                            image_urls = [url for url in images_data.values() if url]
                            display_images_base64(image_urls)
                        else:
                            st.error("Ошибка при получении изображений")
                    else:
                        st.warning("История пользователя не найдена")
                except Exception as e:
                    st.error(f"Ошибка: {str(e)}")

with tab2:
    if st.button("Сформировать рекомендации"):
        if not user_id:
            st.warning("Пожалуйста, введите ID пользователя")
        else:
            with st.spinner("Формирование рекомендаций..."):
                try:
                    # Получаем рекомендации
                    response = requests.get(
                        "http://localhost:8000/recommendations/personal_items/",
                        params={"user_id": user_id}
                    )
                    
                    if response.status_code == 200:
                        recommendations = response.json()
                        article_ids = recommendations.get('recommended_items', [])[:5]  # Первые 5 рекомендаций
                        
                        # Получаем URL изображений
                        response = requests.get(
                            "http://localhost:8000/product_images/",
                            params={"article_ids": article_ids}
                        )
                        
                        if response.status_code == 200:
                            images_data = response.json()
                            image_urls = [url for url in images_data.values() if url]
                            display_images_base64(image_urls)
                        else:
                            st.error("Ошибка при получении изображений")
                    else:
                        st.error("Ошибка при получении рекомендаций")
                except Exception as e:
                    st.error(f"Ошибка: {str(e)}")
