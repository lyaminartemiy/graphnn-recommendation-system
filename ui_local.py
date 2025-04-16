import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import redis
import requests
import json
from io import BytesIO
import base64
from PIL import Image

# Инициализация приложения Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Рекомендательная система"

# Подключение к Redis
r = redis.Redis(
    host="localhost",
    password="redis123",
    port=6379,
    decode_responses=True,
)

# Макет приложения
app.layout = dbc.Container([
    html.H1("Рекомендательная система", className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Input(id='user-id-input', placeholder="Введите ID пользователя", type="text"),
            html.Br(),
            dbc.Button("Получить историю просмотров", id='history-button', color="primary", className="me-2"),
            dbc.Button("Сформировать рекомендации", id='recommend-button', color="success"),
        ], width=12)
    ]),
    
    html.Hr(),
    
    # Табы для истории и рекомендаций
    dbc.Tabs([
        dbc.Tab([
            html.Div(id='history-container'),
            html.Div(id='history-images', style={
                'display': 'flex',
                'overflowX': 'auto',
                'gap': '16px',
                'padding': '10px 0',
                'marginBottom': '20px'
            })
        ], label="История просмотров"),
        
        dbc.Tab([
            html.Div(id='recommendations-container'),
            html.Div(id='recommendations-images', style={
                'display': 'flex',
                'overflowX': 'auto',
                'gap': '16px',
                'padding': '10px 0',
                'marginBottom': '20px'
            })
        ], label="Рекомендации")
    ]),
    
    # Скрытый элемент для хранения данных
    dcc.Store(id='image-store', data={'history': [], 'recommendations': []})
], fluid=True)

# Функция для создания карточки с изображением
def create_image_card(img_data, index, width=200):
    try:
        if img_data is None:
            return None
            
        img_bytes = base64.b64decode(img_data.split(',')[-1])
        img = Image.open(BytesIO(img_bytes))
        
        # Сохраняем изображение в буфер
        buf = BytesIO()
        img.save(buf, format='JPEG')
        encoded_img = base64.b64encode(buf.getvalue()).decode('ascii')
        
        return dbc.Card([
            dbc.CardImg(src=f"data:image/jpeg;base64,{encoded_img}", style={'width': f'{width}px'}),
            dbc.CardBody([
                html.P(f"Товар {index}", className="card-text")
            ])
        ], style={'width': f'{width + 20}px', 'flexShrink': '0'})
    
    except Exception as e:
        print(f"Ошибка загрузки изображения {index}: {e}")
        return None

# Обработчик для получения истории просмотров
@callback(
    [Output('history-images', 'children'),
     Output('image-store', 'data', allow_duplicate=True)],
    Input('history-button', 'n_clicks'),
    [State('user-id-input', 'value'),
     State('image-store', 'data')],
    prevent_initial_call=True
)
def get_user_history(n_clicks, user_id, store_data):
    if not user_id:
        return [dbc.Alert("Пожалуйста, введите ID пользователя", color="warning")], dash.no_update
    
    try:
        # Получаем историю из Redis
        data = r.get(f"user:{user_id}")
        if not data:
            return [dbc.Alert("История пользователя не найдена", color="warning")], dash.no_update
        
        history = json.loads(data)
        article_ids = ["0" + str(article_id) for article_id in history[-10:]]
        
        # Получаем URL изображений
        response = requests.get(
            "http://localhost:8000/recommendations/product_images/",
            params={"article_ids": article_ids}
        )
        
        if response.status_code != 200:
            return [dbc.Alert("Ошибка при получении изображений", color="danger")], dash.no_update
        
        images_data = response.json()
        image_urls = [url for url in images_data.values() if url]
        
        # Создаем карточки с изображениями
        cards = []
        for i, img_data in enumerate(image_urls, 1):
            card = create_image_card(img_data, i)
            if card:
                cards.append(card)
        
        # Обновляем хранилище
        store_data['history'] = image_urls
        return [cards, store_data]
    
    except Exception as e:
        print(f"Ошибка: {e}")
        return [dbc.Alert(f"Ошибка: {str(e)}", color="danger")], dash.no_update

# Обработчик для получения рекомендаций
@callback(
    [Output('recommendations-images', 'children'),
     Output('image-store', 'data')],
    Input('recommend-button', 'n_clicks'),
    [State('user-id-input', 'value'),
     State('image-store', 'data')],
    prevent_initial_call=True
)
def get_recommendations(n_clicks, user_id, store_data):
    if not user_id:
        return [dbc.Alert("Пожалуйста, введите ID пользователя", color="warning")], dash.no_update
    
    try:
        # Получаем рекомендации
        response = requests.get(
            "http://localhost:8000/recommendations/personal_items/",
            params={"user_id": user_id}
        )
        
        if response.status_code != 200:
            return [dbc.Alert("Ошибка при получении рекомендаций", color="danger")], dash.no_update
        
        recommendations = response.json()
        recommendations = recommendations.get("recommendations")
        article_ids = ["0" + str(pair["item_id"]) for pair in recommendations[-10:]]
        print(article_ids)
        
        # Получаем URL изображений
        response = requests.get(
            "http://localhost:8000/recommendations/product_images/",
            params={"article_ids": article_ids}
        )
        
        if response.status_code != 200:
            return [dbc.Alert("Ошибка при получении изображений", color="danger")], dash.no_update
        
        images_data = response.json()
        image_urls = [url for url in images_data.values() if url]
        
        # Создаем карточки с изображениями
        cards = []
        for i, img_data in enumerate(image_urls, 1):
            card = create_image_card(img_data, i)
            if card:
                cards.append(card)
        
        # Обновляем хранилище
        store_data['recommendations'] = image_urls
        return [cards, store_data]
    
    except Exception as e:
        print(f"Ошибка: {e}")
        return [dbc.Alert(f"Ошибка: {str(e)}", color="danger")], dash.no_update


if __name__ == '__main__':
    app.run(debug=True, port=8050)
