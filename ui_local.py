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
            dbc.Button("Загрузить данные", id='load-button', color="primary"),
        ], width=12)
    ]),
    
    html.Hr(),
    
    # Секция истории просмотров
    html.H3("История просмотров"),
    html.Div(id='history-container'),
    html.Div(id='history-images', style={
        'display': 'flex',
        'overflowX': 'auto',
        'gap': '16px',
        'padding': '10px 0',
        'marginBottom': '40px'
    }),
    
    # Секция рекомендаций
    html.H3("Рекомендации для вас"),
    html.Div(id='recommendations-container'),
    html.Div(id='recommendations-images', style={
        'display': 'flex',
        'overflowX': 'auto',
        'gap': '16px',
        'padding': '10px 0',
        'marginBottom': '20px'
    }),
    
    # Скрытые элементы для хранения данных
    dcc.Store(id='history-store', data=[]),
    dcc.Store(id='recommendations-store', data=[]),
    dcc.Store(id='user-history-store', data=[])
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

# Основной обработчик для загрузки данных
@callback(
    [Output('history-images', 'children'),
     Output('recommendations-images', 'children'),
     Output('history-store', 'data'),
     Output('recommendations-store', 'data'),
     Output('user-history-store', 'data')],
    Input('load-button', 'n_clicks'),
    State('user-id-input', 'value'),
    prevent_initial_call=True
)
def load_user_data(n_clicks, user_id):
    if not user_id:
        return (
            [dbc.Alert("Пожалуйста, введите ID пользователя", color="warning")],
            [],
            [],
            [],
            []
        )
    
    try:
        # Получаем историю из Redis
        data = r.get(f"user:{user_id}")
        if not data:
            return (
                [dbc.Alert("История пользователя не найдена", color="warning")],
                [],
                [],
                [],
                []
            )
        
        history = json.loads(data)
        article_ids = ["0" + str(article_id) for article_id in history[-10:]]
        
        # Получаем изображения для истории
        response = requests.get(
            "http://localhost:8000/recommendations/product_images/",
            params={"article_ids": article_ids}
        )
        
        if response.status_code != 200:
            history_images = [dbc.Alert("Ошибка при получении изображений", color="danger")]
        else:
            images_data = response.json()
            history_urls = [url for url in images_data.values() if url]
            history_images = []
            for i, img_data in enumerate(history_urls, 1):
                card = create_image_card(img_data, i)
                if card:
                    history_images.append(card)
        
        # Получаем рекомендации, исключая уже просмотренные товары
        response = requests.get(
            "http://localhost:8000/recommendations/personal_items/",
            params={"user_id": user_id}
        )
        
        if response.status_code != 200:
            recommendations_images = [dbc.Alert("Ошибка при получении рекомендаций", color="danger")]
            recommendation_urls = []
        else:
            recommendations = response.json()
            recommendations = recommendations.get("recommendations", [])
            
            # Фильтруем рекомендации, исключая уже просмотренные товары
            viewed_items = set(history)
            filtered_recommendations = [
                rec for rec in recommendations 
                if str(rec["item_id"]) not in viewed_items
            ][:10]  # Берем первые 10 уникальных рекомендаций
            
            article_ids = ["0" + str(pair["item_id"]) for pair in filtered_recommendations]
            
            # Получаем изображения для рекомендаций
            response = requests.get(
                "http://localhost:8000/recommendations/product_images/",
                params={"article_ids": article_ids}
            )
            
            if response.status_code != 200:
                recommendations_images = [dbc.Alert("Ошибка при получении изображений", color="danger")]
                recommendation_urls = []
            else:
                images_data = response.json()
                recommendation_urls = [url for url in images_data.values() if url]
                recommendations_images = []
                for i, img_data in enumerate(recommendation_urls, 1):
                    card = create_image_card(img_data, i)
                    if card:
                        recommendations_images.append(card)
        
        return (
            history_images,
            recommendations_images,
            history_urls if 'history_urls' in locals() else [],
            recommendation_urls if 'recommendation_urls' in locals() else [],
            history
        )
    
    except Exception as e:
        print(f"Ошибка: {e}")
        return (
            [dbc.Alert(f"Ошибка: {str(e)}", color="danger")],
            [],
            [],
            [],
            []
        )

if __name__ == '__main__':
    app.run(debug=True, port=8050)
