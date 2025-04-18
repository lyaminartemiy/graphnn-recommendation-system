import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import requests
from io import BytesIO
import base64
from PIL import Image
import configparser


# Загрузка конфигурации
config = configparser.ConfigParser()
config.read("../config.ini")

# Конфигурация API
API_CONFIG = {
    "base_url": config.get("API", "base_url", fallback="http://recommendation-service:8000"),
    "product_images_endpoint": config.get(
        "API", "product_images_endpoint", fallback="/recommendations/product_images/"
    ),
    "personal_items_endpoint": config.get(
        "API", "personal_items_endpoint", fallback="/recommendations/personal_items/"
    ),
    "user_events_endpoint": config.get(
        "API", "user_events_endpoint", fallback="/recommendations/events/"
    ),
}

# Настройки приложения
APP_CONFIG = {
    "title": config.get("App", "title", fallback="Рекомендательная система"),
    "debug": config.getboolean("App", "debug", fallback=True),
    "host": config.get("App", "host", fallback="0.0.0.0"),
    "port": config.getint("App", "port", fallback=8050),
    "history_limit": config.getint("App", "history_limit", fallback=10),
    "recommendations_limit": config.getint("App", "recommendations_limit", fallback=10),
    "image_width": config.getint("App", "image_width", fallback=200),
}

# Инициализация приложения Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = APP_CONFIG["title"]


def get_full_api_url(endpoint_key):
    """Возвращает полный URL для API endpoint"""
    return f"{API_CONFIG['base_url']}{API_CONFIG[endpoint_key]}"


def get_user_history(user_id):
    """Получает историю событий пользователя через API"""
    if not user_id:
        return None

    response = requests.get(
        get_full_api_url("user_events_endpoint"), params={"user_id": user_id}
    )

    if response.status_code != 200:
        return None

    data = response.json()
    return [str(event) for event in data.get("events", [])]


def get_product_images(article_ids):
    """Получает изображения товаров по их ID"""
    if not article_ids:
        return {}

    response = requests.get(
        get_full_api_url("product_images_endpoint"), params={"article_ids": article_ids}
    )
    return response.json() if response.status_code == 200 else {}


def get_recommendations(user_id, exclude_items=None):
    """Получает персонализированные рекомендации для пользователя"""
    if not user_id:
        return []

    response = requests.get(
        get_full_api_url("personal_items_endpoint"), params={"user_id": user_id}
    )

    if response.status_code != 200:
        return []

    recommendations = response.json().get("recommendations", [])

    # Фильтрация уже просмотренных товаров
    if exclude_items:
        exclude_set = set(str(item) for item in exclude_items)
        recommendations = [
            rec for rec in recommendations if str(rec["item_id"]) not in exclude_set
        ]

    return recommendations[: APP_CONFIG["recommendations_limit"]]


def create_image_card(img_data, index):
    """Создает карточку с изображением товара"""
    try:
        if not img_data:
            return None

        img_bytes = base64.b64decode(img_data.split(",")[-1])
        img = Image.open(BytesIO(img_bytes))

        # Сохраняем изображение в буфер
        buf = BytesIO()
        img.save(buf, format="JPEG")
        encoded_img = base64.b64encode(buf.getvalue()).decode("ascii")

        width = APP_CONFIG["image_width"]

        return dbc.Card(
            [
                dbc.CardImg(
                    src=f"data:image/jpeg;base64,{encoded_img}",
                    style={"width": f"{width}px"},
                ),
                dbc.CardBody([html.P(f"Товар {index}", className="card-text")]),
            ],
            style={"width": f"{width + 20}px", "flexShrink": "0"},
        )

    except Exception as e:
        print(f"Ошибка загрузки изображения {index}: {e}")
        return None


def create_image_cards_from_urls(urls):
    """Создает список карточек с изображениями из URL"""
    return [
        card for i, url in enumerate(urls, 1) if (card := create_image_card(url, i))
    ]


# Макет приложения
app.layout = dbc.Container(
    [
        html.H1(APP_CONFIG["title"], className="mb-4"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Input(
                            id="user-id-input",
                            placeholder="Введите ID пользователя",
                            type="text",
                        ),
                        html.Br(),
                        dbc.Button(
                            "Загрузить данные", id="load-button", color="primary"
                        ),
                    ],
                    width=12,
                )
            ]
        ),
        html.Hr(),
        # Секция истории просмотров
        html.H3("История просмотров"),
        html.Div(id="history-container"),
        html.Div(
            id="history-images",
            style={
                "display": "flex",
                "overflowX": "auto",
                "gap": "16px",
                "padding": "10px 0",
                "marginBottom": "40px",
            },
        ),
        # Секция рекомендаций
        html.H3("Рекомендации для вас"),
        html.Div(id="recommendations-container"),
        html.Div(
            id="recommendations-images",
            style={
                "display": "flex",
                "overflowX": "auto",
                "gap": "16px",
                "padding": "10px 0",
                "marginBottom": "20px",
            },
        ),
        # Скрытые элементы для хранения данных
        dcc.Store(id="history-store", data=[]),
        dcc.Store(id="recommendations-store", data=[]),
        dcc.Store(id="user-history-store", data=[]),
    ],
    fluid=True,
)


# Основной обработчик для загрузки данных
@callback(
    [
        Output("history-images", "children"),
        Output("recommendations-images", "children"),
        Output("history-store", "data"),
        Output("recommendations-store", "data"),
        Output("user-history-store", "data"),
    ],
    Input("load-button", "n_clicks"),
    State("user-id-input", "value"),
    prevent_initial_call=True,
)
def load_user_data(n_clicks, user_id):
    if not user_id:
        return (
            [dbc.Alert("Пожалуйста, введите ID пользователя", color="warning")],
            [],
            [],
            [],
            [],
        )

    try:
        # Получаем историю пользователя через API
        history = get_user_history(user_id)
        if not history:
            return (
                [dbc.Alert("История пользователя не найдена", color="warning")],
                [],
                [],
                [],
                [],
            )

        # Получаем изображения для истории
        article_ids = [
            str(article_id) for article_id in history[-APP_CONFIG["history_limit"] :]
        ]
        images_data = get_product_images(article_ids)
        history_urls = [url for url in images_data.values() if url]
        history_images = create_image_cards_from_urls(history_urls)

        # Получаем рекомендации
        recommendations = get_recommendations(user_id, exclude_items=history)
        recommendation_ids = [str(pair["item_id"]) for pair in recommendations]

        # Получаем изображения для рекомендаций
        images_data = get_product_images(recommendation_ids)
        recommendation_urls = [url for url in images_data.values() if url]
        recommendations_images = create_image_cards_from_urls(recommendation_urls)

        return (
            history_images or [dbc.Alert("Нет данных для отображения", color="info")],
            recommendations_images
            or [dbc.Alert("Нет рекомендаций для отображения", color="info")],
            history_urls,
            recommendation_urls,
            history,
        )

    except Exception as e:
        print(f"Ошибка: {e}")
        return ([dbc.Alert(f"Ошибка: {str(e)}", color="danger")], [], [], [], [])


if __name__ == "__main__":
    app.run(
        host=APP_CONFIG["host"],
        port=APP_CONFIG["port"],
        debug=APP_CONFIG["debug"],
    )
