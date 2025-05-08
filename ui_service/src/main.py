import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from datetime import datetime
import requests
from io import BytesIO
import base64
from PIL import Image
import configparser
import json
import pika

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

FEEDBACK_CONFIG = {
    "rabbitmq_host": config.get("RabbitMQ", "host", fallback="rabbitmq"),
    "rabbitmq_port": config.getint("RabbitMQ", "port", fallback=5672),
    "rabbitmq_user": config.get("RabbitMQ", "user", fallback="rabbit"),
    "rabbitmq_password": config.get("RabbitMQ", "password", fallback="rabbit123"),
    "feedback_queue": config.get("RabbitMQ", "feedback_queue", fallback="user_feedback"),
}

def send_feedback_to_rabbitmq(user_id, item_id, action):
    """Отправляет фидбек пользователя в RabbitMQ"""
    print("ОТПРАВЛЯЕМ ЭВЕНТ:", user_id, item_id, action)
    try:
        credentials = pika.PlainCredentials(
            FEEDBACK_CONFIG["rabbitmq_user"], 
            FEEDBACK_CONFIG["rabbitmq_password"]
        )
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=FEEDBACK_CONFIG["rabbitmq_host"],
                port=FEEDBACK_CONFIG["rabbitmq_port"],
                credentials=credentials
            )
        )
        channel = connection.channel()
        channel.queue_declare(queue=FEEDBACK_CONFIG["feedback_queue"], durable=True)
        
        message = {
            "user_id": str(user_id),
            "item_id": str(item_id),
            "action": action,  # 'like' или 'dislike'
            "timestamp": datetime.now().isoformat(),
        }
        
        channel.basic_publish(
            exchange='',
            routing_key=FEEDBACK_CONFIG["feedback_queue"],
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # make message persistent
            )
        )
        
        connection.close()
        return True
    except Exception as e:
        print(f"Ошибка отправки фидбека: {e}")
        return False


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

    if exclude_items:
        exclude_set = set(str(item) for item in exclude_items)
        recommendations = [
            rec for rec in recommendations if str(rec["item_id"]) not in exclude_set
        ]

    return recommendations[: APP_CONFIG["recommendations_limit"]]


def create_image_card(img_data, item_id, index):
    try:
        if not img_data:
            return None

        img_bytes = base64.b64decode(img_data.split(",")[-1])
        img = Image.open(BytesIO(img_bytes))

        buf = BytesIO()
        img.save(buf, format="JPEG")
        encoded_img = base64.b64encode(buf.getvalue()).decode("ascii")

        width = APP_CONFIG["image_width"]

        return dbc.Card(
            [
                dbc.CardImg(
                    src=f"data:image/jpeg;base64,{encoded_img}",
                    style={
                        "width": f"{width}px",
                        "height": f"{width}px",
                        "objectFit": "cover",
                        "borderTopLeftRadius": "12px",
                        "borderTopRightRadius": "12px"
                    },
                ),
                dbc.CardBody([
                    html.Div(f"Модель {index}", style={
                        "textAlign": "center",
                        "fontSize": "0.9rem",
                        "fontWeight": "600",
                        "marginBottom": "0.75rem"
                    }),
                    dbc.ButtonGroup([
                        dbc.Button(
                            "❤️",
                            id={"type": "like-btn", "index": item_id},
                            color="light",
                            size="sm",
                            style={
                                "border": "1px solid #ccc",
                                "borderRadius": "8px",
                                "width": "40px",
                                "height": "40px",
                                "padding": "0",
                                "fontSize": "1.2rem",
                                "boxShadow": "0 1px 3px rgba(0,0,0,0.1)"
                            },
                            className="me-1"
                        ),
                        dbc.Button(
                            "✖️",
                            id={"type": "dislike-btn", "index": item_id},
                            color="light",
                            size="sm",
                            style={
                                "border": "1px solid #ccc",
                                "borderRadius": "8px",
                                "width": "40px",
                                "height": "40px",
                                "padding": "0",
                                "fontSize": "1.2rem",
                                "boxShadow": "0 1px 3px rgba(0,0,0,0.1)"
                            },
                        ),
                    ], style={"justifyContent": "center", "display": "flex"})
                ], style={"padding": "0.75rem"})
            ],
            style={
                "width": f"{width}px",
                "border": "none",
                "borderRadius": "12px",
                "overflow": "hidden",
                "backgroundColor": "#fff",
                "boxShadow": "0 4px 12px rgba(0,0,0,0.08)",
                "transition": "transform 0.2s ease-in-out"
            },
        )

    except Exception as e:
        print(f"Ошибка загрузки изображения {index}: {e}")
        return None


def create_history_card(img_data, index):
    try:
        if not img_data:
            return None

        img_bytes = base64.b64decode(img_data.split(",")[-1])
        img = Image.open(BytesIO(img_bytes))

        buf = BytesIO()
        img.save(buf, format="JPEG")
        encoded_img = base64.b64encode(buf.getvalue()).decode("ascii")

        width = APP_CONFIG["image_width"]

        return dbc.Card(
            [
                dbc.CardImg(
                    src=f"data:image/jpeg;base64,{encoded_img}",
                    style={
                        "width": f"{width}px",
                        "height": f"{width}px",
                        "objectFit": "cover",
                        "borderTopLeftRadius": "12px",
                        "borderTopRightRadius": "12px"
                    },
                ),
                dbc.CardBody([
                    html.Div(f"Модель {index}", style={
                        "textAlign": "center",
                        "fontSize": "0.9rem",
                        "fontWeight": "600"
                    })
                ], style={"padding": "0.75rem"})
            ],
            style={
                "width": f"{width}px",
                "border": "none",
                "borderRadius": "12px",
                "overflow": "hidden",
                "backgroundColor": "#fff",
                "boxShadow": "0 4px 12px rgba(0,0,0,0.08)",
                "transition": "transform 0.2s ease-in-out"
            },
        )

    except Exception as e:
        print(f"Ошибка загрузки изображения {index}: {e}")
        return None


# Макет приложения
app.layout = dbc.Container(
    [
        # Шапка сайта
        html.Div(
            [
                html.H2(APP_CONFIG["title"], style={
                    "margin": "0",
                    "fontWeight": "700",
                    "fontSize": "2rem",
                    "letterSpacing": "0.5px",
                    "color": "#222"
                }),
            ],
            style={
                "backgroundColor": "#f8f8f8",
                "padding": "20px",
                "borderRadius": "12px",
                "marginBottom": "30px",
                "boxShadow": "0 2px 6px rgba(0,0,0,0.08)",
                "textAlign": "center"
            }
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Input(
                            id="user-id-input",
                            placeholder="Введите ваш ID для персональных рекомендаций",
                            type="text",
                            debounce=True,
                            style={
                                "borderRadius": "8px",
                                "border": "1px solid #ccc",
                                "padding": "12px"
                            }
                        ),
                        html.Br(),
                        dbc.Button(
                            "Показать рекомендации",
                            id="load-button",
                            color="secondary",
                            className="w-100",
                            size="lg",
                            style={
                                "borderRadius": "8px",
                                "fontWeight": "600",
                                "padding": "12px",
                                "backgroundColor": "#333",
                                "border": "none"
                            }
                        ),
                    ],
                    width=5, className="mx-auto"
                )
            ],
            className="mb-5"
        ),

        html.Hr(),

        html.H4("Вы смотрели ранее", className="mt-5 mb-3", style={
            "fontWeight": "600",
            "letterSpacing": "0.3px"
        }),
        html.Div(
            id="history-images",
            style={
                "display": "flex",
                "overflowX": "auto",
                "gap": "20px",
                "padding": "8px 0 30px 0",
                "scrollBehavior": "smooth",
                "WebkitOverflowScrolling": "touch",
                "paddingBottom": "16px"
            },
        ),

        html.H4("Рекомендуем вам", className="mt-5 mb-3", style={
            "fontWeight": "600",
            "letterSpacing": "0.3px"
        }),
        html.Div(
            id="recommendations-images",
            style={
                "display": "flex",
                "overflowX": "auto",
                "gap": "20px",
                "padding": "8px 0 30px 0",
                "scrollBehavior": "smooth",
                "WebkitOverflowScrolling": "touch",
                "paddingBottom": "16px"
            },
        ),

        html.Div(id="dummy-output", style={"display": "none"}),

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
        # Получаем историю пользователя
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
        article_ids = history[-APP_CONFIG["history_limit"] :]
        images_data = get_product_images(article_ids)
        history_urls = [images_data.get(str(art_id)) for art_id in article_ids]
        history_images = [
            card for i, (art_id, url) in enumerate(zip(article_ids, history_urls), 1) 
            if (card := create_history_card(url, i))
        ]

        # Получаем рекомендации
        recommendations = get_recommendations(user_id, exclude_items=history)
        recommendation_data = [
            {"item_id": str(pair["item_id"]), "score": pair["score"]} 
            for pair in recommendations
        ]
        
        # Получаем изображения для рекомендаций
        rec_images_data = get_product_images([rec["item_id"] for rec in recommendation_data])
        recommendations_images = [
            card for i, rec in enumerate(recommendation_data, 1)
            if (card := create_image_card(
                rec_images_data.get(rec["item_id"]),
                rec["item_id"],
                i
            ))
        ]

        return (
            history_images or [dbc.Alert("Нет данных для отображения", color="info")],
            recommendations_images or [dbc.Alert("Нет рекомендаций для отображения", color="info")],
            history_urls,
            [{"item_id": rec["item_id"], "image": rec_images_data.get(rec["item_id"])} 
             for rec in recommendation_data],
            history,
        )

    except Exception as e:
        print(f"Ошибка: {e}")
        return ([dbc.Alert(f"Ошибка: {str(e)}", color="danger")], [], [], [], [])


@app.callback(
    Output("dummy-output", "children"),
    [
        Input({"type": "like-btn", "index": dash.ALL}, "n_clicks"),
        Input({"type": "dislike-btn", "index": dash.ALL}, "n_clicks"),
    ],
    [State("user-id-input", "value")],
    prevent_initial_call=True,
)
def handle_feedback(like_clicks, dislike_clicks, user_id):
    ctx = dash.callback_context

    # Если ничего не было нажато или ID пользователя пустое
    if not ctx.triggered or not user_id:
        return dash.no_update

    # Проверяем, что значение n_clicks действительно изменилось
    triggered_prop = ctx.triggered[0]["prop_id"]
    if not triggered_prop.endswith(".n_clicks"):
        return dash.no_update

    # Извлекаем id кнопки, которая была нажата
    button_id = triggered_prop.split(".")[0]
    
    try:
        button_info = json.loads(button_id)
        button_type = button_info["type"]
        item_id = button_info["index"]
    except Exception as e:
        print(f"Ошибка при разборе ID кнопки: {e}")
        return dash.no_update

    if button_type == "like-btn":
        action = "like"
    elif button_type == "dislike-btn":
        action = "dislike"
    else:
        return dash.no_update

    print(f"Feedback received - User: {user_id}, Item: {item_id}, Action: {action}")
    
    # Отправляем фидбек в RabbitMQ
    send_feedback_to_rabbitmq(user_id, item_id, action)

    return dash.no_update


if __name__ == "__main__":
    app.run(
        host=APP_CONFIG["host"],
        port=APP_CONFIG["port"],
        debug=APP_CONFIG["debug"],
    )
