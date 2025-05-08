import pika
import json
import redis


def main():
    # Подключение к Redis
    redis_client = redis.Redis(
        host='redis',
        port=6379,
        password='redis123',
        decode_responses=True
    )
    
    # Подключение к RabbitMQ
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host='rabbitmq',
            port=5672,
            credentials=pika.PlainCredentials('rabbit', 'rabbit123')
        )
    )
    channel = connection.channel()
    channel.queue_declare(queue='user_events')
    
    def callback(ch, method, properties, body):
        try:
            event = json.loads(body)
            # Обработка события и сохранение в Redis
            user_id = event.get('user_id')
            event_type = event.get('type')
            
            # Пример: сохраняем последние 100 событий для каждого пользователя
            redis_key = f"user_events:{user_id}"
            redis_client.lpush(redis_key, json.dumps(event))
            redis_client.ltrim(redis_key, 0, 99)
            
            # Можно также обновлять счетчики и другие агрегаты
            if event_type == 'item_view':
                redis_client.zincrby('user:item_views', 1, user_id)
            
        except Exception as e:
            print(f"Error processing event: {str(e)}")
    
    channel.basic_consume(
        queue='user_events',
        on_message_callback=callback,
        auto_ack=True
    )
    
    print(" [*] Waiting for messages. To exit press CTRL+C")
    channel.start_consuming()


if __name__ == '__main__':
    main()
