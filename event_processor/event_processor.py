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
    channel.queue_declare(queue='user_feedback', durable=True)
    
    def feedback_callback(ch, method, properties, body):
        try:
            feedback = json.loads(body)
            user_id = feedback.get('user_id')
            item_id = feedback.get('item_id')
            action = feedback.get('action')
            
            if not all([user_id, item_id, action]):
                return
            
            # Для дизлайков сохраняем отдельно
            if action == 'dislike':
                redis_key = f"user_dislikes:{user_id}"
                redis_client.sadd(redis_key, item_id)
                # Устанавливаем срок хранения дизлайков (например, 30 дней)
                redis_client.expire(redis_key, 30 * 24 * 3600)
            
            # Для лайков добавляем в историю как просмотр
            elif action == 'like':
                event = {
                    'user_id': user_id,
                    'item_id': item_id,
                    'type': 'item_like',
                    'timestamp': feedback.get('timestamp'),
                }
                redis_key = f"user_events:{user_id}"
                redis_client.lpush(redis_key, json.dumps(event))
                redis_client.ltrim(redis_key, 0, 99)
                
        except Exception as e:
            print(f"Error processing feedback: {str(e)}")
    
    channel.basic_consume(
        queue='user_feedback',
        on_message_callback=feedback_callback,
        auto_ack=True,
    )
    
    print(" [*] Waiting for feedback messages...")
    channel.start_consuming()


if __name__ == '__main__':
    main()
