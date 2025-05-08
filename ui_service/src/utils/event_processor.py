import pika
import json

def send_event_to_rabbitmq(event_data):
    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host='rabbitmq',
                port=5672,
                credentials=pika.PlainCredentials('rabbit', 'rabbit123')
            )
        )
        channel = connection.channel()
        channel.queue_declare(queue='user_events')
        
        channel.basic_publish(
            exchange='',
            routing_key='user_events',
            body=json.dumps(event_data)
        )
        
        connection.close()
    except Exception as e:
        print(f"Failed to send event to RabbitMQ: {str(e)}")
