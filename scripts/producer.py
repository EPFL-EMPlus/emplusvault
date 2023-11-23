import pika
import os
import json

def send_message(queue, data):
    broker_url = os.getenv('BROKER_URL', 'amqp://guest:guest@rabbitmq-service:5672')
    params = pika.URLParameters(broker_url)

    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    channel.queue_declare(queue=queue, durable=True)

    channel.basic_publish(
        exchange='',
        routing_key=queue,
        body=json.dumps(data),
        properties=pika.BasicProperties(
            delivery_mode=2,
        ))

    print(f"Sent: {len(data)}")

    connection.close()

if __name__ == "__main__":
    # Retrieving videos that don't have any poses from the db
    pass

    data = {
        'job_type': 'pose',
        'payload': [
        ]
    }

    queue_name = 'pose'
    send_message(queue_name, data)
