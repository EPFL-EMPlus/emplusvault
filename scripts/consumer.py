import pika
import os
import click
import json
import pandas as pd


def callback(ch, method, properties, body):
    process_message(body)
    ch.basic_ack(delivery_tag=method.delivery_tag)
    # Stop consuming after processing the message
    ch.stop_consuming()


def process_message(message):
    print(f"Processing message: {message}")
    data = json.loads(message)
    job_type = data['job_type']
    if job_type == "pose":
        from emv.pipelines.ioc import PipelineIOC

        df = pd.DataFrame(data['payload'])
        pipeline = PipelineIOC()
        pipeline.process_poses(df)


@click.command()
@click.argument('queue')
def main(queue):
    broker_url = os.getenv(
        'BROKER_URL', 'amqp://guest:guest@rabbitmq-service:5672')
    params = pika.URLParameters(broker_url)

    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    channel.queue_declare(queue=queue, durable=True)
    channel.basic_consume(queue=queue, on_message_callback=callback)
    channel.start_consuming()

    connection.close()


if __name__ == "__main__":
    main()
