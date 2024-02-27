import pika
import os
import click
import json
import pandas as pd
from emv.settings import RABBITMQ_SERVER


def process_message(message):
    print(f"Processing message")
    data = json.loads(message)
    job_type = data['job_type']
    df = pd.DataFrame(data['payload'])

    if job_type == "pose":
        from emv.pipelines.ioc import PipelineIOC
        pipeline = PipelineIOC()
        pipeline.process_poses(df)
    elif job_type == "transcript":
        from emv.pipelines.rts import PipelineRTS
        pipeline = PipelineRTS()
        pipeline.ingest(df)

def callback(ch, method, properties, body):
    ch.basic_ack(delivery_tag=method.delivery_tag)
    # Stop consuming now that the message was loaded
    ch.stop_consuming()
    ch.close()
    process_message(body)


@click.command()
@click.argument('queue')
def main(queue):
    try:
        broker_url = os.getenv(
            'BROKER_URL', f'amqp://guest:guest@{RABBITMQ_SERVER}:5672')
        params = pika.URLParameters(broker_url)

        connection = pika.BlockingConnection(params)
        channel = connection.channel()

        channel.queue_declare(queue=queue, durable=True)
        channel.basic_consume(queue=queue, on_message_callback=callback)
        channel.start_consuming()

        connection.close()
    except pika.exceptions.AMQPConnectionError as err:
        print(f"Connection error: {err}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if connection and not connection.is_closed:
            connection.close()

if __name__ == "__main__":
    main()
