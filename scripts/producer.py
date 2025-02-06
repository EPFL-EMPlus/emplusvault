from emv.db.dao import DataAccessObject
from sqlalchemy.sql import text
from emv.pipelines.utils import send_message
from emv.pipelines.utils import filter_by_asset_type
import emv.utils
import os
import json
from emv.settings import RABBITMQ_SERVER
import click
import time
import pika
from itertools import islice
import logging

logging.getLogger('pika').setLevel(logging.WARNING)


# broker_url 'amqp://guest:guest@10.244.3.174:5672'


def count_messages(broker_url, queue_name):
    # Count how many messages are remaining in the queue
    broker_url = os.getenv(
        'BROKER_URL', broker_url)

    params = pika.URLParameters(broker_url)

    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    # Declare the queue to get details, set passive to True
    queue = channel.queue_declare(queue=queue_name, passive=True)
    message_count = queue.method.message_count

    connection.close()
    print("Messages remaining in the queue: ", message_count)


@click.command()
@click.argument('broker_url')
def main(broker_url):
    message_length = 20
    queue_name = 'transcript'

    print("Reading the dataframes..")
    df = emv.utils.dataframe_from_hdf5('/mnt/rts/archives/', 'rts_metadata')

    print("Reading entries to queue..")
    query = text(
        "SELECT * FROM entries_to_queue ORDER BY RANDOM() LIMIT 30000")
    results = DataAccessObject().fetch_all(query)

    all_paths = [x['media_id'].split("/")[-1] for x in results]
    df_exists = df[df.index.isin(all_paths)]

    df_exists.loc[:, 'mediaFolderPath'] = df_exists.loc[:, 'mediaFolderPath'].apply(
        lambda x: "/mnt/rts/archives/" + x.split('/mnt/rts/')[-1] if "archives" not in x else x)

    payloads = []
    no_messages = int(len(df_exists) / message_length)
    for i in range(0, no_messages * message_length, message_length):
        payloads.append(
            df_exists.iloc[i:i+message_length].to_dict('records')
        )

    media_ids_send = []
    for payload in payloads:
        if len(payload) == 0:
            continue

        data = {
            'job_type': 'transcript',
            'payload': payload
        }
        try:
            send_message(queue_name, data, broker_url=broker_url)
        except Exception as e:
            print(f"Error sending message: {e}")
            continue
        media_ids_send += [x['mediaFolderPath'] for x in payload]

    print(f"Sent {len(media_ids_send)} messages to the queue")
    count_messages(broker_url, queue_name)

    def chunks(iterable, size):
        iterator = iter(iterable)
        for first in iterator:
            yield [first] + list(islice(iterator, size - 1))

    # Chunk size, e.g., 1000 ids per batch
    chunk_size = 100

    for chunk in chunks(media_ids_send, chunk_size):
        chunk = [x[18:] for x in chunk]
        query = text(
            "DELETE FROM entries_to_queue WHERE media_id IN :media_ids")
        DataAccessObject().execute_query(query, {"media_ids": tuple(chunk)})
    print(f"Deleted {len(media_ids_send)} entries")

# @click.command()
# @click.argument('queue')
# def main(queue):
#     message_length = 20
#     queue_name = 'transcript'
#     broker_url = f'amqp://guest:guest@{RABBITMQ_SERVER}:5672'

#     print(f"Connecting to rabbitmq server at {broker_url}")

#     df = emv.utils.dataframe_from_hdf5('/mnt/rts/archives/', 'rts_metadata')
#     df['mediaFolderPath'] = df['mediaFolderPath'].apply(
#         lambda x: x.split('/mnt/rts/')[-1])

#     DataAccessObject().set_user_id(1)
#     while True:
#         query = text(
#             "SELECT * FROM entries_to_queue ORDER BY RANDOM() LIMIT 500")
#         results = DataAccessObject().fetch_all(query)
#         all_paths = [x['media_id'] for x in results]
#         if len(all_paths) < 100:
#             time.sleep(10)
#             continue

#         df_exists = df[df['mediaFolderPath'].isin(all_paths)]

#         payloads = []
#         no_messages = 100
#         for i in range(0, no_messages * message_length, message_length):
#             payloads.append(
#                 df_exists.iloc[i:i+message_length].to_dict('records')
#             )
#         media_ids_send = []
#         for payload in payloads:
#             if len(payload) == 0:
#                 continue

#             data = {
#                 'job_type': 'transcript',
#                 'payload': payload
#             }
#             try:
#                 send_message(queue_name, data, broker_url=broker_url)
#             except Exception as e:
#                 print(f"Error sending message: {e}")
#                 continue
#             media_ids_send += [x['mediaFolderPath'] for x in payload]

#         # Delete the rows from the table
#         for media_id in media_ids_send:
#             query = text(
#                 "DELETE FROM entries_to_queue WHERE media_id = :media_id")
#             DataAccessObject().execute_query(query, {"media_id": media_id[9:]})


#         # sleep for 5 seconds
#         time.sleep(5)
if __name__ == "__main__":
    main()
