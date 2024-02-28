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


@click.command()
@click.argument('queue')
def main(queue):
    message_length = 20
    queue_name = 'transcript'
    broker_url = f'amqp://guest:guest@{RABBITMQ_SERVER}:5672'

    print(f"Connecting to rabbitmq server at {broker_url}")

    df = emv.utils.dataframe_from_hdf5('/mnt/rts/archives/', 'rts_metadata')
    df['mediaFolderPath'] = df['mediaFolderPath'].apply(lambda x: x.split('/mnt/rts/')[-1])

    DataAccessObject().set_user_id(1)
    while True:
        query = text("SELECT * FROM entries_to_queue LIMIT 500")
        results = DataAccessObject().fetch_all(query)
        all_paths = [x['media_id'] for x in results]
        if len(all_paths) < 100:
            time.sleep(10)
            continue

        df_exists = df[df['mediaFolderPath'].isin(all_paths)]

        payloads = []
        no_messages = 100
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

        # Delete the rows from the table
        for media_id in media_ids_send:
            query = text("DELETE FROM entries_to_queue WHERE media_id = :media_id")
            DataAccessObject().execute_query(query, {"media_id": media_id})

        # sleep for 5 seconds
        time.sleep(5)
        

if __name__ == "__main__":
    main()
