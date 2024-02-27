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

    df = emv.utils.dataframe_from_hdf5('/mnt/rts/archives/', 'rts_metadata')
    df['mediaFolderPath'] = df['mediaFolderPath'].apply(lambda x: x.split('/mnt/rts/')[-1])

    DataAccessObject().set_user_id(1)
    while True:
        query = text("SELECT * FROM entries_to_queue LIMIT 100")
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

        for payload in payloads:
            if len(payload) == 0:
                continue
            
            data = {
                'job_type': 'transcript',
                'payload': payload
            }
            send_message(queue_name, data, broker_url='amqp://guest:guest@rabbitmq-service:5672')

        # Delete the rows from the table
        for media_id in all_paths:
            query = text("DELETE FROM entries_to_queue WHERE media_id = :media_id")
            DataAccessObject().execute_query(query, {"media_id": media_id})

        # sleep for 10 seconds
        time.sleep(10)
        

if __name__ == "__main__":
    main()
