from __future__ import print_function # Python 2/3 compatibility
import boto3
import json
import decimal

dynamodb = boto3.resource('dynamodb', region_name='us-east-1', endpoint_url="http://localhost:8000")

table = dynamodb.Table('closedcaption')

with open("data/cc_chunks.json") as json_file:
    shows = json.load(json_file) # parse_float = decimal.Decimal)
    for show in shows:
        _id = show['_id']
        channel = show['channel']
        created_at = show['created_at']
        duration = show['duration']
        lang = show['lang']
        machine_id = show['machine_id']
        ts = show['ts']
        tui = int(show['tui'])
        tv = show['tv']
        url = show['url']
        zip_url = show['zip_url']

        print("Adding show:", _id, channel)

        table.put_item(
            Item={
                '_id': _id,
                'channel': channel,
                'created_at': created_at,
                'duration': duration,
                'lang': lang,
                'machine_id': machine_id,
                'ts': ts,
                'tui': tui,
                'tv': tv,
                'url': url,
                'zip_url': zip_url
            }
        )
