# RTS data parsing

## Installation
- Install poetry
- Run `poetry install`

# Meilisearch
```
docker run\
    -p 7700:7700 \
    -e MEILI_MASTER_KEY="1234"\
    -v $(pwd)/meili_data:/meili_data \
    getmeili/meilisearch:v0.30 \
    meilisearch --env="development"
```

# PyAv

Maybe compile ffmpeg with hardware decoder enable and bind to pyAv?

## Notes

If Poetry install issues with Keyring
`export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`

- First put VPN, map network drive
`\\emplussrv1.epfl.ch`

- To mount a network drive to linux, you can use this command:
`EMPLUS-DataSets/SinergiaFutureCinema/RTS/metadata/rts_metadata.db`

`sudo mount -t drvfs '\\KNAS\mjf' /mnt/mjf`
`sudo mount -t drvfs '\\emplussrv1.epfl.ch\EMPLUS-Network\RTS-Data' /mnt/rts`
