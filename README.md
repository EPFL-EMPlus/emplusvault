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

### Front-end
The frontend code is made with [Cables.gl](htttps://cables.gl), a node-based visual framework on the Web. To be able to pull and serve the visualization a few extra installation steps are required.

1. Install [NVM](https://github.com/nvm-sh/nvm)
2. Install Node LTS version:  ```nvm install --lts```
3. Install Cables.gl CLI:  ```npm install -g @cables/cables```
4. Create an specific API-KEY on [Cables.gl](https://cables.gl/settings)
5. Create `~/.cablesrc` file with `apikey=YOUR_API_KEY`.


# Copy data from RTS machine to public demo
```
rsync -am /media/data/rts/metadata/ root@128.178.218.107:/media/data/rts/metadata/
```