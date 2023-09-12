# EMPLUS Vault

Data Vault for digital archives. 

## Installation

There are two versions of installation:
- Just the packages for using the cli
- Backend infrastructure and service

### Install Tools for the cli
This requires the installation of poetry as a first step (https://python-poetry.org/docs/). 

- Install poetry
- Run `poetry install`
- Run `poe force-torch`

### Run infrastructure locally

The settings in the .env file should be changed before the first docker compose up command.
There are two .env files. One for the docker containers and one for the cli and web server. 

- Docker installation for minio and postgres
```
cd docker

# Copy the fake env vars
cp .env.example .env

# Edit the environment file and fill in all the fields with credentials
vi .env

# Start
docker compose up
```

After the docker images are running, the database needs to be instantiated and the setup for the cli can be done:
```
# Go back to the main directory
cd ..

# Copy the CLI environment file (make sure that you are now not in the docker folder anymore)
cp .env.example .env

# Edit the environment file and fill in all the fields with credentials
vi .env

# Initialze the database
poe init-db

```

### Run the server

```
cd rts/api/
uvicorn server:app --host 0.0.0.0 --port 8763 --reload
```

#### Create a new database user

Create a new database user to use row level security (RLS). This can't be the default superuser as it has privileges to bypass RLS.


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

# Copy data from RTS to transfer drive
```
rsync -am /media/data/rts/metadata/ /mnt/transfer/rts/metadata/
rsync -am /media/data/rts/metadata/ /mnt/transfer/rts/metadata/
```

# Copy data from RTS machine to public demo
```
rsync -am /media/data/rts/metadata/ root@128.178.218.107:/media/data/rts/metadata/


```

# Tests

Running the tests needs the pgvector docker image to be running. To start it, run the following command:
```
docker compose up 
```

## Run tests
```
pytest tests
```
