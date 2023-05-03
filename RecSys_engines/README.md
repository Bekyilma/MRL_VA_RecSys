# Multi-Modal Visual Art RecSys Services

Different Multi-modal RecSys engines will provide a series of recommendations based on the user preferences.

## Install

Python >= 3.5 and pip3 are required:
```sh
sudo apt install python3 python3-pip
```

Install Python dependencies:
```sh
pip3 install -r requirements.txt
```

Download and unzip services data:
```sh
wget https://project-banana.eu/va-recsys/vadata.zip

unzip -q vadata.zip
```

Download the trained painting_CLIP, unzip it and place the extracted ```clip``` folder in the ```/data``` folder: 
```sh
$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1PJ1kFGsg6U-FzJOOAOUm8Pq1rHEkxNh_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1PJ1kFGsg6U-FzJOOAOUm8Pq1rHEkxNh_" -O clip.zip && rm -rf /tmp/cookies.txt

$ unzip -q clip.zip
```
Download the trained painting_BLIP, unzip it and place the extracted ```blip``` folder in the ```/data``` folder: 
```sh
$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1pPQ16MRsab6b_Tvn4nMlcqdhw64-YjfD' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1pPQ16MRsab6b_Tvn4nMlcqdhw64-YjfD" -O blip.zip && rm -rf /tmp/cookies.txt

$ unzip -q blip.zip
```
## Running the services

```sh
bash start.sh
```

## Stopping the services

```sh
bash stop.sh
```

## Restarting the services

```sh
bash restart.sh
```

## Monitoring the services

```sh
bash status.sh
```


**Note:** You should run all services through a [WSGI application in production](https://flask.palletsprojects.com/en/2.0.x/deploying/fastcgi/), for better performance.
