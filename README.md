
# Iluvatar

With each note, Eru crafts worlds, blending music and magic into stunning realities.


## Installation

Install requirements using conda or pip

```bash
  pip install -r requirements.txt
```
  or
```bash
  conda install --file requirements.txt
```

### Configure .env file and run cli
Running with bash(output will be on mesh_noise.json):
```bash
  python main.py {artist} {music-name}
```
## API Reference
```bash
  uvicorn app:app --port 8080
```

#### Get music

```http
  GET /musics/${id}
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `id`      | `string` | **Required**. Music id |


#### Post new music to noise

```http
  POST /musics/
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `artist`      | `string` | **Required**. Artist Name |
| `music`      | `string` | **Required**. Music Music Name |

