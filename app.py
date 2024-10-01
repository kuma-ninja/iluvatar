from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from elasticsearch import Elasticsearch, exceptions as es_exceptions
from iluvatar import Iluvatar, MusicDownloader
import os
from dotenv import load_dotenv

INDEX = 'musics'
load_dotenv('config.env')
OUTPUT_PATH = os.getenv('OUTPUT_PATH')
YOUTUBE_TOKEN = os.getenv('YOUTUBE_TOKEN')
md = MusicDownloader(YOUTUBE_TOKEN, OUTPUT_PATH)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

es = Elasticsearch("https://localhost:9200/", http_auth=('elastic', '123456'),
                   ca_certs="ca/ca.crt", client_cert="ca/ca.crt", 
                   client_key="ca/ca.key", verify_certs=True)

# create index if doesnt exists
es.indices.create(index=INDEX, ignore=400)

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )

@app.get("/")
async def read_root():
    return {"message": "Iluvatar is Up :)"}

@app.get("/musics/{music_id}")
async def read_music(music_id: int):
    try:
        response = es.get(index=INDEX, id=music_id)
        return response["_source"]
    except es_exceptions.NotFoundError:
        raise HTTPException(status_code=404, detail="music not found")

@app.post("/musics/")
async def create_music(music: dict):
    if "artist" not in music:
        raise HTTPException(status_code=400, detail="Bad Request: 'artist' is required.")
    try:
        list_musics = await read_music(music['music'], music['artist'])
        if(len(list_musics['hits']['hits']) > 0 ):
            return {"music_id": list_musics['hits']['hits'][0]["_id"], "music": list_musics['hits']['hits'][0]['_source']}
        music_path = md.download_music(music['artist'], music['music'])
        features = Iluvatar.extract_features(music_path)
        mesh_noise = Iluvatar.process_input(features)
        music['simple_noise'] = music_feature = Iluvatar.format_to_godot_noise(mesh_noise)
        music['in_use'] = False
        response = es.index(index=INDEX, body=music)
        return {"music_id": response["_id"], "music": music}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/musics/")
async def read_music(music: str, artist: str):
    response = es.search(index=INDEX, body={
        "query": {
            "dis_max": {
                "queries": [
                    {"match": {"music": music}},
                    {"match": {"artist": artist}}
                    ]}
                }
        })
    return response

@app.get("/secure-data")
async def read_secure_data(token: str = Depends(oauth2_scheme)):
    if token != "valid-token":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return

