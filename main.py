from iluvatar import Iluvatar 
from iluvatar import MusicDownloader
import json
import os
import sys
from glob import glob
from dotenv import load_dotenv


def main():
    load_dotenv('config.env')
    OUTPUT_PATH = os.getenv('OUTPUT_PATH')
    YOUTUBE_TOKEN = os.getenv('YOUTUBE_TOKEN')
    md = MusicDownloader(YOUTUBE_TOKEN, OUTPUT_PATH)
    md.download_music(sys.argv[1], sys.argv[2])
    noise_meshes = []
    for music in glob(f"{OUTPUT_PATH}/*"):
        features = Iluvatar.extract_features(music)
        mesh_noise = Iluvatar.process_input(features)
        noise_meshes.append(Iluvatar.format_to_godot_noise(mesh_noise))
    with open('noise_meshes.json', 'w', encoding='utf-8') as f:
        json.dump(noise_meshes, f, ensure_ascii=False, indent=4)
    pass


if __name__ == '__main__':
    main()
    pass
    
