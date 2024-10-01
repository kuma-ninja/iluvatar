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
    music_path = md.download_music(sys.argv[1], sys.argv[2])
    features = Iluvatar.extract_features(music_path)
    mesh_noise = Iluvatar.process_input(features)
    mesh_noise = music_feature = Iluvatar.format_to_godot_noise(mesh_noise)
    with open('noise_meshe.json', 'w', encoding='utf-8') as f:
        json.dump(mesh_noise, f, ensure_ascii=False, indent=4)
    pass


if __name__ == '__main__':
    main()
    pass
    
