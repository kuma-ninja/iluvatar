import yt_dlp as youtube_dl
import os
from googleapiclient.discovery import build


class MyLogger(object):
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        print(msg)
        pass

    def my_hook(d, enable=False):
        if enable:
            pass
        pass


class MusicDownloader:
    def __init__(self, YOUTUBE_TOKEN, OUTPUT_PATH):
        self.YOUTUBE_TOKEN = YOUTUBE_TOKEN
        self.OUTPUT_PATH = OUTPUT_PATH
        self.youtube = self.initialize()

    def initialize(self):
        try:
            return build('youtube', 'v3', developerKey=self.YOUTUBE_TOKEN)
        except FileNotFoundError:
            print("Failed to Load .env variables")
        return

    def search_music(self, artist, music_name):
        search_query = f"{artist} {music_name}"
        # try:
        request = self.youtube.search().list(
                q=search_query,
                part='id,snippet',
                maxResults=1,
                type='video'
            )
        response = request.execute()
        if response['items']:
            video_id = response['items'][0]['id']['videoId']
            return f"https://www.youtube.com.br/watch?v={video_id}"
    # except:
        #     print("Number of quotas exceed")
        return

    def download_music(self, artist, music_name):
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'logger': MyLogger(),
            # 'progress_hooks': [my_hook],
            'outtmpl': os.path.join(self.OUTPUT_PATH, f"{artist} - {music_name}")
        }
        music_url = self.search_music(artist, music_name)
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([music_url])
        return os.path.join(self.OUTPUT_PATH, f"{artist} - {music_name}.mp3")

