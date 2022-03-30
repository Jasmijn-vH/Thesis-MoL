import os
import subprocess
import pandas as pd
import youtube_dl
import datetime
from pydub import AudioSegment
import ffmpeg

def get_sec(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

audio_dir = 'audio'
if not os.path.exists(audio_dir):
    os.makedirs(audio_dir)


# Eurovision Song Contest
contestantsESC = pd.read_csv('songsESC.csv')

for i, r in contestantsESC.iterrows():
    destination_dir = os.path.join(audio_dir, 'ESC', str(r['year']))
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Retrieve the youtube-link and starting point of the fragment from the dataframe
    youtube_url = r['youtube_url_studio']
    start = r['start']
    start_sec = get_sec(start)

    if youtube_url:
        fn = '{}_{}_{}_{}'.format(
            r['place_contest'], r['country'], r['song'], r['performer'])

        # Skip if file already exists
        fp = os.path.join(destination_dir, fn)
        if not os.path.exists(fp + '_.mp3'):

            ydl_opts = {
                'outtmpl': fp + '.%(ext)s',
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }]
            }

            try:
                with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url])

                    audio_input = ffmpeg.input(fp + '.mp3')
                    audio_cut = audio_input.audio.filter('atrim', start=start_sec, duration=29)
                    audio_output = ffmpeg.output(audio_cut, fp + '_.mp3')
                    ffmpeg.run(audio_output)
                    os.remove(fp + '.mp3')
            except Exception as e:
                print(e)
                pass
        else:
            print('{} already exists'.format(fp))
 
        
# Festival di Sanremo
contestantsSR = pd.read_csv('songsSanremo.csv')

for i, r in contestantsSR.iterrows():
    destination_dir = os.path.join(audio_dir, 'SanRemo', str(r['year']))
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Retrieve the youtube-link and starting point of the fragment from the dataframe
    youtube_url = r['youtube_url_studio']
    start = r['start']
    start_sec = get_sec(start)

    if youtube_url:
        fn = '{}_{}_{}'.format(
            r['place_contest'], r['song'], r['performer'])

        # Skip if file already exists
        fp = os.path.join(destination_dir, fn)
        if not os.path.exists(fp + '_.mp3'):
            
            ydl_opts = {
                'outtmpl': fp + '.%(ext)s',
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }]
            }

            try:
                with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url])

                    audio_input = ffmpeg.input(fp + '.mp3')
                    audio_cut = audio_input.audio.filter('atrim', start=start_sec, duration=29)
                    audio_output = ffmpeg.output(audio_cut, fp + '_.mp3')
                    ffmpeg.run(audio_output)
                    os.remove(fp + '.mp3')
            except Exception as e:
                print(e)
                pass
        else:
            print('{} already exists'.format(fp))
