import os
import subprocess
import pandas as pd
import youtube_dl
import datetime

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
    # Determine the end point of the audio fragment by taking 29 seconds
    datetime_start = datetime.datetime.strptime(start, "%H:%M:%S") 
    datetime_end = datetime_start + datetime.timedelta(seconds=29)
    end = datetime_end.strftime("%H:%M:%S")

    if youtube_url:
        fn = '{}_{}_{}'.format(
            r['country'], r['song'], r['performer'])

        # Skip if file already exists
        fp = os.path.join(destination_dir, fn)
        if not os.path.exists(fp + '.mp3'):

            ydl_opts = {
                'outtmpl': fp + '.%(ext)s',
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'postprocessor_args': ['-ss', start, '-to', end]     # save audio only from the predetermined fragment
            }

            try:
                with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url])
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
    # Determine the end point of the audio fragment by taking 29 seconds
    datetime_start = datetime.datetime.strptime(start, "%H:%M:%S") 
    datetime_end = datetime_start + datetime.timedelta(seconds=29)
    end = datetime_end.strftime("%H:%M:%S")

    if youtube_url:
        fn = '{}_{}'.format(
            r['song'], r['performer'])

        # Skip if file already exists
        fp = os.path.join(destination_dir, fn)
        if not os.path.exists(fp + '.mp3'):
            
            ydl_opts = {
                'outtmpl': fp + '.%(ext)s',
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'postprocessor_args': ['-ss', start, '-to', end]     # save audio only from the predetermined fragment
            }

            try:
                with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url])
            except Exception as e:
                print(e)
                pass
        else:
            print('{} already exists'.format(fp))
        