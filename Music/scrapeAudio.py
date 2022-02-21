import os
import subprocess
import pandas as pd
import youtube_dl

audio_dir = 'audio'
if not os.path.exists(audio_dir):
    os.makedirs(audio_dir)

# Eurovision Song Contest
contestantsESC = pd.read_csv('songsESC.csv')
print(contestantsESC)
for i, r in contestantsESC.iterrows():
    destination_dir = os.path.join(audio_dir, 'ESC', str(r['year']))
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    youtube_url = r['youtube_url_studio']
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
                'postprocessor_args': ['-ss', '00:01:00', '-to', '00:01:29']     # take audio from 1:00 to 1:29 minutes
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
print(contestantsSR)
for i, r in contestantsSR.iterrows():
    destination_dir = os.path.join(audio_dir, 'SanRemo', str(r['year']))
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    youtube_url = r['youtube_url_studio']
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
                'postprocessor_args': ['-ss', '00:01:00', '-to', '00:01:29']     # take audio from 1:00 to 1:30 minutes
            }

            try:
                with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url])
            except Exception as e:
                print(e)
                pass
        else:
            print('{} already exists'.format(fp))
        