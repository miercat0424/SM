import os

from glob import glob

# Move to each folders
videos = glob("TDMS_Files\*.avi")

for video in videos :
    pathv = video
    video = video.replace("TDMS_Files\\","")
    vos = os.rename(f"{pathv}",f"AVI_Files/{video}")
    del vos

wavs = glob("TDMS_Files\*.wav")

for wav in wavs :
    pathw = wav
    wav = wav.replace("TDMS_Files\\","")
    wos = os.rename(f"{pathw}",f"WAV_Files/{wav}")
    del wos










# import moviepy.editor as mp 


# Path
# # converting WAV from AVI
# for video in videos : 
#     video = video.replace(".avi","")
#     with mp.VideoFileClip(f"{video}.avi") as my_clip :
#         my_clip.audio.write_audiofile(f"{video}.wav")
        