from nptdms import TdmsFile, channel_data
from time import sleep

with TdmsFile.open("TDMS_Files/SeeSV-S206_20210914_175437.tdms") as tdms_file:
    # Use tdms_file
    files = tdms_file.groups()
    # for i in files:
    #     print(f"----------------------{i}--------------------")
    files1 = files[2].channels()
    files2 = files1[:]
    print(files2)
        # print()
        # for j in i.channels():
        #     print(f"----------------------{j}--------------------")
        #     # for k in j[:]:
        #         # print(k)
        #         # sleep(10)
        # print("============================================================================================================================")
    # group = tdms_file["RawData"]
    # print(group.channels())
# tdms_file = TdmsFile.read("TDMS_Files/SeeSV-S206_20210914_175437.tdms")
# print(tdms_file)
# print(channel_data)
# group     = tdms_file["group name"]
# channel   = group["channel name"]
# channel_data = channel[:]
# channel_properties = channel.properties
