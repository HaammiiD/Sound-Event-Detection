

import os
import struct
import librosa
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 

    
Data_path = '/Users/hamid/Box Sync/GitHub/Sound Event Detection/UrbanSound Dataset sample/audio/'

metadata = pd.read_csv('/Users/hamid/Box Sync/GitHub/Sound Event Detection/UrbanSound Dataset sample/metadata/UrbanSound8K.csv')

metadata.head()

audiodata = []
for index, row in metadata.iterrows():
    
    file_name = os.path.join(os.path.abspath(Data_path),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    wave_file = open(file_name,"rb")
        
    riff = wave_file.read(12)
    fmt = wave_file.read(36)
        
    num_channels_string = fmt[10:12]
    num_channels = struct.unpack('<H', num_channels_string)[0]

    sample_rate_string = fmt[12:16]
    sample_rate = struct.unpack("<I",sample_rate_string)[0]
        
    bit_depth_string = fmt[22:24]
    bit_depth = struct.unpack("<H",bit_depth_string)[0]
    data = (num_channels, sample_rate, bit_depth)
    audiodata.append(data)

# Convert into a Panda dataframe
audiodf = pd.DataFrame(audiodata, columns=['num_channels','sample_rate','bit_depth'])

