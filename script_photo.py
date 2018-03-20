from foraging import *
import os

data_folder = 'D:\\Dropbox\\Work\\Behavioural Tasks\\foraging\\data\\2017-10-27-photometry_select'

exp_photo = di.Experiment(data_folder)

pdata_folder = os.path.join(data_folder, 'photometry data')

pdata_filepath = os.path.join(pdata_folder, 'P3_VTA-2017-12-06-100408.tdms')

session = exp_photo.get_sessions(when='2017-12-06')[0]

# Filename maps:  (log file, photometry file)

filename_map_VTA =[
    ('p3-2017-12-01-150229.txt', 'p3_VTA-2017-12-01-150147.tdms'),
    ('p3-2017-12-02-140416.txt', 'p3_VTA-2017-12-02-141839.tdms'),
    ('p3-2017-12-04-160250.txt', 'p3_VTA-2017-12-04-161859.tdms'),
    ('p3-2017-12-05-105302.txt', 'p3_VTA-2017-12-05-110532.tdms'),
    ('p3-2017-12-05-170808.txt', 'p3_VTA-2017-12-05-171234.tdms'),
    ('p3-2017-12-06-094709.txt', 'P3_VTA-2017-12-06-100408.tdms'),
    ('p3-2017-12-06-171230.txt', 'P3_VTA-2017-12-06-171254.tdms'),
    ('p3-2017-12-07-110824.txt', 'P3_VTA-2017-12-07-112400.tdms'),
    ('p3-2017-12-07-172009.txt', 'P3_VTA-2017-12-07-172548.tdms'),
    ('p3-2017-12-08-093123.txt', 'P3_VTA-2017-12-08-104209.tdms'), 
    ('p3-2017-12-08-161640.txt', 'p3_VTA-2017-12-08-161930.tdms'),
    ('p5_NAc-2017-11-14-161523.txt', 'P5_NAc-2017-11-14-161505.tdms'),
    ('p5_NAc-2017-11-23-141332.txt', 'p5_NAc-2017-11-23-141451.tdms')]

