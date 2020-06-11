#!/usr/bin/env python3
from optparse import OptionParser
from os.path import expandvars
import os, sys, random
import numpy as np
import strax, straxen, wfsim
from time import time

sys.path.append(os.path.dirname(__file__))
from WFsim_configs import WFsim_config, TrainingPatternS2WFsim

usage = "usage: %prog [options] inputfile"
parser = OptionParser(usage)
parser.add_option("-c", "--config", default=None, 
                dest= "CONFIG", help = "Name of the configuration/run") 
parser.add_option("-o", "--outfile", default = "test.hdf5", 
                dest = "OUTFILE", help = "Name of the output file")
parser.add_option("-r", "--runid", default = "1", type=int,
                 dest = "RUNID", help = "Number of the run, should be integer")
parser.add_option("-f", "--filenr", default = None, type=int,
                 dest = "FILENR", help = "Number of the current file, should be integer")
parser.add_option("-n", "--numevents", default = 1000, type=int,
                 dest = "NUMEVENTS", help = "Number of events/patterns to generate")
(options,args) = parser.parse_args()



print("Using configuration : ", options.CONFIG)
print("Using run id : " , options.RUNID)
print("File number : " , options.FILENR)
assert options.FILENR !=None , "File number in -f must be selected"
assert options.FILENR >=0, "File number is smaller than 0!"
seed = 21692380183192084+int(options.RUNID)*10000+int(options.FILENR) 
run_id=str(options.RUNID)
print("Events to generate : ", options.NUMEVENTS)
print("Seed for RNG : ", seed)
print("OUTFILE : ", options.OUTFILE)
cur_RG =  np.random.Generator(np.random.PCG64(seed))
print("Testing RNG : ", cur_RG.uniform(size=10))

gen = WFsim_config(options.CONFIG,cur_RG)

config = dict(detector='XENONnT',
                    **straxen.contexts.xnt_common_config,
                    to_pe_file= 'https://raw.githubusercontent.com/XENONnT/'
                                   'strax_auxiliary_files/master/fax_files/to_pe_nt.npy',
                    fax_config='https://raw.githubusercontent.com/XENONnT/'
                               'strax_auxiliary_files/master/fax_files/fax_config_nt.json',
                    
                    )
config.update(dict(gain_model=('to_pe_constant', 0.00612255) ))



print("===== Config =====")
print(config)
print("==================")
st = strax.Context(
        storage=strax.DataDirectory('./strax_data'),
        register=wfsim.RawRecordsFromFax,
        config=config,
        timeout= 3600,
        **straxen.contexts.common_opts)

config = dict(nchunk=1, event_rate=1, chunk_size=200)#options.NUMEVENTS,)
st.set_config(config)
st.set_config(dict(fax_file=None))

gen.SetNumberOfEvents(options.NUMEVENTS)
wfsim.strax_interface.rand_instructions = gen.GenerateS2

start = time()
truth = st.get_array(run_id, 'truth')

#print("=======   Truth   ==========\n", truth)
#truth_df = st.get_df(run_id, 'truth')
raw_records = st.get_array(run_id,'raw_records')
#records = st.get_array(run_id,'records')
#peaks = st.get_array(run_id, "peaks")
#peak_basics = st.get_array(run_id,'peak_basics')

st.register(TrainingPatternS2WFsim)
training_patterns_s2 = st.get_array(run_id,'training_patterns_s2')
stop = time()
# Now transforming the data
raw_pos = np.zeros( (len(training_patterns_s2['x']),3) )
raw_pos[:,0] = training_patterns_s2['x']
raw_pos[:,1] = training_patterns_s2['y'] 
raw_pos[:,2] = training_patterns_s2['z']

corrected_pos = np.zeros( (len(training_patterns_s2['x']),3) )
angle = 210*np.pi/180
corrected_pos[:,0] =  training_patterns_s2['x']*np.cos(angle) + training_patterns_s2['y']*np.sin(angle) 
corrected_pos[:,1] = -training_patterns_s2['x']*np.sin(angle) + training_patterns_s2['y']*np.cos(angle) 
corrected_pos[:,2] = training_patterns_s2['z']

area_per_channel = training_patterns_s2['area_per_channel']
n_electron = training_patterns_s2['n_electron']
amplitude = training_patterns_s2['amp']

print("Total pattern to save: ", len(training_patterns_s2['x']))
#


#print("Raw pos", raw_pos)
import h5py
outfile = h5py.File(options.OUTFILE, "w")
outfile.create_dataset("raw_pos", data = raw_pos, compression="gzip")
outfile.create_dataset("corr_pos", data = corrected_pos, compression="gzip")
outfile.create_dataset("area_per_channel", data = area_per_channel, compression="gzip")
outfile.create_dataset("n_electron", data = n_electron, compression="gzip")
outfile.create_dataset("amplitude", data = amplitude, compression="gzip")
conf_to_save = str(st.config)

outfile.create_dataset("strax_config", data=conf_to_save,dtype=h5py.string_dtype())

outfile.close()

print("Total time to generate: ",stop-start, " s")
