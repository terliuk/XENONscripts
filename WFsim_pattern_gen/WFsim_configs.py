import numpy as np
import wfsim
import strax 

class TrainingPatternS2WFsim(strax.LoopPlugin):
    depends_on = ('peaks','peak_basics','truth')
    provides = 'training_patterns_s2'
    __version__ = '0.0.1'

    dtype = [('time',np.int),    
             ('endtime',np.int),
             ('x',np.float),
             ('y',np.float),
             ('z',np.float),
             ('npeaks', np.int), 
             (("Integrated signal per PMT [ PE ]","area_per_channel"), np.float, (494,)), 
             ("amp", np.float), 
             ("n_electron", np.float),
             ("area", np.float)
            ]
    def compute(self, peaks, truth):
        #print("Types : " , truth['type'])
        true_S2s = truth[truth['type']==2]
        result = np.zeros(len(true_S2s), dtype = self.dtype)
        for i, t in enumerate(true_S2s):
            result['time'][i] = t['time']
            result['endtime'][i] = t['endtime']
            result['x'][i] = t['x']
            result['y'][i] = t['y']
            result['z'][i] = t['z']
            result['amp'][i] = t['amp']
            result['n_electron'][i] = t['n_electron']
            p = peaks[((peaks['time'] < t['t_mean_photon'])&
                       (peaks['endtime'] > t['t_mean_photon']))]
            result['npeaks'][i] = len(p)
            if len(p)==1:
                result['area_per_channel'][i] = p["area_per_channel"]
                result['area'][i] = p['area']
            elif len(p)>1:
                result['area_per_channel'][i] = p["area_per_channel"].sum(axis=0)
                result['area'][i]  = p["area"].sum(axis=0)
        return(result)

class WFsim_config: 
    """
    This is a simple function that creates configuration string
    and turns it into WFsim generator
    :params 
    - config - name of the configuration in format <A_configuration__Z_configuration>
    - PRNG - random generator, standard NumPy will be used if no generator provided 
    """
    def __init__(self, config="", PRNG = None):
        if PRNG==None:
            print("INFO: no RNG is provided, using standard NumPy")
            self.randgen = np.random
        else: 
            self.randgen = PRNG
        self.Nevents = -1
        self.confname = config
        if config == "":
            print("INFO: no config provided! Use SetAconf() and SetZconf() to set config")
            self.confname = "__"
        else:   
            confA,confZ = self.confname.split("__")
            print("Found config for amplitude : ",confA) 
            print("Found confog for Z : ", confZ)
            self.SetAconf(confA)
            self.SetZconf(confZ)
        
    def SetAconf(self, confA):
        confA_split = confA.split("_")
        self.typeA = confA_split[1]
        if self.typeA == "const": 
            assert len(confA_split)==3, "wrong format for constant amplitude, example A_const_1000"
            self.amp=float(confA_split[2])
            print ("Will generate amplitude : ", self.amp, " electrons")
            self.GenerateA = self.GenerateConstA
        elif self.typeA == "lin":
            assert len(confA_split)==4, "wrong format, use \"A_lin_min_max\", for example A_lin_100_1000"
            self.minA = float(confA_split[2])
            self.maxA = float(confA_split[3])
            assert self.maxA >= self.minA, "maximal value should be larger than minimal"
            print ("Will generate amplitude in range between", self.minA, "and", self.maxA)
            self.GenerateA = self.GenerateLinA
        elif self.typeA == "log":
            assert len(confA_split)==4, "wrong format, use \"A_log_min_max\", for example A_log_100_1000"
            self.minA = float(confA_split[2])
            self.maxA = float(confA_split[3])
            assert self.maxA >= self.minA, "maximal value should be larger than minimal"
            print ("Will generate amplitude logarithmically in range between", self.minA, "and", self.maxA)
            self.GenerateA = self.GenerateLogA
        else: 
            raise ValueError("Unknown configuration for amplitude generation provided : "+confA)
    def SetZconf(self, confZ):
        
        confZ_split = confZ.split("_")
        self.typeZ = confZ_split[1]
        if self.typeZ == "const": 
            assert len(confZ_split)==3, "wrong format for constant Z, example Z_const_-5.0"
            self.zvalue=float(confZ_split[2])
            self.GenerateZ = self.GenerateConstZ
            print ("Will generate Z = ", self.zvalue, " cm")
        elif self.typeZ == "lin":
            assert len(confZ_split)==4, "wrong format for Z range, use Z_lin_min_max"
            self.minZ = float(confZ_split[2])
            self.maxZ = float(confZ_split[3])
            assert self.maxZ >= self.minZ, "maximal value should be larger than minimal!"
            print ("Will generate Z in range between", self.minZ, "and", self.maxZ)
            self.GenerateZ = self.GenerateLinZ
        elif self.typeZ == "full":
            assert len(confZ_split)==2, "too many arguments for full config "
            print ("Will generate Z in full range")
            self.GenerateZ = self.GenerateFullZ
        else:
            raise ValueError("Unknown configuration for Z generation provided : "+confZ)
    def SetNumberOfEvents(self, nevt):
        self.Nevents = nevt
        print("Set number of events manually to : ", self.Nevents )
    def SetManualInstruction(self, instruction):
        self.instruction =  instruction
    def GetManualInstruction(self,c, verbose = False ):
        print("==== Instruction ========\n", self.instruction,"\n=============")
        return(self.instruction)
    def GenerateConstA(self, size, c=None):
        return(self.amp*np.ones(size))
    def GenerateLinA(self, size, c=None):
        return(self.randgen.uniform(self.minA, self.maxA,size=size))
    def GenerateLogA(self, size, c=None):
        return(10.**self.randgen.uniform(np.log10(self.minA), np.log10(self.maxA), size=size) )
    def GenerateConstZ(self, size, c=None):
        return(self.zvalue*np.ones(size))  
    def GenerateLinZ(self, size, c=None):
        return(self.randgen.uniform(self.minZ, self.maxZ,size=size))
    def GenerateFullZ(self, size, c=None):
        return(self.randgen.uniform(- c['tpc_length'], -0.0,size=size))    
    def GenerateS2(self, c):
        if self.Nevents < 0:
            n = c['nevents'] = c['event_rate'] * c['chunk_size'] * c['nchunk']
            c['total_time'] = c['chunk_size'] * c['nchunk']    
        else: 
            n = self.Nevents
            c['total_time'] = c['chunk_size'] * c['nchunk']    
        #print("Generating" , n)    
        #print("TPC radius: ",c['tpc_radius'])
        instructions = np.zeros(n, dtype=wfsim.instruction_dtype)
        instructions['event_number'] = np.arange(1,n+1)
        instructions['time'] = 1e9*np.arange(0,n, dtype= float) +1e6
        r = np.sqrt(self.randgen.uniform(0., (c['tpc_radius'])**2.0, n) ) ## not going all the way up to the edge
        phi = self.randgen.uniform(-np.pi, np.pi,n)
        instructions['x'] = r * np.cos(phi) 
        instructions['y'] = r * np.sin(phi) 
        instructions['z'] = self.GenerateZ(n,c) 
        instructions['type'] = np.repeat([2], n)
        instructions['amp'] = self.GenerateA(n,c)
        instructions['recoil'] = np.repeat(np.array(["er"], dtype='<U2'), n)
        #print(instructions)
        #print( np.column_stack ( 
        #                ( np.arange(1,len(instructions['x'])+1), 
        #                  np.sqrt(instructions['x']**2 + instructions['y']**2 ))  ) 
        #    )
        return(instructions)
