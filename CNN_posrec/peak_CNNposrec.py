import strax, straxen
import numpy as np
from copy import deepcopy

class DoubleWidthConverter():
    """
    This is a small helping function that converts 
    standard row of PMT charge areas to 2D pattern
    needed for CNN models. 

    It uses doublewidth transformation to get from 
    HEX placing of PMTs to standard square binning
    
    """
    def PMTnumuber_doublewidth(self):
        """
        This function calculates the relation between 
        PMT channel number and position on 2D "image"
        """
        coords = {}
        row_nPMT = np.array([0, 6, 9, 12, 13, 14, 15, 
                             16, 17, 16, 17, 16, 17, 16, 
                             15, 14, 13, 12, 9, 6])
        row_nPMT_cumulative = np.cumsum(row_nPMT)
        tot_rows = len(row_nPMT)-2 #
        pairs = []
        for i in range(0, 253):
            n_row = np.argwhere( (i>=row_nPMT_cumulative[0:-1])*
                                 (i<row_nPMT_cumulative[1:])  
                               )[0][0]
            x_offset = ( 2*int(np.ceil( 
                           0.5* (row_nPMT.max()  - row_nPMT[n_row+1]))) 
                         - (row_nPMT[n_row+1]+1)%2)
            i_PMT = i - row_nPMT_cumulative[n_row]
            x_coord = x_offset + 2*i_PMT 
            y_coord = tot_rows - n_row
            coords[i] = [x_coord, y_coord]
            pairs.append( (x_coord, y_coord))
        return(coords,np.array(pairs) )
    
    def __init__(self):
        self.coords,self.pairs = self.PMTnumuber_doublewidth()
        self.size = (33,19)
        
    def get_coordinates(self):
        """
        this function returns dictionary of cooridnates
        if form dict(ch: [x,y]) 
        """
        return(deepcopy(self.coords))
    
    def get_size(self):
        """ 
        return sice of the 2D image 
        """
        return(deepcopy(self.size))
    
    def convert_pattern(self, inarr):
        """
        This function converts an array of PMT areas to 2D pattern
        """
        pattern = np.zeros(self.size)
        pattern[self.pairs[:,0],self.pairs[:,1] ] = np.array(inarr)
        return(pattern)
    
    def convert_multiple_patterns(self, inarr):
        """
        This function converts a 2D array of PMT arreas (for multiple events)
        """
        pattern = np.zeros((inarr.shape[0],self.size[0],self.size[1]))
        pattern[:,self.pairs[:,0],self.pairs[:,1] ] = np.array(inarr)
        return(pattern)

export, __all__ = strax.exporter()
@export
@strax.takes_config(
    strax.Option('min_reconstruction_area',
                 help='Skip reconstruction if area (PE) is less than this',
                 default=10),
    strax.Option('n_top_pmts', default=straxen.n_top_pmts,
                 help="Number of top PMTs"),
    strax.Option("s2_cnn_model_path", 
                 help="Path to the CNN model in hdf5 format. WARING, this should include the whole model file and not the weights file", 
                 default=("/project2/lgrandi/terliuk/CNNmodels/CNN_models_FRice/"+
                          "CNN_standard/disk_const_PE/CNN_dw_maxnorm_v2_3L_A_const_100__Z_const_-1.hdf5")
                )
)
class CNNS2PostionReconstruction(strax.Plugin):
    """
    This pluging provides S2 position reconstruction for top array
    
    returns variables : 
        - x_TFS2CNN - reconstructed X position in [ cm ]
        - y_TFS2CNN - reconstructed Y position in [ cm ]
        - patterns - 2D array of areas as used for CNN
    """
    dtype = [('x_TFS2CNN', np.float32,
              'Reconstructed CNN S2 X position [ cm ] '),
             ('y_TFS2CNN', np.float32,
              'Reconstructed CNN S2 Y position [ cm ] '), 
             (("Patterns after DW transformation normalized to max PMT area",
                 "patterns"), np.float, (33, 19,)), 
             ] + strax.time_fields
    depends_on = ('peaks',)
    parallel = False
    provides = "CNNS2PostionReconstruction"
    __version__ = '0.0.0'  
    
    def setup(self):
        import tensorflow as tf
        keras = tf.keras
        nn_model = keras.models.load_model(self.config['s2_cnn_model_path'])
        self.cnn_model = nn_model
        self.converter = DoubleWidthConverter()
        print("====== Loaded TF CNN model =====")
        self.cnn_model.summary()
        print("====== end of model summary =====")

    def compute(self, peaks):
        result = np.ones(len(peaks), dtype=self.dtype)
        result['time'], result['endtime'] = peaks['time'], strax.endtime(peaks)
        result['x_TFS2CNN'] *= float('nan')
        result['y_TFS2CNN'] *= float('nan')
        result['patterns'] *=np.nan

        peak_mask = peaks['area'] > self.config['min_reconstruction_area']
        if not np.sum(peak_mask):
            # Nothing to do, and .predict crashes on empty arrays
            return result        
        areas  = peaks['area_per_channel'][peak_mask,0:self.config['n_top_pmts']]
        patterns = self.converter.convert_multiple_patterns(areas)
        patterns = patterns/patterns.max(axis=(1,2))[:,None,None]  
        # renormalizing since CNNs are done normalized to PMT with the largest area   
        result['patterns'][peak_mask] = patterns
        reco_pos = self.cnn_model.predict(patterns)
        result['x_TFS2CNN'][peak_mask] = reco_pos[:, 0]/10.0
        result['y_TFS2CNN'][peak_mask] = reco_pos[:, 1]/10.0 # CNN is in mm, but here we use cm
        return result
