import numpy as np 
from copy import deepcopy
class PatternConverter():
    
    def PMTnumuber_doublewidth(self):
        coords = {}
        row_nPMT = np.array([0, 6, 9, 12, 13, 14, 15, 16, 17, 16, 17, 16, 17, 16, 15, 14, 13, 12, 9, 6])
        row_nPMT_cumulative = np.cumsum(row_nPMT)
        tot_rows = len(row_nPMT)-2 #
        pairs = []
        for i in range(0, 253):
            n_row = np.argwhere((i>=row_nPMT_cumulative[0:-1])*(i<row_nPMT_cumulative[1:])  )[0][0]
            x_offset =  2*int(np.ceil( 0.5* (row_nPMT.max()  - row_nPMT[n_row+1]))) - (row_nPMT[n_row+1]+1)%2
            i_PMT = i - row_nPMT_cumulative[n_row]
            x_coord = x_offset + 2*i_PMT 
            y_coord = tot_rows - n_row
            coords[i] = [x_coord, y_coord]
            pairs.append( (x_coord, y_coord))
        return(coords,np.array(pairs) )
    def PMTnumuber_offset(self):
        coords = {}
        row_nPMT = np.array([0, 6, 9, 12, 13, 14, 15, 16, 17, 16, 17, 16, 17, 16, 15, 14, 13, 12, 9, 6])
        row_nPMT_cumulative = np.cumsum(row_nPMT)
        tot_rows = len(row_nPMT)-2 #
        pairs = []
        for i in range(0, 253):
            n_row = np.argwhere((i>=row_nPMT_cumulative[0:-1])*(i<row_nPMT_cumulative[1:])  )[0][0]
            x_offset =  int(np.ceil( 0.5* (row_nPMT.max()  - row_nPMT[n_row+1])))
            i_PMT = i - row_nPMT_cumulative[n_row]
            x_coord = x_offset + i_PMT 
            y_coord = tot_rows - n_row
            coords[i] = [x_coord, y_coord]
            pairs.append( (x_coord, y_coord))
        return(coords,np.array(pairs) )

    def __init__(self, model = "doublewidth"):
        if model=="doublewidth" or model == "dw":
            self.coords,self.pairs = self.PMTnumuber_doublewidth()
            self.size = (33,19)
        elif model=="offset":
            self.coords,self.pairs = self.PMTnumuber_offset()
            self.size = (17,19)
        else: 
            raise ValueError("Wrong transformation model")
    def get_coordinates(self):
        return(deepcopy(self.coords))
    def get_size(self):
        return(deepcopy(self.size))
    def convert_pattern(self, inarr):
        pattern = np.zeros(self.size)
        pattern[self.pairs[:,0],self.pairs[:,1] ] = np.array(inarr)
        return(pattern)

