import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as colors

def plot_pattern(pmthits , 
                   pmtpositions,
                 cmap = "YlGn", 
                 cbar_lim = "global",
                 tpc_radius = 66.4,
                 pmt_radius = 3.875, 
                 scale = colors.Normalize):
    
    fig = plt.figure(figsize=(24, 12))
    ###
    pmtpos = pmtpositions#straxen.pmt_positions()
    pmtpos_top = pmtpos[pmtpos['array']=="top"]
    PMTs_top = pmtpos_top.shape[0]
    pmtpos_bottom = pmtpos[pmtpos['array']=="bottom"]
    PMTs_bottom = pmtpos_bottom.shape[0] 
    ###
    if cbar_lim == "global":
        vmin_top = pmthits[0:PMTs_top+PMTs_bottom].min()
        vmax_top = pmthits[0:PMTs_top+PMTs_bottom].max() 
        vmin_bottom = vmin_top
        vmax_bottom = vmax_top
    elif cbar_lim == "relative":
        vmin_top = pmthits[0:PMTs_top].min()
        vmax_top = pmthits[0:PMTs_top].max() 
        vmin_bottom = pmthits[PMTs_top:PMTs_top+PMTs_bottom].min()
        vmax_bottom = pmthits[PMTs_top:PMTs_top+PMTs_bottom].max() 
    elif len(cbar_lim)==4:
        vmin_top = cbar_lim[0]
        vmax_top = cbar_lim[1]
        vmin_bottom = cbar_lim[2]
        vmax_bottom = cbar_lim[3]
    elif len(cbar_lim)==2:
        vmin_top = cbar_lim[0]
        vmax_top = cbar_lim[1]
        vmin_bottom = cbar_lim[0]
        vmax_bottom = cbar_lim[1]        
    ###
    ax_top = fig.add_axes([0.04, 0.10, 0.40, 0.80])
    ax_top.set_title("Top array", fontsize=12)
    ax_top.add_patch(plt.Circle( (0., 0.), tpc_radius, facecolor="none", edgecolor="black" ) )
    circles_top = []
    for pmt in pmtpos_top["i"]:
        circles_top.append(
            plt.Circle( (pmtpos_top[pmtpos_top['i']==pmt]['x'].values[0], 
                         pmtpos_top[pmtpos_top['i']==pmt]['y'].values[0]),
                         pmt_radius,
                         color ="white") 
                      )   
    p = mpl.collections.PatchCollection(circles_top, alpha=1.0, edgecolor='black', zorder = 2 )
    p.set_array(pmthits[0:PMTs_top])
    p.set_norm(scale(vmin_top, vmax_top))
    p.set_cmap(cmap)
    for pmt in pmtpos_top["i"]:
        ax_top.text(pmtpos_top[pmtpos_top['i']==pmt]['x'].values[0], 
                    pmtpos_top[pmtpos_top['i']==pmt]['y'].values[0],
                    str(pmt), 
                    va = "center", ha="center")
    ax_top.add_collection(p)    
    ax_top.set_xlim(-70,70)    
    ax_top.set_ylim(-70,70)
    ax_top.set_xlabel("X [ cm ]", fontsize=18)
    ax_top.set_ylabel("Y [ cm ]", fontsize=18)
    ax_top_cbar = fig.add_axes([0.445, 0.10, 0.025, 0.8])
    cbar_top = plt.colorbar(p, cax = ax_top_cbar)
    ####
    ax_bottom = fig.add_axes([0.54, 0.10, 0.40, 0.80])
    ax_bottom.set_title("Bottom array", fontsize=12)
    ax_bottom.add_patch(plt.Circle( (0., 0.),tpc_radius, facecolor="none", edgecolor="black" ) )
    
    circles_bottom = []
    for pmt in pmtpos_bottom["i"]:
        circles_bottom.append(
            plt.Circle( (pmtpos_bottom[pmtpos_bottom['i']==pmt]['x'].values[0], 
                         pmtpos_bottom[pmtpos_bottom['i']==pmt]['y'].values[0]),
                         pmt_radius,
                         color ="white") 
                      )  
    p = mpl.collections.PatchCollection(circles_bottom, alpha=1.0, edgecolor='black', zorder = 2 )
    p.set_cmap(cmap)
    p.set_norm(scale(vmin_bottom, vmax_bottom) )
    p.set_array(pmthits[PMTs_top:PMTs_top+PMTs_bottom])
    for pmt in pmtpos_bottom["i"]:
        ax_bottom.text(pmtpos_bottom[pmtpos_bottom['i']==pmt]['x'].values[0], 
                       pmtpos_bottom[pmtpos_bottom['i']==pmt]['y'].values[0],
                       str(pmt), 
                       va = "center", ha="center")
    ax_bottom.add_collection(p)    
    ax_bottom.set_xlim(-70,70)    
    ax_bottom.set_ylim(-70,70)  
    ax_bottom.set_xlabel("X [ cm ]", fontsize=18)
    ax_bottom.set_ylabel("Y [ cm ]", fontsize=18)
    ax_bottom_cbar = fig.add_axes([0.945, 0.10, 0.025, 0.8])
    cbar_bottom = plt.colorbar(p, cax = ax_bottom_cbar)
    return fig, {"ax_top": ax_top, 
                 "ax_bottom": ax_bottom, 
                 "cbar_top": cbar_top, 
                 "cbar_bottom": cbar_bottom}
def plot_one_array(pmthits , 
                   pmtpositions,
                   array = "top", 
                   cmap = "YlGn", 
                   cbar_lim = "global",
                   tpc_radius = 66.4,
                   pmt_radius = 3.875, 
                   scale = colors.Normalize,
                   figsize=(10.5,10.0),
                   show_cbar = True,
                    ):
    
    fig = plt.figure(figsize=figsize, facecolor="w")
    ###
    pmtpos = pmtpositions# straxen.pmt_positions()
    pmtpos_cur = pmtpos[pmtpos['array']==array]
    PMTs_top = pmtpos[pmtpos['array']=="top"].shape[0]    
    PMTs_bottom =pmtpos[pmtpos['array']=="bottom"].shape[0]  
    if array =="top":
        startpmt = 0
        endpmt = PMTs_top
    elif array=="bottom":
        startpmt = PMTs_top
        endpmt = PMTs_top + PMTs_bottom
    else:
        print("Error! Unknown array type %s"%array)
        return
    ###
    if cbar_lim == "global":
        vmin = pmthits[startpmt:endpmt].min()
        vmax = pmthits[startpmt:endpmt].max() 
    elif len(cbar_lim)==2:
        vmin = cbar_lim[0]
        vmax = cbar_lim[1]        
    ###
    yfrac = 0.82
    xfrac = yfrac*figsize[1]/figsize[0]
    #print(yfrac, xfrac)
    ax = fig.add_axes([0.08, 0.10, xfrac, yfrac])
    ax.add_patch(plt.Circle( (0., 0.), tpc_radius, facecolor="none", edgecolor="black" ) )
    
    circles = []
    
    for pmt in pmtpos_cur["i"]:
        circles.append(
            plt.Circle( (pmtpos_cur[pmtpos_cur['i']==pmt]['x'].values[0], 
                         pmtpos_cur[pmtpos_cur['i']==pmt]['y'].values[0]),
                         pmt_radius,
                         facecolor ="white") 
                      ) 
    p = mpl.collections.PatchCollection(circles, alpha=1.0, edgecolor='black', zorder = 2 )
    p.set_array(pmthits[startpmt:endpmt])
    p.set_norm(scale(vmin, vmax))
    p.set_cmap(cmap)
    for pmt in pmtpos_cur["i"]:
        ax.text(pmtpos_cur[pmtpos_cur['i']==pmt]['x'].values[0], 
                pmtpos_cur[pmtpos_cur['i']==pmt]['y'].values[0],
                str(pmt), 
                va = "center", ha="center")
        
    ax.add_collection(p)  
    ax.set_xlim(-70,70)    
    ax.set_ylim(-70,70)
    ax.text(0.02,0.96,"Array : %s"%array, transform = ax.transAxes, 
           fontsize=20, ha ="left")
    if show_cbar:
        ax_cbar = fig.add_axes([0.08+xfrac+0.015, 0.10, 0.035, yfrac])
        cbar = plt.colorbar(p, cax = ax_cbar)
    else: ax_cbar=None
    ax.set_xlabel("X [ cm ]", fontsize=18)
    ax.set_ylabel("Y [ cm ]", fontsize=18)
    return (fig, ax,ax_cbar)
