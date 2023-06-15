import numpy as np
from jwst.stpipe import Step
from jwst import datamodels

from jwst.ramp_fitting.ramp_fit_step import  get_reference_file_subarrays
from jwst.stpipe import Step
from jwst import datamodels
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def get_gain_ron_2d(input_model):
    """ 
    Returns the readnoise and gain subarray in units of counts (electrons).
    Updates input_model.meta.exposure.gain_factor when appropriate for BOTS. 
    
    Gain is a conversion from DN to counts for each detector pixel. The 
    readout noise (RON) is the noise per pixel on each non-destructive read (NDR).
    This function retrieves the gain and RON calibrations from the archive, 
    reproducing in part the functionality implemented in the standard pipeline 
    RampFitStep.
    
    Parameters
    ----------
    input_model : data model
        input data model, assumed to be of type RampModel

    Returns
    -------
    readnoise_2d : float, 2D array
        readnoise subarray in units of CT

    gain_2d : float, 2D array
        gain subarray in units of CT
    
    """
    st =  Step()
    readnoise_filename = st.get_reference_file(input_model, 'readnoise')
    gain_filename = st.get_reference_file(input_model, 'gain')
    frames_per_group = input_model.meta.exposure.nframes
    log.info('Using READNOISE reference file: %s', readnoise_filename)
    log.info('Using GAIN reference file: %s', gain_filename)
    
    with datamodels.ReadnoiseModel(readnoise_filename) as readnoise_model, \
        datamodels.GainModel(gain_filename) as gain_model:
            
            if gain_model.meta.exposure.gain_factor is not None:
                input_model.meta.exposure.gain_factor = gain_model.meta.exposure.gain_factor
            frames_per_group = input_model.meta.exposure.nframes
            readnoise_2d, gain_2d = get_reference_file_subarrays(input_model, readnoise_model, gain_model, frames_per_group) 
    return gain_2d, readnoise_2d*gain_2d



class CDSCustomStep(Step):
    """
    The default CDS is estimated as the difference between the last and first 
    groups in one integration.

    All pixels in a column  𝑐 are set to saturated in a group  𝑔 if the given
    column  𝑐 has at least one pixel flagged as saturated.

    Therefore, the CDS for these columns need to be built from the difference 
    between group  𝑔−1 and group  0.
    
    If  𝑔=0, then the CDS cannot be build and is flagged DO_NOT_USE.

    To implement this, I estimate the difference across group flags, and select
    transients, i.e. select those columns that are set to saturated in  𝑔, but 
    are not in  𝑔−1.

    Then I select the frames, groups and colums that are newly saturated, and 
    replace the default CDS with the new one.The CDS is normalised to the number of groups (ngroup) used to estimate it. Therefore, it is a rate, rather than a classical CDS.

    Shot noise variance is proportional to the signal:
        𝑉𝑎𝑟𝑃=𝐶𝐷𝑆/ngroup∝1/ngroup

    The RON variance is
        𝑉ar𝑅 = 2𝜎^2/ngroup^2
    
    The error estimate is the sqrt(VarP + VarR).
    """
    
    class_alias = "CDS"

    spec = """
        int_name = string(default='')
        save_opt = boolean(default=False) # Save optional output
        opt_name = string(default='')
    """
    def process(self, input):

        with datamodels.RampModel(input) as input_model:
            gain_2d, readnoise_2d = get_gain_ron_2d(input_model)
            
            n0, n1, n2, n3 = input_model.shape
            
            p = np.diff(
                input_model.groupdq[:,:,n2//2,:] & datamodels.dqflags.pixel['SATURATED'], 
                axis=1)
            idx = np.where(p == datamodels.dqflags.pixel['SATURATED'])

            # Define defaults
            cds  = (input_model.data[:, -1, :, :] - input_model.data[:, 0, :, :])/(n1-1)
            mask = input_model.groupdq[:, -1, :, :] | input_model.groupdq[:, 0, :, :] | input_model.pixeldq
            ngroup = np.zeros_like(cds) + (n1-1)

            for i, j, k in zip(*idx):
                # note: j = 0 if saturation occurs in group 1.   CDS cannot be estimated
                #       j = 1 if saturation occurs in group 2.   CDS estimated from groups 1 and 0; ngroup = 1
                #       j = 2 if saturation occurs in group 3.   CDS estimated from groups 2 and 0; ngroup = 2
                #                ...                             ...                                ...
                #                saturation occurs in group j+1. CDS estimated from groups j and 0; ngroup = j

                norm = 1 if j == 0 else j
                do_not_use = datamodels.dqflags.pixel['DO_NOT_USE'] if j == 0 else 0
                #do_not_use = dqflags.pixel['DO_NOT_USE'] 
                cds[i, :, k]    = (input_model.data[i, j, :, k] - input_model.data[i, 0, :, k])/norm
                mask[i, :, k]   = input_model.groupdq[i, j, :, k] | input_model.groupdq[i, 0, :, k] | \
                                  input_model.pixeldq[:, k] | do_not_use
                ngroup[i, :, k] = np.nan if j == 0 else j
    
        cds  = np.ma.array(data= gain_2d*cds, mask=mask > 0)
        varP = cds/ngroup
        varR = np.ma.array(data=2*readnoise_2d**2/ngroup**2, mask=mask > 0)

            
        out_model = datamodels.CubeModel(
            data = cds,
            dq = mask,
            var_poisson = varP,
            var_rnoise = varR,
            err = np.ma.sqrt(varP + varR),
            int_times = input_model.int_times.copy()
        )
        
        setattr(out_model, 'ngroup', ngroup)
        out_model.update(input_model)
        out_model.meta.bunit_data = 'CT/s'
        out_model.meta.bunit_err = 'CT/s'
        out_model.meta.cal_step.ramp_fit = 'COMPLETE'
        
        return out_model
