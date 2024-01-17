import numpy as np
import os, time
import logging


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from jwst.dq_init import DQInitStep
from jwst.saturation import SaturationStep
from jwst.superbias import SuperBiasStep
from jwst.linearity import LinearityStep
from jwst.dark_current import DarkCurrentStep
from jwst.assign_wcs import AssignWcsStep
from jwst.srctype import SourceTypeStep
from jwst.extract_2d import Extract2dStep
from jwst.wavecorr import WavecorrStep
from jwst import datamodels

from .cds_custom_step import CDSCustomStep
from .refcorr_custom_step import ReferenceCorrectionCustomStep
from .flagging_custom_step import FlaggingCustomStep
from .timeaverage_custom_step import TimeAverageStep

def FlagSaturatedColumns(input_model, ncols=1):
    """
    Flag columns containing at least one saturated pixels as saturated.
    The flagging is extended to the preceding and following ncols columns.
    The algorithm is implemented as follows:
    
    for each group
        select saturated pixel, then
            flag the corresponding columns as saturated, and flag as saturated the adjacent columns as well
    
    This selects any pixel saturation transients, flagging to saturated the whole column, including 
    cosmic ray and space ball saturation events.
    
    Parameters
    ----------
    input_model : data model
        input data model, assumed to be of type RampModel

    ncols : int
        number columns to flag that are adjacent to the column containing the
        saturated pixel. If ncols=0, flag only the column containing the saturated
        pixel

    Returns
    -------
    output_model : data model
        same data model as input_model
    """
    data = input_model
    flag_sat = datamodels.dqflags.pixel['SATURATED'] | datamodels.dqflags.pixel['DO_NOT_USE']

    idx = np.where(data.groupdq & datamodels.dqflags.pixel['SATURATED'])
    for i, j, k in zip(idx[0], idx[1], idx[3]):
        data.groupdq[i, j, :, k-ncols:k+ncols+1] |= flag_sat
            
    return data


def run(in_file, out_file=None, num_ave=25):
    
    # DQInitStep from STScI JWST pipeline
    start = time.time()
    dq_init_step = DQInitStep()
    data = dq_init_step(in_file)
    end = time.time()
    log.info('DQInitStep Completed in {:.2f} s'.format(end-start))
     
   
    # SaturationStep from STScI JWST pipeline
    start = time.time()
    saturation_step = SaturationStep()
    data = saturation_step.run(data)
    end = time.time()
    log.info('SaturationStep Completed in {:.2f} s'.format(end-start))

    # SuperBiasStep from STScI JWST pipeline
    start = time.time()
    superbias_step = SuperBiasStep()
    data = superbias_step.run(data)
    end = time.time()
    log.info('SuperBiasStep Completed in {:.2f} s'.format(end-start))
    
    # LinearityStep from STScI JWST pipeline
    start = time.time()
    linearity_step = LinearityStep()
    data = linearity_step.run(data)
    end = time.time()
    log.info('LinearityStep Completed in {:.2f} s'.format(end-start))

    # DarkCurrentStep from STScI JWST pipeline
    start = time.time()
    dark_step = DarkCurrentStep()
    data = dark_step.run(data)
    end = time.time()
    log.info('DarkCurrentStep Completed in {:.2f} s'.format(end-start))
    
    # FlagSaturatedColumns custom step
    start = time.time()
    data = FlagSaturatedColumns(data, ncols=1)
    end = time.time()
    log.info('FlagSaturatedColumns Completed in {:.2f} s'.format(end-start))    

    # CDSCustomStep
    start = time.time()
    cds_step = CDSCustomStep()
    data_cube = cds_step(data)
    end = time.time()
    log.info('CDSCustomStep Completed in {:.2f} s'.format(end-start))    
    
    del data
    
    # ReferenceCorrectionStep 
    start = time.time()
    ref_corr = ReferenceCorrectionCustomStep()
    data_cube = ref_corr(data_cube)
    end = time.time()
    log.info('ReferenceCorrectionStep Completed in {:.2f} s'.format(end-start))    
    
    # FlaggingCustomStep 
    start = time.time()
    flag_step = FlaggingCustomStep()
    data_cube = flag_step(data_cube)
    end = time.time()
    log.info('FlaggingCustomStep Completed in {:.2f} s'.format(end-start))    
    

    # TimeAverageStep 
    start = time.time()
    ta_step = TimeAverageStep(num_ave=num_ave)
    data_cube_mean = ta_step(data_cube)
    end = time.time()
    log.info('FlaggingCustomStep Completed in {:.2f} s'.format(end-start))    
    
    # AssignWcsStep from STScI JWST pipeline
    assignWcs = AssignWcsStep()
    data_cube_mean = assignWcs.run(data_cube_mean)
    
    # SourceTypeStep from STScI JWST pipeline
    srctype=SourceTypeStep()
    data_cube_mean = srctype(data_cube_mean)
    
    # Extract2dStep from STScI JWST pipeline
    extract2d = Extract2dStep()
    _dd_ = extract2d(data_cube_mean)
    
    # WavecorrStep from STScI JWST pipeline
    wavecorr = WavecorrStep()
    _dd_ = wavecorr(_dd_)
    
    # Here I assign the wavelength solution to the datacube, avoiding
    #   messing with the data as extract2d does.
      
    yslice = slice(_dd_.ystart-1, _dd_.ystart-1+_dd_.ysize)
    xslice = slice(_dd_.xstart-1, _dd_.xstart-1+_dd_.xsize) 
    
    data_cube_mean.wavelength += np.nan
    data_cube_mean.wavelength[yslice, xslice] = _dd_.wavelength
    
    if out_file:
        data_cube_mean.save(out_file)
        return 0
    
    return data_cube_mean, data_cube
    
          
if __name__ == "__main__":
    segment = 1
    input_dir = "/export/sata01/USER_DATA/enzo.pascale/JWST/jwst01366004001"
    uncal_file = "jw01366004001_04101_00001-seg{:03d}_nrs1_uncal.fits".format(segment)

    in_file = os.path.expanduser(os.path.join(input_dir, uncal_file))
    run(in_file)
          
