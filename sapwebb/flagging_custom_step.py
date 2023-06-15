from astropy.stats import sigma_clip
from jwst.stpipe import Step
from jwst import datamodels

class FlaggingCustomStep(Step):
    """
    Flag outlier¶
    Here we use the sigma_clip Astropy implementation, applied pixel-by-pixel
    along the temporal axis. This is not fully "correct" because of the 
    transit signal contributing to the RMS. In principle, the transit signal 
    should be removed before applying the the sigma_clip. We will consider 
    a more robust algorithm if needed, later on.
    """
    
    class_alias = "Flagging"

    spec = """
        int_name = string(default='')
        save_opt = boolean(default=False) # Save optional output
        opt_name = string(default='')
    """
    def process(self, input):
        with datamodels.CubeModel(input) as input_model:
            _cds_ = sigma_clip(input_model.data, sigma=5, axis=0)
            
            new_outliers = (input_model.dq > 0) ^ _cds_.mask 
            out_model = input_model.copy()
        
        out_model.dq[new_outliers] |= datamodels.dqflags.pixel['JUMP_DET'] | datamodels.dqflags.pixel['DO_NOT_USE']
        out_model.data = _cds_
        
        return out_model
