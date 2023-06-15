import numpy as np
from jwst.stpipe import Step
from jwst import datamodels

class ReferenceCorrectionCustomStep(Step):
    """
    Background and 1/f noise correction
    
    The CDS background is corrected column-wise estimating the median over the
    set of 12 pixels in each colum: the 6 from the top and the 6 from the bottom.
    This also takes care of any eventual backgrounds (although tese are 
    negligible and therefore irrelevant for the Poisson noise estimate).

    In the same step, I update the pixel noise estimates:
        shot noise variance: unchanged.
        RON variance: is increaded becasue of the pixel-wise background subtraction
    """
    
    class_alias = "RefCorr"

    spec = """
        int_name = string(default='')
        save_opt = boolean(default=False) # Save optional output
        opt_name = string(default='')
    """
    def process(self, input):
        with datamodels.CubeModel(input) as input_model:
            n0, n1, n2 = input_model.shape
            background = np.ma.stack( 
                (input_model.data[:,0:6, :], input_model.data[:, -6:, :]), 
                axis=1
            ).reshape(n0, -1, n2)
            median_background = np.ma.median(background, axis = 1)
            
            output_model = input_model.copy()
            for k in range(n0):
                output_model.data[k] = input_model.data[k]-median_background[k]
            
        # Update noise estimates
        #   not fully correct as it does not account for 
        #   flagged pixel and outliers
        output_model.var_rnoise  *= 1.0 + 1.0/background.shape[1] 
        setattr(output_model, "background", median_background)
        return output_model
