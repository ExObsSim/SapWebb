import numpy as np
from jwst.stpipe import Step
from jwst import datamodels

class TimeAverageStep(Step):
    """
    Time average the pixel timeline    
    In this way we deal with transients.
    """
    
    class_alias = "TimeAverage"

    spec = """
        int_name = string(default='')
        num_ave  = integer(default=25)
        save_opt = boolean(default=False) # Save optional output
        opt_name = string(default='')
    """
    def process(self, input):
        num_ave = self.num_ave
        with datamodels.CubeModel(input) as input_model:
            n0, n1, n2 = input_model.shape
            new_n0 = n0 - n0%num_ave
            mask = input_model.dq[0:new_n0, ...] > 0
            data = np.ma.array(data = input_model.data[0:new_n0, ...],
                              mask = mask,
                              fill_value = 0.0)
            varp = np.ma.array(data = input_model.var_poisson[0:new_n0, ...],
                              mask = mask,
                              fill_value = 0.0)
            varr = np.ma.array(data = input_model.var_rnoise[0:new_n0, ...],
                              mask = mask,
                              fill_value = 0.0)
            count = data.reshape(-1, num_ave, n1, n2).count(axis=1)
            data  = data.reshape(-1, num_ave, n1, n2).mean(axis=1)
            varp  = varp.reshape(-1, num_ave, n1, n2).mean(axis=1)/count
            varr  = varr.reshape(-1, num_ave, n1, n2).mean(axis=1)/count
            err   = np.sqrt(varp + varr)
            dq    = np.ma.getmask(data).astype(np.uint32)
            
            
            
            out_model = datamodels.CubeModel(
                data        = data.filled(fill_value=0.0),
                dq          = dq,
                var_poisson = varp.filled(fill_value=0.0),
                var_rnoise  = varr.filled(fill_value=0.0),
                err         = err.filled(fill_value=0.0))
                
            out_model.update(input_model)
            if hasattr(input_model, 'ngroup'):
                ngroup = input_model.ngroup[0:new_n0, ...].reshape(-1, num_ave, n1, n2).sum(axis=1)
                setattr(out_model, 'ngroup', ngroup)
            
            if hasattr(input_model, 'background'):
                background = input_model.background[0:new_n0, ...].reshape(-1, num_ave, n2).mean(axis=1)
                setattr(out_model, 'background', background)
            
        
        return out_model
   
