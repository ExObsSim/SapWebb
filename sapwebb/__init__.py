from .__version__ import __version__
from .config import __url__, __author__, __description__, __branch__, __commit__

import os
if not ( "CRDS_PATH" in os.environ.keys() and 
        "CRDS_SERVER_URL" in os.environ.keys() ):

    raise  Exception("Environment variables CRDS_PATH or CRDS_SERVER_URL not configured.")

from .uncal2L1 import *
from .cds_custom_step import *
from .refcorr_custom_step import *
from .flagging_custom_step import *
from .timeaverage_custom_step import *

