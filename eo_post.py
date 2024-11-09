#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2024 James James Johnson. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
1. merging PNGs, JPGs, PDFs, SVGs across jobs. PNGs, JPGs, and SVGs are merged in HTML.
   a. https://stackoverflow.com/questions/3444645/merge-pdf-files
   b. https://stackoverflow.blog/2022/12/27/picture-perfect-images-with-the-modern-element/
   c. https://stackoverflow.com/questions/25002310/best-practice-for-using-svg-images
      https://stackoverflow.com/questions/4476526/do-i-use-img-object-or-embed-for-svg-files?rq=3
2. zip compression of results
"""

import sys
from glob import glob
try:
    from PyPDF2 import PdfReader, PdfWriter, PdfMerger
except ImportError:
    from pyPdf import PdfFileReader, PdfFileWriter, PdfMerger

###############################################################################

def main() :
    '''<function description>

    Parameters
    ----------
    <parameter name> : <parameter type> [, optional]
        <parameter description>. [(default is <default value>)]
    ...

    Raises
    ------
    <exception type>
        <exception message/description>
    ...

    Returns
    -------
    <returned value name> : <returned value type>
        <returned value description>
    '''
    pass

###############################################################################

if __name__ ==  '__main__' :
    main()

###############################################################################
