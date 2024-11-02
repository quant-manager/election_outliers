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

###############################################################################

"""Election Outliers

The software tool "Election Outliers" may have a purpose to certify an election
result prior to, during, and after its precinct-level audit and recounting. In
fact, anyone using this tool can "certify" elections in the comfort of one's
own home. Specifically, the tool allows to achieve the following objectives:

1. Identify precincts that happen to cause the worst predictive power of their
   election results.
2. Determine whether this relatively small set of the least predictable
   precints produces results that are predominantly beneficial
   (or disadvantageous) to specific electoral choice(s).
3. Determine whether these least predictable precints have an impact on the
   rank ordering among electoral choices, especially on the winner of the
   entire contest.
4. Select these least predictable precints as the best candidate-precincts for
   detailed audit and vote recount, mark them as "audited" in the tool after a
   reliable and successful audit is completed, and iteratively rerun the tool
   until the impact of these biased least predictable but audited precints on
   rank ordering of election choices is eliminated from the right-most tail,
   since these precincts are moved to a different slot in the ordered sequence
   of all precincts.

This tool accepts comma separated value files (.csv) as well as ...

This script requires that 'pandas' be installed within the Python
environment you are running this script in.

This module is documented with the format from
https://realpython.com/documenting-python-code

This file can also be imported as a module and contains the following
functions:

    * <functuion name> - <function description>
    * ...
    * main - the main function of the script
"""

###############################################################################

import numpy as np
import pandas as pd
import math
import sys
import time
import warnings
import os
import shutil
import json
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.metrics import d2_absolute_error_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
# mean_tweedie_deviance with p = 0:
from sklearn.metrics import root_mean_squared_error
# mean_tweedie_deviance with p = 1:
from sklearn.metrics import mean_poisson_deviance
# mean_tweedie_deviance with p = 2:
from sklearn.metrics import mean_gamma_deviance
from sklearn.metrics import root_mean_squared_log_error
from sklearn.metrics import mean_absolute_percentage_error
import datetime
import pytz
import re
import uuid
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from scipy.stats import multivariate_hypergeom
from scipy.stats import hypergeom
from decimal_hypergeom import custom_hypergeom_min_cdf_sf
from decimal_hypergeom import scipy_hypergeom_min_cdf_sf
from decimal import Decimal
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None  # default='warn'

FLT_NANOSECONDS_PER_MILLISECOND = float(1_000_000)
FLT_NANOSECONDS_PER_SECOND = float(1_000_000_000)

STR_INPUT_JOBS_BATCH_CSV_FILE_NAME = "_default_batch"

STR_INPUT_BATCHES_BAT_PATH = "input_batches_bat"
STR_INPUT_BATCHES_CSV_PATH = "input_batches_csv"
STR_INPUT_DATA_CSV_PATH = "input_data_csv"
STR_INPUT_JOBS_JSON_PATH = "input_jobs_json"

STR_OUTPUT_DATA_NPAR_CSV_PATH = "output_data_npar_csv"
STR_OUTPUT_DATA_PARM_CSV_PATH = "output_data_parm_csv"
STR_OUTPUT_DIAGN_PARM_CSV_PATH = "output_diagn_parm_csv"
STR_OUTPUT_JOBS_DIR_PATH = "output_jobs_dir"
STR_OUTPUT_PARAMS_PARM_CSV_PATH = "output_params_parm_csv"
STR_OUTPUT_PLOTS_NPAR_JPG_PATH = "output_plots_npar_jpg"
STR_OUTPUT_PLOTS_PARM_JPG_PATH = "output_plots_parm_jpg"
STR_OUTPUT_PLOTS_NPAR_PDF_PATH = "output_plots_npar_pdf"
STR_OUTPUT_PLOTS_PARM_PDF_PATH = "output_plots_parm_pdf"
STR_OUTPUT_PLOTS_NPAR_PNG_PATH = "output_plots_npar_png"
STR_OUTPUT_PLOTS_PARM_PNG_PATH = "output_plots_parm_png"
STR_OUTPUT_PLOTS_NPAR_SVG_PATH = "output_plots_npar_svg"
STR_OUTPUT_PLOTS_PARM_SVG_PATH = "output_plots_parm_svg"
STR_OUTPUT_SCORES_PARM_CSV_PATH = "output_scores_parm_csv"

STR_OUTPUT_DATA_NPAR_FILE_SUFFIX = "_data_npar"
STR_OUTPUT_DATA_PARM_FILE_SUFFIX = "_data_parm"
STR_OUTPUT_DIAGN_PARM_FILE_SUFFIX = "_diagn_parm"
STR_OUTPUT_PARAMS_PARM_FILE_SUFFIX = "_params_parm"
STR_OUTPUT_PLOTS_NPAR_FILE_SUFFIX = "_plots_npar"
STR_OUTPUT_PLOTS_PARM_FILE_SUFFIX = "_plots_parm"
STR_OUTPUT_SCORES_PARM_FILE_SUFFIX = "_scores_parm"

STR_PLOT_TITLE_NPAR_SUFFIX = " Non-Param. Model."
STR_PLOT_TITLE_PARM_OLS_GAM_SUFFIX = " OLS Gen. Add. Model."
STR_PLOT_TITLE_PARM_LASSO_GAM_SUFFIX = " Lasso Gen. Add. Model."

STR_IS_AUDITED_COLUMN_NAME = "IS_AUDITED"
STR_COUNTY_NAME_COLUMN_NAME = "COUNTY_NAME"
STR_PRECINCT_NAME_COLUMN_NAME = "PRECINCT_NAME"
STR_PRECINCT_CODE_COLUMN_NAME = "PRECINCT_CODE"

# Keep this intercept flag set at False! This is methodologically correct:
BOOL_FIT_REGRESSION_INTERCEPT = False

STR_COL_NAME_FOR_CUMUL_PLOT_X_AXIS = "Row_Total_Cumul_Pct"
STR_COL_NAME_POSTFIX_FOR_CUMUL_PLOT_Y_AXIS = "_Cumul_Pct"

###############################################################################

def print_exception_message(exception) :
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print("Exception of type " + str(exc_type) + ' in file "' +
          fname + '" on line ' + str(exc_tb.tb_lineno) + ": " +
          str(exception)) # TBD: save the error message to the Error log file.

###############################################################################

def assemble_predictors(
        str_predicted_for_actual_votes_counts_column_name,
        df_actual_vote_counts = None,
        df_benchmark_vote_counts = None,
        #######################################################################
        lst_str_actual_merge_keys = [],
        lst_str_benchmark_merge_keys = [],
        lst_str_actual_all_vote_count_column_names = [],
        lst_str_benchmark_all_vote_count_column_names = [],
        #######################################################################
        lst_str_predicting_from_ln_of_incremented_actual_vote_count_column_names = [],
        lst_str_predicting_from_power_one_quarter_of_actual_vote_count_column_names = [],
        lst_str_predicting_from_square_root_of_actual_vote_count_column_names = [],
        lst_str_predicting_from_power_three_quarters_of_actual_vote_count_column_names = [],
        lst_str_predicting_from_actual_vote_count_column_names = [],
        lst_str_predicting_from_power_one_and_a_half_of_actual_vote_count_column_names = [],
        lst_str_predicting_from_squared_actual_vote_count_column_names = [],
        #
        lst_str_predicting_from_ln_of_incremented_benchmark_vote_count_column_names = [],
        lst_str_predicting_from_power_one_quarter_of_benchmark_vote_count_column_names = [],
        lst_str_predicting_from_square_root_of_benchmark_vote_count_column_names = [],
        lst_str_predicting_from_power_three_quarters_of_benchmark_vote_count_column_names = [],
        lst_str_predicting_from_benchmark_vote_count_column_names = [],
        lst_str_predicting_from_power_one_and_a_half_of_benchmark_vote_count_column_names = [],
        lst_str_predicting_from_squared_benchmark_vote_count_column_names = [],
        #
        lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_actual_vote_count_column_names = [],
        lst_pairs_str_predicting_from_square_root_of_actual_interactions_actual_vote_count_column_names = [],
        lst_pairs_str_predicting_from_actual_interactions_actual_vote_count_column_names = [],
        #
        lst_pairs_str_predicting_from_fourth_root_of_benchmark_interactions_benchmark_vote_count_column_names = [],
        lst_pairs_str_predicting_from_square_root_of_benchmark_interactions_benchmark_vote_count_column_names = [],
        lst_pairs_str_predicting_from_benchmark_interactions_benchmark_vote_count_column_names = [],
        #
        lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_benchmark_vote_count_column_names = [],
        lst_pairs_str_predicting_from_square_root_of_actual_interactions_benchmark_vote_count_column_names = [],
        lst_pairs_str_predicting_from_actual_interactions_benchmark_vote_count_column_names = [],
        #######################################################################
        bool_drop_predicted_variable_during_computation_of_predicting_tally = True,
        #######################################################################
        bool_predicting_from_ln_of_incremented_actual_tally = False,
        bool_predicting_from_power_one_quarter_of_actual_tally = False,
        bool_predicting_from_square_root_of_actual_tally = False,
        bool_predicting_from_power_three_quarters_of_actual_tally = False,
        bool_predicting_from_actual_tally = False,
        bool_predicting_from_power_one_and_a_half_of_actual_tally = False,
        bool_predicting_from_squared_actual_tally = False,
        #
        bool_predicting_from_ln_of_incremented_benchmark_tally = False,
        bool_predicting_from_power_one_quarter_of_benchmark_tally = False,
        bool_predicting_from_square_root_of_benchmark_tally = False,
        bool_predicting_from_power_three_quarters_of_benchmark_tally = False,
        bool_predicting_from_benchmark_tally = False,
        bool_predicting_from_power_one_and_a_half_of_benchmark_tally = False,
        bool_predicting_from_squared_benchmark_tally = False,
        #
        bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally = False,
        bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally = False,
        bool_predicting_from_actual_tally_interaction_benchmark_tally = False,
        #######################################################################
        bool_predicting_from_ln_of_incremented_actual_tally_interaction_county_indicator = False,
        bool_predicting_from_power_one_quarter_of_actual_tally_interaction_county_indicator = False,
        bool_predicting_from_square_root_of_actual_tally_interaction_county_indicator = False,
        bool_predicting_from_power_three_quarters_of_actual_tally_interaction_county_indicator = False,
        bool_predicting_from_actual_tally_interaction_county_indicator = False,
        bool_predicting_from_power_one_and_a_half_of_actual_tally_interaction_county_indicator = False,
        bool_predicting_from_squared_actual_tally_interaction_county_indicator = False,
        #
        bool_predicting_from_ln_of_incremented_benchmark_tally_interaction_county_indicator = False,
        bool_predicting_from_power_one_quarter_of_benchmark_tally_interaction_county_indicator = False,
        bool_predicting_from_square_root_of_benchmark_tally_interaction_county_indicator = False,
        bool_predicting_from_power_three_quarters_of_benchmark_tally_interaction_county_indicator = False,
        bool_predicting_from_benchmark_tally_interaction_county_indicator = False,
        bool_predicting_from_power_one_and_a_half_of_benchmark_tally_interaction_county_indicator = False,
        bool_predicting_from_squared_benchmark_tally_interaction_county_indicator = False,
        #
        bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator = False,
        bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator = False,
        bool_predicting_from_actual_tally_interaction_benchmark_tally_interaction_county_indicator = False,
        ) :
    '''<function description>
    
    <calling functions>
    <called functions>

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
    def add_or_append_predictors(X, lst_str_X_columns_names, X_partial, lst_partial) :
        if X is None:
            X = X_partial
            lst_str_X_columns_names = lst_partial
        else :
            X = pd.concat([X, X_partial], ignore_index=True, axis=1)
            lst_str_X_columns_names += lst_partial
        return (X, lst_str_X_columns_names)

    X = None
    lst_str_X_columns_names = []
    if df_actual_vote_counts is not None or df_benchmark_vote_counts is not None :

        #######################################################################
        # Exclude str_predicted_for_actual_votes_counts_column_name:

        lst_str_filtered_actual_nat_log_of_incremented_votes_counts_columns_names = [
            e for e in lst_str_predicting_from_ln_of_incremented_actual_vote_count_column_names
            if e != str_predicted_for_actual_votes_counts_column_name]
        lst_str_filtered_actual_power_one_quarter_votes_counts_columns_names = [
            e for e in lst_str_predicting_from_power_one_quarter_of_actual_vote_count_column_names
            if e != str_predicted_for_actual_votes_counts_column_name]
        lst_str_filtered_actual_square_root_votes_counts_columns_names = [
            e for e in lst_str_predicting_from_square_root_of_actual_vote_count_column_names
            if e != str_predicted_for_actual_votes_counts_column_name]
        lst_str_filtered_actual_power_three_quarters_votes_counts_columns_names = [
            e for e in lst_str_predicting_from_power_three_quarters_of_actual_vote_count_column_names
            if e != str_predicted_for_actual_votes_counts_column_name]
        lst_str_filtered_actual_votes_counts_columns_names = [
            e for e in lst_str_predicting_from_actual_vote_count_column_names
            if e != str_predicted_for_actual_votes_counts_column_name]
        lst_str_filtered_actual_power_one_and_a_half_votes_counts_columns_names = [
            e for e in lst_str_predicting_from_power_one_and_a_half_of_actual_vote_count_column_names
            if e != str_predicted_for_actual_votes_counts_column_name]
        lst_str_filtered_actual_squared_votes_counts_columns_names = [
            e for e in lst_str_predicting_from_squared_actual_vote_count_column_names
            if e != str_predicted_for_actual_votes_counts_column_name]

        #######################################################################
        # Exclude str_predicted_for_actual_votes_counts_column_name:

        lst_pairs_str_filtered_fourth_root_of_actual_interactions_actual_votes_counts_columns_names = [
            e for e in lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_actual_vote_count_column_names
            if e[0] != str_predicted_for_actual_votes_counts_column_name and
            e[1] != str_predicted_for_actual_votes_counts_column_name]
        if len(lst_pairs_str_filtered_fourth_root_of_actual_interactions_actual_votes_counts_columns_names) > 0 :
            (tpl_str_filtered_left_fourth_root_of_actual_interactions_actual_votes_counts_columns_names,
             tpl_str_filtered_right_fourth_root_of_actual_interactions_actual_votes_counts_columns_names) = \
                zip(*lst_pairs_str_filtered_fourth_root_of_actual_interactions_actual_votes_counts_columns_names)
            lst_str_filtered_left_fourth_root_of_actual_interactions_actual_votes_counts_columns_names = \
                list(tpl_str_filtered_left_fourth_root_of_actual_interactions_actual_votes_counts_columns_names)
            lst_str_filtered_right_fourth_root_of_actual_interactions_actual_votes_counts_columns_names = \
                list(tpl_str_filtered_right_fourth_root_of_actual_interactions_actual_votes_counts_columns_names)
        else :
            lst_str_filtered_left_fourth_root_of_actual_interactions_actual_votes_counts_columns_names = []
            lst_str_filtered_right_fourth_root_of_actual_interactions_actual_votes_counts_columns_names = []

        lst_pairs_str_filtered_square_root_of_actual_interactions_actual_votes_counts_columns_names = [
            e for e in lst_pairs_str_predicting_from_square_root_of_actual_interactions_actual_vote_count_column_names
            if e[0] != str_predicted_for_actual_votes_counts_column_name and
            e[1] != str_predicted_for_actual_votes_counts_column_name]
        if len(lst_pairs_str_filtered_square_root_of_actual_interactions_actual_votes_counts_columns_names) > 0 :
            (tpl_str_filtered_left_square_root_of_actual_interactions_actual_votes_counts_columns_names,
             tpl_str_filtered_right_square_root_of_actual_interactions_actual_votes_counts_columns_names) = \
                zip(*lst_pairs_str_filtered_square_root_of_actual_interactions_actual_votes_counts_columns_names)
            lst_str_filtered_left_square_root_of_actual_interactions_actual_votes_counts_columns_names = \
                list(tpl_str_filtered_left_square_root_of_actual_interactions_actual_votes_counts_columns_names)
            lst_str_filtered_right_square_root_of_actual_interactions_actual_votes_counts_columns_names = \
                list(tpl_str_filtered_right_square_root_of_actual_interactions_actual_votes_counts_columns_names)
        else :
            lst_str_filtered_left_square_root_of_actual_interactions_actual_votes_counts_columns_names = []
            lst_str_filtered_right_square_root_of_actual_interactions_actual_votes_counts_columns_names = []

        lst_pairs_str_filtered_actual_interactions_actual_votes_counts_columns_names = [
            e for e in lst_pairs_str_predicting_from_actual_interactions_actual_vote_count_column_names
            if e[0] != str_predicted_for_actual_votes_counts_column_name and
            e[1] != str_predicted_for_actual_votes_counts_column_name]
        if len(lst_pairs_str_filtered_actual_interactions_actual_votes_counts_columns_names) > 0 :
            (tpl_str_filtered_left_actual_interactions_actual_votes_counts_columns_names,
             tpl_str_filtered_right_actual_interactions_actual_votes_counts_columns_names) = \
                zip(*lst_pairs_str_filtered_actual_interactions_actual_votes_counts_columns_names)
            lst_str_filtered_left_actual_interactions_actual_votes_counts_columns_names = \
                list(tpl_str_filtered_left_actual_interactions_actual_votes_counts_columns_names)
            lst_str_filtered_right_actual_interactions_actual_votes_counts_columns_names = \
                list(tpl_str_filtered_right_actual_interactions_actual_votes_counts_columns_names)
        else :
            lst_str_filtered_left_actual_interactions_actual_votes_counts_columns_names = []
            lst_str_filtered_right_actual_interactions_actual_votes_counts_columns_names = []

        lst_pairs_str_filtered_fourth_root_of_benchmark_interactions_benchmark_votes_counts_columns_names = [
            e for e in lst_pairs_str_predicting_from_fourth_root_of_benchmark_interactions_benchmark_vote_count_column_names
            if e[0] != str_predicted_for_actual_votes_counts_column_name and
            e[1] != str_predicted_for_actual_votes_counts_column_name]
        if len(lst_pairs_str_filtered_fourth_root_of_benchmark_interactions_benchmark_votes_counts_columns_names) > 0 :
            (tpl_str_filtered_left_fourth_root_of_benchmark_interactions_benchmark_votes_counts_columns_names,
             tpl_str_filtered_right_fourth_root_of_benchmark_interactions_benchmark_votes_counts_columns_names) = \
                zip(*lst_pairs_str_filtered_fourth_root_of_benchmark_interactions_benchmark_votes_counts_columns_names)
            lst_str_filtered_left_fourth_root_of_benchmark_interactions_benchmark_votes_counts_columns_names = \
                list(tpl_str_filtered_left_fourth_root_of_benchmark_interactions_benchmark_votes_counts_columns_names)
            lst_str_filtered_right_fourth_root_of_benchmark_interactions_benchmark_votes_counts_columns_names = \
                list(tpl_str_filtered_right_fourth_root_of_benchmark_interactions_benchmark_votes_counts_columns_names)
        else :
            lst_str_filtered_left_fourth_root_of_benchmark_interactions_benchmark_votes_counts_columns_names = []
            lst_str_filtered_right_fourth_root_of_benchmark_interactions_benchmark_votes_counts_columns_names = []

        lst_pairs_str_filtered_square_root_of_benchmark_interactions_benchmark_votes_counts_columns_names = [
            e for e in lst_pairs_str_predicting_from_square_root_of_benchmark_interactions_benchmark_vote_count_column_names
            if e[0] != str_predicted_for_actual_votes_counts_column_name and
            e[1] != str_predicted_for_actual_votes_counts_column_name]
        if len(lst_pairs_str_filtered_square_root_of_benchmark_interactions_benchmark_votes_counts_columns_names) > 0 :
            (tpl_str_filtered_left_square_root_of_benchmark_interactions_benchmark_votes_counts_columns_names,
             tpl_str_filtered_right_square_root_of_benchmark_interactions_benchmark_votes_counts_columns_names) = \
                zip(*lst_pairs_str_filtered_square_root_of_benchmark_interactions_benchmark_votes_counts_columns_names)
            lst_str_filtered_left_square_root_of_benchmark_interactions_benchmark_votes_counts_columns_names = \
                list(tpl_str_filtered_left_square_root_of_benchmark_interactions_benchmark_votes_counts_columns_names)
            lst_str_filtered_right_square_root_of_benchmark_interactions_benchmark_votes_counts_columns_names = \
                list(tpl_str_filtered_right_square_root_of_benchmark_interactions_benchmark_votes_counts_columns_names)
        else :
            lst_str_filtered_left_square_root_of_benchmark_interactions_benchmark_votes_counts_columns_names = []
            lst_str_filtered_right_square_root_of_benchmark_interactions_benchmark_votes_counts_columns_names = []

        lst_pairs_str_filtered_benchmark_interactions_benchmark_votes_counts_columns_names = [
            e for e in lst_pairs_str_predicting_from_benchmark_interactions_benchmark_vote_count_column_names
            if e[0] != str_predicted_for_actual_votes_counts_column_name and
            e[1] != str_predicted_for_actual_votes_counts_column_name]
        if len(lst_pairs_str_filtered_benchmark_interactions_benchmark_votes_counts_columns_names) > 0 :
            (tpl_str_filtered_left_benchmark_interactions_benchmark_votes_counts_columns_names,
             tpl_str_filtered_right_benchmark_interactions_benchmark_votes_counts_columns_names) = \
                zip(*lst_pairs_str_filtered_benchmark_interactions_benchmark_votes_counts_columns_names)
            lst_str_filtered_left_benchmark_interactions_benchmark_votes_counts_columns_names = \
                list(tpl_str_filtered_left_benchmark_interactions_benchmark_votes_counts_columns_names)
            lst_str_filtered_right_benchmark_interactions_benchmark_votes_counts_columns_names = \
                list(tpl_str_filtered_right_benchmark_interactions_benchmark_votes_counts_columns_names)
        else :
            lst_str_filtered_left_benchmark_interactions_benchmark_votes_counts_columns_names = []
            lst_str_filtered_right_benchmark_interactions_benchmark_votes_counts_columns_names = []

        lst_pairs_str_filtered_fourth_root_of_actual_interactions_benchmark_votes_counts_columns_names = [
            e for e in lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_benchmark_vote_count_column_names
            if e[0] != str_predicted_for_actual_votes_counts_column_name and
            e[1] != str_predicted_for_actual_votes_counts_column_name]
        if len(lst_pairs_str_filtered_fourth_root_of_actual_interactions_benchmark_votes_counts_columns_names) > 0 :
            (tpl_str_filtered_left_fourth_root_of_actual_interactions_benchmark_votes_counts_columns_names,
             tpl_str_filtered_right_fourth_root_of_actual_interactions_benchmark_votes_counts_columns_names) = \
                zip(*lst_pairs_str_filtered_fourth_root_of_actual_interactions_benchmark_votes_counts_columns_names)
            lst_str_filtered_left_fourth_root_of_actual_interactions_benchmark_votes_counts_columns_names = \
                list(tpl_str_filtered_left_fourth_root_of_actual_interactions_benchmark_votes_counts_columns_names)
            lst_str_filtered_right_fourth_root_of_actual_interactions_benchmark_votes_counts_columns_names = \
                list(tpl_str_filtered_right_fourth_root_of_actual_interactions_benchmark_votes_counts_columns_names)
        else :
            lst_str_filtered_left_fourth_root_of_actual_interactions_benchmark_votes_counts_columns_names = []
            lst_str_filtered_right_fourth_root_of_actual_interactions_benchmark_votes_counts_columns_names = []

        lst_pairs_str_filtered_square_root_of_actual_interactions_benchmark_votes_counts_columns_names = [
            e for e in lst_pairs_str_predicting_from_square_root_of_actual_interactions_benchmark_vote_count_column_names
            if e[0] != str_predicted_for_actual_votes_counts_column_name and
            e[1] != str_predicted_for_actual_votes_counts_column_name]
        if len(lst_pairs_str_filtered_square_root_of_actual_interactions_benchmark_votes_counts_columns_names) > 0 :
            (tpl_str_filtered_left_square_root_of_actual_interactions_benchmark_votes_counts_columns_names,
             tpl_str_filtered_right_square_root_of_actual_interactions_benchmark_votes_counts_columns_names) = \
                zip(*lst_pairs_str_filtered_square_root_of_actual_interactions_benchmark_votes_counts_columns_names)
            lst_str_filtered_left_square_root_of_actual_interactions_benchmark_votes_counts_columns_names = \
                list(tpl_str_filtered_left_square_root_of_actual_interactions_benchmark_votes_counts_columns_names)
            lst_str_filtered_right_square_root_of_actual_interactions_benchmark_votes_counts_columns_names = \
                list(tpl_str_filtered_right_square_root_of_actual_interactions_benchmark_votes_counts_columns_names)
        else :
            lst_str_filtered_left_square_root_of_actual_interactions_benchmark_votes_counts_columns_names = []
            lst_str_filtered_right_square_root_of_actual_interactions_benchmark_votes_counts_columns_names = []

        lst_pairs_str_filtered_actual_interactions_benchmark_votes_counts_columns_names = [
            e for e in lst_pairs_str_predicting_from_actual_interactions_benchmark_vote_count_column_names
            if e[0] != str_predicted_for_actual_votes_counts_column_name and
            e[1] != str_predicted_for_actual_votes_counts_column_name]
        if len(lst_pairs_str_filtered_actual_interactions_benchmark_votes_counts_columns_names) > 0 :
            (tpl_str_filtered_left_actual_interactions_benchmark_votes_counts_columns_names,
             tpl_str_filtered_right_actual_interactions_benchmark_votes_counts_columns_names) = \
                zip(*lst_pairs_str_filtered_actual_interactions_benchmark_votes_counts_columns_names)
            lst_str_filtered_left_actual_interactions_benchmark_votes_counts_columns_names = \
                list(tpl_str_filtered_left_actual_interactions_benchmark_votes_counts_columns_names)
            lst_str_filtered_right_actual_interactions_benchmark_votes_counts_columns_names = \
                list(tpl_str_filtered_right_actual_interactions_benchmark_votes_counts_columns_names)
        else :
            lst_str_filtered_left_actual_interactions_benchmark_votes_counts_columns_names = []
            lst_str_filtered_right_actual_interactions_benchmark_votes_counts_columns_names = []

        #######################################################################
        # Join "actual" dataframe with "benchmark" data frame, if any.

        if df_actual_vote_counts is not None and df_benchmark_vote_counts is not None :
            X_tmp = df_actual_vote_counts[list(dict.fromkeys(
                lst_str_actual_merge_keys + lst_str_actual_all_vote_count_column_names))].merge(
                    df_benchmark_vote_counts[list(dict.fromkeys(
                        lst_str_benchmark_merge_keys + lst_str_benchmark_all_vote_count_column_names))],
                left_on=lst_str_actual_merge_keys, right_on=lst_str_benchmark_merge_keys, how='inner',)
        elif df_actual_vote_counts is not None and df_benchmark_vote_counts is None :
            X_tmp = df_actual_vote_counts[list(dict.fromkeys(
                lst_str_actual_merge_keys + lst_str_actual_all_vote_count_column_names))]
        elif df_actual_vote_counts is None and df_benchmark_vote_counts is not None :
            X_tmp = df_benchmark_vote_counts[list(dict.fromkeys(
                lst_str_benchmark_merge_keys + lst_str_benchmark_all_vote_count_column_names))]
        else :
            return X

        #######################################################################

        if len(lst_str_filtered_actual_nat_log_of_incremented_votes_counts_columns_names) > 0 :
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = np.log(1.00 + X_tmp[lst_str_filtered_actual_nat_log_of_incremented_votes_counts_columns_names]),
                lst_partial = ["LN_1p00_PLUS__" + t for t in
                               lst_str_filtered_actual_nat_log_of_incremented_votes_counts_columns_names],)
        if len(lst_str_filtered_actual_power_one_quarter_votes_counts_columns_names) > 0 :
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = X_tmp[lst_str_filtered_actual_power_one_quarter_votes_counts_columns_names] ** 0.25,
                lst_partial = ["POWER_0p25__" + t for t in
                               lst_str_filtered_actual_power_one_quarter_votes_counts_columns_names],)
        if len(lst_str_filtered_actual_square_root_votes_counts_columns_names) > 0 :
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = X_tmp[lst_str_filtered_actual_square_root_votes_counts_columns_names] ** 0.50,
                lst_partial = ["POWER_0p50__" + t for t in
                               lst_str_filtered_actual_square_root_votes_counts_columns_names],)
        if len(lst_str_filtered_actual_power_three_quarters_votes_counts_columns_names) > 0 :
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = X_tmp[lst_str_filtered_actual_power_three_quarters_votes_counts_columns_names] ** 0.75,
                lst_partial = ["POWER_0p75__" + t for t in
                               lst_str_filtered_actual_power_three_quarters_votes_counts_columns_names],)
        if len(lst_str_filtered_actual_votes_counts_columns_names) > 0 :
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = X_tmp[lst_str_filtered_actual_votes_counts_columns_names],
                lst_partial = ["POWER_1p00__" + t for t in
                               lst_str_filtered_actual_votes_counts_columns_names],)
        if len(lst_str_filtered_actual_power_one_and_a_half_votes_counts_columns_names) > 0 :
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = X_tmp[lst_str_filtered_actual_power_one_and_a_half_votes_counts_columns_names] ** 1.50,
                lst_partial = ["POWER_1p50__" + t for t in
                               lst_str_filtered_actual_power_one_and_a_half_votes_counts_columns_names],)
        if len(lst_str_filtered_actual_squared_votes_counts_columns_names) > 0 :
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = X_tmp[lst_str_filtered_actual_squared_votes_counts_columns_names] ** 2.00,
                lst_partial = ["POWER_2p00__" + t for t in
                               lst_str_filtered_actual_squared_votes_counts_columns_names],)

        #######################################################################

        if len(lst_str_predicting_from_ln_of_incremented_benchmark_vote_count_column_names) > 0 :
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = np.log(1.00 + X_tmp[lst_str_predicting_from_ln_of_incremented_benchmark_vote_count_column_names]),
                lst_partial = ["LN_1p00_PLUS__" + t for t in
                               lst_str_predicting_from_ln_of_incremented_benchmark_vote_count_column_names],)
        if len(lst_str_predicting_from_power_one_quarter_of_benchmark_vote_count_column_names) > 0 :
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = X_tmp[lst_str_predicting_from_power_one_quarter_of_benchmark_vote_count_column_names] ** 0.25,
                lst_partial = ["POWER_0p25__" + t for t in
                               lst_str_predicting_from_power_one_quarter_of_benchmark_vote_count_column_names],)
        if len(lst_str_predicting_from_square_root_of_benchmark_vote_count_column_names) > 0 :
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = X_tmp[lst_str_predicting_from_square_root_of_benchmark_vote_count_column_names] ** 0.50,
                lst_partial = ["POWER_0p50__" + t for t in
                               lst_str_predicting_from_square_root_of_benchmark_vote_count_column_names],)
        if len(lst_str_predicting_from_power_three_quarters_of_benchmark_vote_count_column_names) > 0 :
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = X_tmp[lst_str_predicting_from_power_three_quarters_of_benchmark_vote_count_column_names] ** 0.75,
                lst_partial = ["POWER_0p75__" + t for t in
                               lst_str_predicting_from_power_three_quarters_of_benchmark_vote_count_column_names],)
        if len(lst_str_predicting_from_benchmark_vote_count_column_names) > 0 :
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = X_tmp[lst_str_predicting_from_benchmark_vote_count_column_names],
                lst_partial = ["POWER_1p00__" + t for t in
                               lst_str_predicting_from_benchmark_vote_count_column_names],)
        if len(lst_str_predicting_from_power_one_and_a_half_of_benchmark_vote_count_column_names) > 0 :
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = X_tmp[lst_str_predicting_from_power_one_and_a_half_of_benchmark_vote_count_column_names] ** 1.50,
                lst_partial = ["POWER_1p50__" + t for t in
                               lst_str_predicting_from_power_one_and_a_half_of_benchmark_vote_count_column_names],)
        if len(lst_str_predicting_from_squared_benchmark_vote_count_column_names) > 0 :
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = X_tmp[lst_str_predicting_from_squared_benchmark_vote_count_column_names] ** 2.00,
                lst_partial = ["POWER_2p00__" + t for t in
                               lst_str_predicting_from_squared_benchmark_vote_count_column_names],)

        #######################################################################

        if len(lst_str_filtered_left_fourth_root_of_actual_interactions_actual_votes_counts_columns_names) > 0 and \
           len(lst_str_filtered_right_fourth_root_of_actual_interactions_actual_votes_counts_columns_names) > 0 and \
           len(lst_str_filtered_left_fourth_root_of_actual_interactions_actual_votes_counts_columns_names) == \
           len(lst_str_filtered_right_fourth_root_of_actual_interactions_actual_votes_counts_columns_names) :
            lst_str_filtered_left_right_fourth_root_of_actual_interactions_actual_votes_counts_columns_names = \
                ["MUL__" + "POWER_0p25__" + t[0] + "__" + "POWER_0p25__" + t[1] for t in list(zip(
                lst_str_filtered_left_fourth_root_of_actual_interactions_actual_votes_counts_columns_names,
                lst_str_filtered_right_fourth_root_of_actual_interactions_actual_votes_counts_columns_names))]
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = (X_tmp[lst_str_filtered_left_fourth_root_of_actual_interactions_actual_votes_counts_columns_names] ** 0.25).mul(
                    (X_tmp[lst_str_filtered_right_fourth_root_of_actual_interactions_actual_votes_counts_columns_names].values ** 0.25), axis=0),
                lst_partial = lst_str_filtered_left_right_fourth_root_of_actual_interactions_actual_votes_counts_columns_names,)

        if len(lst_str_filtered_left_square_root_of_actual_interactions_actual_votes_counts_columns_names) > 0 and \
           len(lst_str_filtered_right_square_root_of_actual_interactions_actual_votes_counts_columns_names) > 0 and \
           len(lst_str_filtered_left_square_root_of_actual_interactions_actual_votes_counts_columns_names) == \
           len(lst_str_filtered_right_square_root_of_actual_interactions_actual_votes_counts_columns_names) :
            lst_str_filtered_left_right_square_root_of_actual_interactions_actual_votes_counts_columns_names = \
                ["MUL__" + "POWER_0p50__" + t[0] + "__" + "POWER_0p50__" + t[1] for t in list(zip(
                lst_str_filtered_left_square_root_of_actual_interactions_actual_votes_counts_columns_names,
                lst_str_filtered_right_square_root_of_actual_interactions_actual_votes_counts_columns_names))]
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = np.sqrt(X_tmp[lst_str_filtered_left_square_root_of_actual_interactions_actual_votes_counts_columns_names]).mul(
                    np.sqrt(X_tmp[lst_str_filtered_right_square_root_of_actual_interactions_actual_votes_counts_columns_names].values), axis=0),
                lst_partial = lst_str_filtered_left_right_square_root_of_actual_interactions_actual_votes_counts_columns_names,)

        if len(lst_str_filtered_left_actual_interactions_actual_votes_counts_columns_names) > 0 and \
           len(lst_str_filtered_right_actual_interactions_actual_votes_counts_columns_names) > 0 and \
           len(lst_str_filtered_left_actual_interactions_actual_votes_counts_columns_names) == \
           len(lst_str_filtered_right_actual_interactions_actual_votes_counts_columns_names) :
            lst_str_filtered_left_right_actual_interactions_actual_votes_counts_columns_names = \
                ["MUL__" + "POWER_1p00__" + t[0] + "__" + "POWER_1p00__" + t[1] for t in list(zip(
                lst_str_filtered_left_actual_interactions_actual_votes_counts_columns_names,
                lst_str_filtered_right_actual_interactions_actual_votes_counts_columns_names))]
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = X_tmp[lst_str_filtered_left_actual_interactions_actual_votes_counts_columns_names].mul(
                    X_tmp[lst_str_filtered_right_actual_interactions_actual_votes_counts_columns_names].values, axis=0),
                lst_partial = lst_str_filtered_left_right_actual_interactions_actual_votes_counts_columns_names,)

        if len(lst_str_filtered_left_fourth_root_of_benchmark_interactions_benchmark_votes_counts_columns_names) > 0 and \
           len(lst_str_filtered_right_fourth_root_of_benchmark_interactions_benchmark_votes_counts_columns_names) > 0 and \
           len(lst_str_filtered_left_fourth_root_of_benchmark_interactions_benchmark_votes_counts_columns_names) == \
           len(lst_str_filtered_right_fourth_root_of_benchmark_interactions_benchmark_votes_counts_columns_names) :
            lst_str_filtered_left_right_fourth_root_of_benchmark_interactions_benchmark_votes_counts_columns_names = \
                ["MUL__" + "POWER_0p25__" + t[0] + "__" + "POWER_0p25__" + t[1] for t in list(zip(
                lst_str_filtered_left_fourth_root_of_benchmark_interactions_benchmark_votes_counts_columns_names,
                lst_str_filtered_right_fourth_root_of_benchmark_interactions_benchmark_votes_counts_columns_names))]
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = (X_tmp[lst_str_filtered_left_fourth_root_of_benchmark_interactions_benchmark_votes_counts_columns_names] ** 0.25).mul(
                    (X_tmp[lst_str_filtered_right_fourth_root_of_benchmark_interactions_benchmark_votes_counts_columns_names].values ** 0.25), axis=0),
                lst_partial = lst_str_filtered_left_right_fourth_root_of_benchmark_interactions_benchmark_votes_counts_columns_names,)

        if len(lst_str_filtered_left_square_root_of_benchmark_interactions_benchmark_votes_counts_columns_names) > 0 and \
           len(lst_str_filtered_right_square_root_of_benchmark_interactions_benchmark_votes_counts_columns_names) > 0 and \
           len(lst_str_filtered_left_square_root_of_benchmark_interactions_benchmark_votes_counts_columns_names) == \
           len(lst_str_filtered_right_square_root_of_benchmark_interactions_benchmark_votes_counts_columns_names) :
            lst_str_filtered_left_right_square_root_of_benchmark_interactions_benchmark_votes_counts_columns_names = \
                ["MUL__" + "POWER_0p50__" + t[0] + "__" + "POWER_0p50__" + t[1] for t in list(zip(
                lst_str_filtered_left_square_root_of_benchmark_interactions_benchmark_votes_counts_columns_names,
                lst_str_filtered_right_square_root_of_benchmark_interactions_benchmark_votes_counts_columns_names))]
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = np.sqrt(X_tmp[lst_str_filtered_left_square_root_of_benchmark_interactions_benchmark_votes_counts_columns_names]).mul(
                    np.sqrt(X_tmp[lst_str_filtered_right_square_root_of_benchmark_interactions_benchmark_votes_counts_columns_names].values), axis=0),
                lst_partial = lst_str_filtered_left_right_square_root_of_benchmark_interactions_benchmark_votes_counts_columns_names,)

        if len(lst_str_filtered_left_benchmark_interactions_benchmark_votes_counts_columns_names) > 0 and \
           len(lst_str_filtered_right_benchmark_interactions_benchmark_votes_counts_columns_names) > 0 and \
           len(lst_str_filtered_left_benchmark_interactions_benchmark_votes_counts_columns_names) == \
           len(lst_str_filtered_right_benchmark_interactions_benchmark_votes_counts_columns_names) :
            lst_str_filtered_left_right_benchmark_interactions_benchmark_votes_counts_columns_names = \
                ["MUL__" + "POWER_1p00__" + t[0] + "__" + "POWER_1p00__" + t[1] for t in list(zip(
                lst_str_filtered_left_benchmark_interactions_benchmark_votes_counts_columns_names,
                lst_str_filtered_right_benchmark_interactions_benchmark_votes_counts_columns_names))]
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = X_tmp[lst_str_filtered_left_benchmark_interactions_benchmark_votes_counts_columns_names].mul(
                    X_tmp[lst_str_filtered_right_benchmark_interactions_benchmark_votes_counts_columns_names].values, axis=0),
                lst_partial = lst_str_filtered_left_right_benchmark_interactions_benchmark_votes_counts_columns_names,)
        
        if len(lst_str_filtered_left_fourth_root_of_actual_interactions_benchmark_votes_counts_columns_names) > 0 and \
           len(lst_str_filtered_right_fourth_root_of_actual_interactions_benchmark_votes_counts_columns_names) > 0 and \
           len(lst_str_filtered_left_fourth_root_of_actual_interactions_benchmark_votes_counts_columns_names) == \
           len(lst_str_filtered_right_fourth_root_of_actual_interactions_benchmark_votes_counts_columns_names) :
            lst_str_filtered_left_right_fourth_root_of_actual_interactions_benchmark_votes_counts_columns_names = \
                ["MUL__" + "POWER_0p25__" + t[0] + "__" + "POWER_0p25__" + t[1] for t in list(zip(
                lst_str_filtered_left_fourth_root_of_actual_interactions_benchmark_votes_counts_columns_names,
                lst_str_filtered_right_fourth_root_of_actual_interactions_benchmark_votes_counts_columns_names))]
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = (X_tmp[lst_str_filtered_left_fourth_root_of_actual_interactions_benchmark_votes_counts_columns_names] ** 0.25).mul(
                    (X_tmp[lst_str_filtered_right_fourth_root_of_actual_interactions_benchmark_votes_counts_columns_names].values ** 0.25), axis=0),
                lst_partial = lst_str_filtered_left_right_fourth_root_of_actual_interactions_benchmark_votes_counts_columns_names,)

        if len(lst_str_filtered_left_square_root_of_actual_interactions_benchmark_votes_counts_columns_names) > 0 and \
           len(lst_str_filtered_right_square_root_of_actual_interactions_benchmark_votes_counts_columns_names) > 0 and \
           len(lst_str_filtered_left_square_root_of_actual_interactions_benchmark_votes_counts_columns_names) == \
           len(lst_str_filtered_right_square_root_of_actual_interactions_benchmark_votes_counts_columns_names) :
            lst_str_filtered_left_right_square_root_of_actual_interactions_benchmark_votes_counts_columns_names = \
                ["MUL__" + "POWER_0p50__" + t[0] + "__" + "POWER_0p50__" + t[1] for t in list(zip(
                lst_str_filtered_left_square_root_of_actual_interactions_benchmark_votes_counts_columns_names,
                lst_str_filtered_right_square_root_of_actual_interactions_benchmark_votes_counts_columns_names))]
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = np.sqrt(X_tmp[lst_str_filtered_left_square_root_of_actual_interactions_benchmark_votes_counts_columns_names]).mul(
                    np.sqrt(X_tmp[lst_str_filtered_right_square_root_of_actual_interactions_benchmark_votes_counts_columns_names].values), axis=0),
                lst_partial = lst_str_filtered_left_right_square_root_of_actual_interactions_benchmark_votes_counts_columns_names,)

        if len(lst_str_filtered_left_actual_interactions_benchmark_votes_counts_columns_names) > 0 and \
           len(lst_str_filtered_right_actual_interactions_benchmark_votes_counts_columns_names) > 0 and \
           len(lst_str_filtered_left_actual_interactions_benchmark_votes_counts_columns_names) == \
           len(lst_str_filtered_right_actual_interactions_benchmark_votes_counts_columns_names) :
            lst_str_filtered_left_right_actual_interactions_benchmark_votes_counts_columns_names = \
                ["MUL__" + "POWER_1p00__" + t[0] + "__" + "POWER_1p00__" + t[1] for t in list(zip(
                lst_str_filtered_left_actual_interactions_benchmark_votes_counts_columns_names,
                lst_str_filtered_right_actual_interactions_benchmark_votes_counts_columns_names))]
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = X_tmp[lst_str_filtered_left_actual_interactions_benchmark_votes_counts_columns_names].mul(
                    X_tmp[lst_str_filtered_right_actual_interactions_benchmark_votes_counts_columns_names].values, axis=0),
                lst_partial = lst_str_filtered_left_right_actual_interactions_benchmark_votes_counts_columns_names,)

        #######################################################################
        # Exclude str_predicted_for_actual_votes_counts_column_name:

        if bool_predicting_from_ln_of_incremented_actual_tally :
            if bool_drop_predicted_variable_during_computation_of_predicting_tally and \
               len(lst_str_actual_all_vote_count_column_names) > 0 :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = np.log(1.00 + X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1) - X_tmp[
                        str_predicted_for_actual_votes_counts_column_name]),
                    lst_partial = ["ACTUAL_LN_1p00_PLUS__TALLY__EXCLUDING__" +
                                   str_predicted_for_actual_votes_counts_column_name],)
            else :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = np.log(1.00 + X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1)),
                    lst_partial = ["ACTUAL_LN_1p00_PLUS__TALLY"],)
        if bool_predicting_from_power_one_quarter_of_actual_tally :
            if bool_drop_predicted_variable_during_computation_of_predicting_tally and \
               len(lst_str_actual_all_vote_count_column_names) > 0 :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = (X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1) - X_tmp[
                        str_predicted_for_actual_votes_counts_column_name]) ** 0.25,
                    lst_partial = ["ACTUAL_POWER_0p25_TALLY__EXCLUDING__" +
                                   str_predicted_for_actual_votes_counts_column_name],)
            else :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = (X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1)) ** 0.25,
                    lst_partial = ["ACTUAL_POWER_0p25_TALLY"],)
        if bool_predicting_from_square_root_of_actual_tally :
            if bool_drop_predicted_variable_during_computation_of_predicting_tally and \
               len(lst_str_actual_all_vote_count_column_names) > 0 :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = (X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1) - X_tmp[
                        str_predicted_for_actual_votes_counts_column_name]) ** 0.50,
                    lst_partial = ["ACTUAL_POWER_0p50_TALLY__EXCLUDING__" +
                                   str_predicted_for_actual_votes_counts_column_name],)
            else :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = (X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1)) ** 0.50,
                    lst_partial = ["ACTUAL_POWER_0p50_TALLY"],)
        if bool_predicting_from_power_three_quarters_of_actual_tally :
            if bool_drop_predicted_variable_during_computation_of_predicting_tally and \
               len(lst_str_actual_all_vote_count_column_names) > 0 :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = (X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1) - X_tmp[
                        str_predicted_for_actual_votes_counts_column_name]) ** 0.75,
                    lst_partial = ["ACTUAL_POWER_0p75_TALLY__EXCLUDING__" +
                                   str_predicted_for_actual_votes_counts_column_name],)
            else :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = (X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1)) ** 0.75,
                    lst_partial = ["ACTUAL_POWER_0p75_TALLY"],)
        if bool_predicting_from_actual_tally :
            if bool_drop_predicted_variable_during_computation_of_predicting_tally and \
               len(lst_str_actual_all_vote_count_column_names) > 0 :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1) - X_tmp[
                        str_predicted_for_actual_votes_counts_column_name],
                    lst_partial = ["ACTUAL_POWER_1p00_TALLY__EXCLUDING__" +
                                   str_predicted_for_actual_votes_counts_column_name],)
            else :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1),
                    lst_partial = ["ACTUAL_POWER_1p00_TALLY"],)
        if bool_predicting_from_power_one_and_a_half_of_actual_tally :
            if bool_drop_predicted_variable_during_computation_of_predicting_tally and \
               len(lst_str_actual_all_vote_count_column_names) > 0 :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = (X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1) - X_tmp[
                        str_predicted_for_actual_votes_counts_column_name]) ** 1.50,
                    lst_partial = ["ACTUAL_POWER_1p50_TALLY__EXCLUDING__" +
                                   str_predicted_for_actual_votes_counts_column_name],)
            else :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = (X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1)) ** 1.50,
                    lst_partial = ["ACTUAL_POWER_1p50_TALLY"],)
        if bool_predicting_from_squared_actual_tally :
            if bool_drop_predicted_variable_during_computation_of_predicting_tally and \
               len(lst_str_actual_all_vote_count_column_names) > 0 :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = (X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1) - X_tmp[
                        str_predicted_for_actual_votes_counts_column_name]) ** 2.00,
                    lst_partial = ["ACTUAL_POWER_2p00_TALLY__EXCLUDING__" +
                                   str_predicted_for_actual_votes_counts_column_name],)
            else :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = (X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1)) ** 2.00,
                    lst_partial = ["ACTUAL_POWER_2p00_TALLY"],)

        if bool_predicting_from_ln_of_incremented_benchmark_tally :
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = np.log(1.00 + X_tmp[lst_str_benchmark_all_vote_count_column_names].sum(axis=1)),
                lst_partial = ["BENCHMARK_LN_1p00_PLUS__TALLY"],)
        if bool_predicting_from_power_one_quarter_of_benchmark_tally :
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = (X_tmp[lst_str_benchmark_all_vote_count_column_names].sum(axis=1)) ** 0.25,
                lst_partial = ["BENCHMARK_POWER_0p25_TALLY"],)
        if bool_predicting_from_square_root_of_benchmark_tally :
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = (X_tmp[lst_str_benchmark_all_vote_count_column_names].sum(axis=1)) ** 0.50,
                lst_partial = ["BENCHMARK_POWER_0p50_TALLY"],)
        if bool_predicting_from_power_three_quarters_of_benchmark_tally :
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = (X_tmp[lst_str_benchmark_all_vote_count_column_names].sum(axis=1)) ** 0.75,
                lst_partial = ["BENCHMARK_POWER_0p75_TALLY"],)
        if bool_predicting_from_benchmark_tally :
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = X_tmp[lst_str_benchmark_all_vote_count_column_names].sum(axis=1),
                lst_partial = ["BENCHMARK_POWER_1p00_TALLY"],)
        if bool_predicting_from_power_one_and_a_half_of_benchmark_tally :
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = (X_tmp[lst_str_benchmark_all_vote_count_column_names].sum(axis=1)) ** 1.50,
                lst_partial = ["BENCHMARK_POWER_1p50_TALLY"],)
        if bool_predicting_from_squared_benchmark_tally :
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = (X_tmp[lst_str_benchmark_all_vote_count_column_names].sum(axis=1)) ** 2.00,
                lst_partial = ["BENCHMARK_POWER_2p00_TALLY"],)

        #######################################################################
        # Exclude str_predicted_for_actual_votes_counts_column_name:

        if bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally :
            if bool_drop_predicted_variable_during_computation_of_predicting_tally and \
               len(lst_str_actual_all_vote_count_column_names) > 0 :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = ((X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1) - X_tmp[
                        str_predicted_for_actual_votes_counts_column_name]) ** 0.25).mul(
                        ((X_tmp[lst_str_benchmark_all_vote_count_column_names].sum(axis=1)) ** 0.25), axis=0),
                    lst_partial = ["MUL__POWER_0p25__ACTUAL_TALLY__EXCLUDING__" +
                                   str_predicted_for_actual_votes_counts_column_name +
                                   "__POWER_0p25__BENCHMARK_TALLY"],)
            else :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = ((X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1)) ** 0.25).mul(
                        ((X_tmp[lst_str_benchmark_all_vote_count_column_names].sum(axis=1)) ** 0.25), axis=0),
                    lst_partial = ["MUL__POWER_0p25__ACTUAL_TALLY__POWER_0p25__BENCHMARK_TALLY"],)
        if bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally :
            if bool_drop_predicted_variable_during_computation_of_predicting_tally and \
               len(lst_str_actual_all_vote_count_column_names) > 0 :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = np.sqrt(X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1) - X_tmp[
                        str_predicted_for_actual_votes_counts_column_name]).mul(
                        np.sqrt(X_tmp[lst_str_benchmark_all_vote_count_column_names].sum(axis=1)), axis=0),
                    lst_partial = ["MUL__POWER_0p50__ACTUAL_TALLY__EXCLUDING__" +
                                   str_predicted_for_actual_votes_counts_column_name +
                                   "__POWER_0p50__BENCHMARK_TALLY"],)
            else :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = np.sqrt(X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1)).mul(
                        np.sqrt(X_tmp[lst_str_benchmark_all_vote_count_column_names].sum(axis=1)), axis=0),
                    lst_partial = ["MUL__POWER_0p50__ACTUAL_TALLY__POWER_0p50__BENCHMARK_TALLY"],)
        if bool_predicting_from_actual_tally_interaction_benchmark_tally :
            if bool_drop_predicted_variable_during_computation_of_predicting_tally and \
               len(lst_str_actual_all_vote_count_column_names) > 0 :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = (X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1) - X_tmp[
                        str_predicted_for_actual_votes_counts_column_name]).mul(
                        X_tmp[lst_str_benchmark_all_vote_count_column_names].sum(axis=1), axis=0),
                    lst_partial = ["MUL__POWER_1p00__ACTUAL_TALLY__EXCLUDING__" +
                                   str_predicted_for_actual_votes_counts_column_name +
                                   "__POWER_1p00__BENCHMARK_TALLY"],)
            else :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = (X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1)).mul(
                        X_tmp[lst_str_benchmark_all_vote_count_column_names].sum(axis=1), axis=0),
                    lst_partial = ["MUL__POWER_1p00__ACTUAL_TALLY__POWER_1p00__BENCHMARK_TALLY"],)

        if (
            bool_predicting_from_ln_of_incremented_actual_tally_interaction_county_indicator or
            bool_predicting_from_power_one_quarter_of_actual_tally_interaction_county_indicator or
            bool_predicting_from_square_root_of_actual_tally_interaction_county_indicator or
            bool_predicting_from_power_three_quarters_of_actual_tally_interaction_county_indicator or
            bool_predicting_from_actual_tally_interaction_county_indicator or
            bool_predicting_from_power_one_and_a_half_of_actual_tally_interaction_county_indicator or
            bool_predicting_from_squared_actual_tally_interaction_county_indicator or
            #
            bool_predicting_from_ln_of_incremented_benchmark_tally_interaction_county_indicator or
            bool_predicting_from_power_one_quarter_of_benchmark_tally_interaction_county_indicator or
            bool_predicting_from_square_root_of_benchmark_tally_interaction_county_indicator or
            bool_predicting_from_power_three_quarters_of_benchmark_tally_interaction_county_indicator or
            bool_predicting_from_benchmark_tally_interaction_county_indicator or
            bool_predicting_from_power_one_and_a_half_of_benchmark_tally_interaction_county_indicator or
            bool_predicting_from_squared_benchmark_tally_interaction_county_indicator or
            #
            bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator or
            bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator or
            bool_predicting_from_actual_tally_interaction_benchmark_tally_interaction_county_indicator
           ) :
            X_dummies = pd.get_dummies(X_tmp[STR_COUNTY_NAME_COLUMN_NAME])
        else :
            X_dummies = None

        #######################################################################
        # Exclude str_predicted_for_actual_votes_counts_column_name:
        if bool_predicting_from_ln_of_incremented_actual_tally_interaction_county_indicator :
            if bool_drop_predicted_variable_during_computation_of_predicting_tally and \
               len(lst_str_actual_all_vote_count_column_names) > 0 :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = X_dummies.mul(np.log(1.00 + X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1) - X_tmp[
                        str_predicted_for_actual_votes_counts_column_name]), axis=0),
                    lst_partial = ["MUL__ONE_HOT_INDICATOR_" + str_county_name + "__LN_1p00_PLUS__ACTUAL_TALLY__EXCLUDING__" +
                                   str_predicted_for_actual_votes_counts_column_name for str_county_name in
                                   list(X_dummies.columns)],)
            else :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = X_dummies.mul(np.log(1.00 + X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1)), axis=0),
                    lst_partial = ["MUL__ONE_HOT_INDICATOR_" + str_county_name + "__LN_1p00_PLUS__ACTUAL_TALLY"
                                   for str_county_name in list(X_dummies.columns)],)
        if bool_predicting_from_power_one_quarter_of_actual_tally_interaction_county_indicator :
            if bool_drop_predicted_variable_during_computation_of_predicting_tally and \
               len(lst_str_actual_all_vote_count_column_names) > 0 :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = X_dummies.mul((X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1) - X_tmp[
                        str_predicted_for_actual_votes_counts_column_name]) ** 0.25, axis=0),
                    lst_partial = ["MUL__ONE_HOT_INDICATOR_" + str_county_name + "__POWER_0p25__ACTUAL_TALLY__EXCLUDING__" +
                                   str_predicted_for_actual_votes_counts_column_name for str_county_name in
                                   list(X_dummies.columns)],)
            else :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = X_dummies.mul((X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1)) ** 0.25, axis=0),
                    lst_partial = ["MUL__ONE_HOT_INDICATOR_" + str_county_name + "__POWER_0p25__ACTUAL_TALLY"
                                   for str_county_name in list(X_dummies.columns)],)
        if bool_predicting_from_square_root_of_actual_tally_interaction_county_indicator :
            if bool_drop_predicted_variable_during_computation_of_predicting_tally and \
               len(lst_str_actual_all_vote_count_column_names) > 0 :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = X_dummies.mul((X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1) - X_tmp[
                        str_predicted_for_actual_votes_counts_column_name]) ** 0.50, axis=0),
                    lst_partial = ["MUL__ONE_HOT_INDICATOR_" + str_county_name + "__POWER_0p50__ACTUAL_TALLY__EXCLUDING__" +
                                   str_predicted_for_actual_votes_counts_column_name for str_county_name in
                                   list(X_dummies.columns)],)
            else :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = X_dummies.mul((X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1)) ** 0.50, axis=0),
                    lst_partial = ["MUL__ONE_HOT_INDICATOR_" + str_county_name + "__POWER_0p50__ACTUAL_TALLY"
                                   for str_county_name in list(X_dummies.columns)],)
        if bool_predicting_from_power_three_quarters_of_actual_tally_interaction_county_indicator :
            if bool_drop_predicted_variable_during_computation_of_predicting_tally and \
               len(lst_str_actual_all_vote_count_column_names) > 0 :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = X_dummies.mul((X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1) - X_tmp[
                        str_predicted_for_actual_votes_counts_column_name]) ** 0.75, axis=0),
                    lst_partial = ["MUL__ONE_HOT_INDICATOR_" + str_county_name + "__POWER_0p75__ACTUAL_TALLY__EXCLUDING__" +
                                   str_predicted_for_actual_votes_counts_column_name for str_county_name in
                                   list(X_dummies.columns)],)
            else :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = X_dummies.mul((X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1)) ** 0.75, axis=0),
                    lst_partial = ["MUL__ONE_HOT_INDICATOR_" + str_county_name + "__POWER_0p75__ACTUAL_TALLY"
                                   for str_county_name in list(X_dummies.columns)],)
        if bool_predicting_from_actual_tally_interaction_county_indicator :
            if bool_drop_predicted_variable_during_computation_of_predicting_tally and \
               len(lst_str_actual_all_vote_count_column_names) > 0 :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = X_dummies.mul((X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1) - X_tmp[
                        str_predicted_for_actual_votes_counts_column_name]), axis=0),
                    lst_partial = ["MUL__ONE_HOT_INDICATOR_" + str_county_name + "__POWER_1p00__ACTUAL_TALLY__EXCLUDING__" +
                               str_predicted_for_actual_votes_counts_column_name for str_county_name in
                               list(X_dummies.columns)],)
            else :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = X_dummies.mul((X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1)), axis=0),
                    lst_partial = ["MUL__ONE_HOT_INDICATOR_" + str_county_name + "__POWER_1p00__ACTUAL_TALLY"
                                   for str_county_name in list(X_dummies.columns)],)
        if bool_predicting_from_power_one_and_a_half_of_actual_tally_interaction_county_indicator :
            if bool_drop_predicted_variable_during_computation_of_predicting_tally and \
               len(lst_str_actual_all_vote_count_column_names) > 0 :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = X_dummies.mul((X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1) - X_tmp[
                        str_predicted_for_actual_votes_counts_column_name]) ** 1.50, axis=0),
                    lst_partial = ["MUL__ONE_HOT_INDICATOR_" + str_county_name + "__POWER_1p50__ACTUAL_TALLY__EXCLUDING__" +
                                   str_predicted_for_actual_votes_counts_column_name for str_county_name in
                                   list(X_dummies.columns)],)
            else :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = X_dummies.mul((X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1)) ** 1.50, axis=0),
                    lst_partial = ["MUL__ONE_HOT_INDICATOR_" + str_county_name + "__POWER_1p50__ACTUAL_TALLY"
                                   for str_county_name in list(X_dummies.columns)],)
        if bool_predicting_from_squared_actual_tally_interaction_county_indicator :
            if bool_drop_predicted_variable_during_computation_of_predicting_tally and \
               len(lst_str_actual_all_vote_count_column_names) > 0 :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = X_dummies.mul((X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1) - X_tmp[
                        str_predicted_for_actual_votes_counts_column_name]) ** 2.00, axis=0),
                    lst_partial = ["MUL__ONE_HOT_INDICATOR_" + str_county_name + "__POWER_2p00__ACTUAL_TALLY__EXCLUDING__" +
                                   str_predicted_for_actual_votes_counts_column_name for str_county_name in
                                   list(X_dummies.columns)],)
            else :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = X_dummies.mul((X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1)) ** 2.00, axis=0),
                    lst_partial = ["MUL__ONE_HOT_INDICATOR_" + str_county_name + "__POWER_2p00__ACTUAL_TALLY"
                                   for str_county_name in list(X_dummies.columns)],)

        if bool_predicting_from_ln_of_incremented_benchmark_tally_interaction_county_indicator :
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = X_dummies.mul(np.log(1.00 + X_tmp[lst_str_benchmark_all_vote_count_column_names].sum(axis=1)), axis=0),
                lst_partial = ["MUL__ONE_HOT_INDICATOR_" + str_county_name + "__LN_1p00_PLUS__BENCHMARK_TALLY"
                               for str_county_name in list(X_dummies.columns)],)
        if bool_predicting_from_power_one_quarter_of_benchmark_tally_interaction_county_indicator :
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = X_dummies.mul((X_tmp[lst_str_benchmark_all_vote_count_column_names].sum(axis=1)) ** 0.25, axis=0),
                lst_partial = ["MUL__ONE_HOT_INDICATOR_" + str_county_name + "__POWER_0p25__BENCHMARK_TALLY"
                               for str_county_name in list(X_dummies.columns)],)
        if bool_predicting_from_square_root_of_benchmark_tally_interaction_county_indicator :
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = X_dummies.mul((X_tmp[lst_str_benchmark_all_vote_count_column_names].sum(axis=1)) ** 0.50, axis=0),
                lst_partial = ["MUL__ONE_HOT_INDICATOR_" + str_county_name + "__POWER_0p50__BENCHMARK_TALLY"
                               for str_county_name in list(X_dummies.columns)],)
        if bool_predicting_from_power_three_quarters_of_benchmark_tally_interaction_county_indicator :
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = X_dummies.mul((X_tmp[lst_str_benchmark_all_vote_count_column_names].sum(axis=1)) ** 0.75, axis=0),
                lst_partial = ["MUL__ONE_HOT_INDICATOR_" + str_county_name + "__POWER_0p75__BENCHMARK_TALLY"
                               for str_county_name in list(X_dummies.columns)],)
        if bool_predicting_from_benchmark_tally_interaction_county_indicator :
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = X_dummies.mul(X_tmp[lst_str_benchmark_all_vote_count_column_names].sum(axis=1), axis=0),
                lst_partial = ["MUL__ONE_HOT_INDICATOR_" + str_county_name + "__POWER_1p00__BENCHMARK_TALLY"
                               for str_county_name in list(X_dummies.columns)],)
        if bool_predicting_from_power_one_and_a_half_of_benchmark_tally_interaction_county_indicator :
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = X_dummies.mul((X_tmp[lst_str_benchmark_all_vote_count_column_names].sum(axis=1)) ** 1.50, axis=0),
                lst_partial = ["MUL__ONE_HOT_INDICATOR_" + str_county_name + "__POWER_1p50__BENCHMARK_TALLY"
                               for str_county_name in list(X_dummies.columns)],)
        if bool_predicting_from_squared_benchmark_tally_interaction_county_indicator :
            (X, lst_str_X_columns_names) = add_or_append_predictors(
                X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                X_partial = X_dummies.mul((X_tmp[lst_str_benchmark_all_vote_count_column_names].sum(axis=1)) ** 2.00, axis=0),
                lst_partial = ["MUL__ONE_HOT_INDICATOR_" + str_county_name + "__POWER_2p00__BENCHMARK_TALLY"
                               for str_county_name in list(X_dummies.columns)],)

        #######################################################################
        # Exclude str_predicted_for_actual_votes_counts_column_name:
        if bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator :
            if bool_drop_predicted_variable_during_computation_of_predicting_tally and \
               len(lst_str_actual_all_vote_count_column_names) > 0 :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = X_dummies.mul(((X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1) - X_tmp[
                        str_predicted_for_actual_votes_counts_column_name]) ** 0.25).mul(
                        ((X_tmp[lst_str_benchmark_all_vote_count_column_names].sum(axis=1)) ** 0.25), axis=0), axis=0),
                    lst_partial = ["MUL__ONE_HOT_INDICATOR_" + str_county_name + "__POWER_0p25__ACTUAL_TALLY__EXCLUDING__" +
                                   str_predicted_for_actual_votes_counts_column_name + "__POWER_0p25__BENCHMARK_TALLY"
                                   for str_county_name in list(X_dummies.columns)],)
            else :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = X_dummies.mul(((X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1)) ** 0.25).mul(
                        ((X_tmp[lst_str_benchmark_all_vote_count_column_names].sum(axis=1)) ** 0.25), axis=0), axis=0),
                    lst_partial = ["MUL__ONE_HOT_INDICATOR_" + str_county_name +
                                   "__POWER_0p25__ACTUAL_TALLY__POWER_0p25__BENCHMARK_TALLY"
                                   for str_county_name in list(X_dummies.columns)],)
        if bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator :
            if bool_drop_predicted_variable_during_computation_of_predicting_tally and \
               len(lst_str_actual_all_vote_count_column_names) > 0 :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = X_dummies.mul(np.sqrt(X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1) - X_tmp[
                        str_predicted_for_actual_votes_counts_column_name]).mul(
                        np.sqrt(X_tmp[lst_str_benchmark_all_vote_count_column_names].sum(axis=1)), axis=0), axis=0),
                    lst_partial = ["MUL__ONE_HOT_INDICATOR_" + str_county_name + "__POWER_0p50__ACTUAL_TALLY__EXCLUDING__" +
                                   str_predicted_for_actual_votes_counts_column_name + "__POWER_0p50__BENCHMARK_TALLY"
                                   for str_county_name in list(X_dummies.columns)],)
            else :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = X_dummies.mul(np.sqrt(X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1)).mul(
                        np.sqrt(X_tmp[lst_str_benchmark_all_vote_count_column_names].sum(axis=1)), axis=0), axis=0),
                    lst_partial = ["MUL__ONE_HOT_INDICATOR_" + str_county_name +
                                   "__POWER_0p50__ACTUAL_TALLY__POWER_0p50__BENCHMARK_TALLY"
                                   for str_county_name in list(X_dummies.columns)],)
        if bool_predicting_from_actual_tally_interaction_benchmark_tally_interaction_county_indicator :
            if bool_drop_predicted_variable_during_computation_of_predicting_tally and \
               len(lst_str_actual_all_vote_count_column_names) > 0 :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = X_dummies.mul((X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1) - X_tmp[
                        str_predicted_for_actual_votes_counts_column_name]).mul(
                        X_tmp[lst_str_benchmark_all_vote_count_column_names].sum(axis=1), axis=0), axis=0),
                    lst_partial = ["MUL__ONE_HOT_INDICATOR_" + str_county_name + "__POWER_1p00__ACTUAL_TALLY__EXCLUDING__" +
                                   str_predicted_for_actual_votes_counts_column_name + "__POWER_1p00__BENCHMARK_TALLY"
                                   for str_county_name in list(X_dummies.columns)],)
            else :
                (X, lst_str_X_columns_names) = add_or_append_predictors(
                    X = X, lst_str_X_columns_names = lst_str_X_columns_names,
                    X_partial = X_dummies.mul((X_tmp[lst_str_actual_all_vote_count_column_names].sum(axis=1)).mul(
                        X_tmp[lst_str_benchmark_all_vote_count_column_names].sum(axis=1), axis=0), axis=0),
                    lst_partial = ["MUL__ONE_HOT_INDICATOR_" + str_county_name +
                                   "__POWER_1p00__ACTUAL_TALLY__POWER_1p00__BENCHMARK_TALLY"
                                   for str_county_name in list(X_dummies.columns)],)

        X.columns = lst_str_X_columns_names
    return X

###############################################################################

def estimate_linear_model_with_summary(
        y,
        X=None,
        flt_lasso_alpha_l1_regularization_strength_term = None, # [0.; 1.]. # None by default (1.)
        int_lasso_maximum_number_of_iterations = 1_000_000,
        int_lasso_optimization_tolerance = 0.00001,
        flt_lasso_cv_length_of_alphas_regularization_path = 1., # (0.; +1.] # 1 by default (0.001); alpha_min / alpha_max
        int_lasso_cv_num_candidate_alphas_on_regularization_path = 0, # [0; +Inf] # 0 by default (100)
        int_lasso_cv_maximum_number_of_iterations = 1_000_000,
        int_lasso_cv_optimization_tolerance = 0.00001,
        int_lasso_cv_number_of_folds_in_cross_validation = 10,
        bool_estimate_model_parameters_diagnostics = False,
        ) :
    '''<function description>
    
    <calling functions>
    <called functions>

    https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
    https://stats.stackexchange.com/questions/419393/how-to-find-t-value-without-data

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
    if X is not None :
        if not y.equals(X) :

            if (flt_lasso_alpha_l1_regularization_strength_term is None and
                flt_lasso_cv_length_of_alphas_regularization_path == 1. and
                int_lasso_cv_num_candidate_alphas_on_regularization_path == 0) or \
               (flt_lasso_alpha_l1_regularization_strength_term is not None and
                flt_lasso_alpha_l1_regularization_strength_term == 0. and
                flt_lasso_cv_length_of_alphas_regularization_path == 1. and
                int_lasso_cv_num_candidate_alphas_on_regularization_path == 0) :
                bool_is_plot_for_parametric_lasso_model = False
            else :
                bool_is_plot_for_parametric_lasso_model = True

            if not bool_is_plot_for_parametric_lasso_model :
                model = LinearRegression(
                    fit_intercept = BOOL_FIT_REGRESSION_INTERCEPT,)
                flt_lasso_alpha = 0.
            else :
                if flt_lasso_cv_length_of_alphas_regularization_path < 1. and \
                   int_lasso_cv_num_candidate_alphas_on_regularization_path > 0 :
                    # eps: Length of the path. eps=1e-3 means that
                    #       alpha_min / alpha_max = 1e-3. float, default = 1e-3
                    # n_alphas = Number of alphas along the regularization path.
                    #       int, default=100.
                    lasso_cv_result = LassoCV(
                        eps = flt_lasso_cv_length_of_alphas_regularization_path,
                        n_alphas = int_lasso_cv_num_candidate_alphas_on_regularization_path,
                        fit_intercept = BOOL_FIT_REGRESSION_INTERCEPT,
                        max_iter = int_lasso_cv_maximum_number_of_iterations,
                        tol = int_lasso_cv_optimization_tolerance,
                        cv = int_lasso_cv_number_of_folds_in_cross_validation,
                        n_jobs = None,
                        random_state = None,
                        selection = 'cyclic', # 'random'
                        ).fit(X=X.astype(np.float64),
                              y=y.astype(np.float64).values.ravel())
                    flt_lasso_alpha = lasso_cv_result.alpha_
                else :
                    flt_lasso_alpha = flt_lasso_alpha_l1_regularization_strength_term
                # When alpha = 0., the objective is equivalent to ordinary least squares,
                # solved by the LinearRegression object. For numerical reasons, using
                # alpha = 0 with the Lasso object is not advised. Instead, you should
                # use the LinearRegression object.
                model = Lasso(
                    alpha = flt_lasso_alpha,
                    fit_intercept = BOOL_FIT_REGRESSION_INTERCEPT,
                    max_iter = int_lasso_maximum_number_of_iterations,
                    tol = int_lasso_optimization_tolerance,
                    random_state = None,
                    selection = 'cyclic', # 'random'
                    )
            model.fit(X=X.astype(np.float64), y=y.astype(np.float64))
            y_hat = model.predict(X=X).ravel()
            flt_intercept = model.intercept_[0] \
                if BOOL_FIT_REGRESSION_INTERCEPT else model.intercept_
            coef = np.append(flt_intercept, model.coef_)

            if bool_estimate_model_parameters_diagnostics :
                bool_use_dataframe = False
                if bool_use_dataframe :
                    onesX = pd.DataFrame(
                        {"Constant" : np.ones(len(X))}).join(pd.DataFrame(X))
                    df = len(onesX) - len(onesX.columns)
                else :
                    onesX = np.append(np.ones((len(X), 1)), X, axis=1)
                    df = len(onesX) - len(onesX[0])
                mse = sum((np.array(y).ravel() - y_hat) ** 2.) / df
                onesXsq = np.dot(onesX.T, onesX)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    det_onesXsq = np.linalg.det(onesXsq)
                if not np.isnan(det_onesXsq) and \
                   not np.isinf(det_onesXsq) and \
                   det_onesXsq != 0. :
                    std_err = np.sqrt(mse * (np.linalg.inv(onesXsq).diagonal()))
                    t_value = coef / std_err
                    p_value = [2. * (1. - stats.t.cdf(np.abs(t), df))
                               for t in t_value]

                    alpha = .05
                    t_stat_25bp = stats.t.ppf(1. - alpha / 2., df)
                    coef_low_25bp = coef - std_err * t_stat_25bp
                    coef_upp_25bp = coef + std_err * t_stat_25bp

                    alpha = .01
                    t_stat_5bp = stats.t.ppf(1. - alpha / 2., df)
                    coef_low_5bp = coef - std_err * t_stat_5bp
                    coef_upp_5bp = coef + std_err * t_stat_5bp

                    # Value of flt_lasso_alpha is not used for the above computations.
                    coef = np.append(flt_lasso_alpha, coef)
                    std_err = np.append(0., std_err)
                    t_value = np.append(0., t_value)
                    p_value = np.append(0., p_value)
                    coef_low_25bp = np.append(flt_lasso_alpha, coef_low_25bp)
                    coef_upp_25bp = np.append(flt_lasso_alpha, coef_upp_25bp)
                    coef_low_5bp = np.append(flt_lasso_alpha, coef_low_5bp)
                    coef_upp_5bp = np.append(flt_lasso_alpha, coef_upp_5bp)
                else :
                    coef = np.append(flt_lasso_alpha, coef)
                    std_err = np.array([0.] * len(coef))
                    t_value = np.array([0.] * len(coef))
                    p_value = np.array([0.] * len(coef))
                    coef_low_25bp = np.array([0.] * len(coef))
                    coef_upp_25bp = np.array([0.] * len(coef))
                    coef_low_5bp = np.array([0.] * len(coef))
                    coef_upp_5bp = np.array([0.] * len(coef))

            else :
                coef = np.append(flt_lasso_alpha, coef)
                std_err = np.array([0.] * len(coef))
                t_value = np.array([0.] * len(coef))
                p_value = np.array([0.] * len(coef))
                coef_low_25bp = np.array(coef)
                coef_upp_25bp = np.array(coef)
                coef_low_5bp = np.array(coef)
                coef_upp_5bp = np.array(coef)
        else : # if not y.equals(X) :
            (y_hat, coef, std_err, t_value, p_value, coef_low_25bp,
             coef_upp_25bp, coef_low_5bp, coef_upp_5bp) = \
                (y.copy(), np.array([[0.], [1.]]), np.array([[0.], [0.]]),
                 np.array([[0.], [0.]]), np.array([[0.], [0.]]),
                 np.array([[0.], [1.]]), np.array([[0.], [1.]]),
                 np.array([[0.], [1.]]), np.array([[0.], [1.]]),)
    else : # if X is not None
        flt_y_mean = y.mean()
        (y_hat, coef, std_err, t_value, p_value, coef_low_25bp,
         coef_upp_25bp, coef_low_5bp, coef_upp_5bp) = \
            (flt_y_mean*len(y), np.array([flt_y_mean]*1), np.array([0.]*1),
             np.array([0.]*1), np.array([0.]*1),
             np.array([flt_y_mean]*1), np.array([flt_y_mean]*1),
             np.array([flt_y_mean]*1), np.array([flt_y_mean]*1),)

    return (y_hat, coef, std_err, t_value, p_value, coef_low_25bp,
            coef_upp_25bp, coef_low_5bp, coef_upp_5bp)

###############################################################################

def estimate_linear_model_with_summary_in_dataframe(
        str_y_name,
        y,
        X = None,
        flt_lasso_alpha_l1_regularization_strength_term = None, # [0.; 1.]. # None by default (1.)
        int_lasso_maximum_number_of_iterations = 1_000_000,
        int_lasso_optimization_tolerance = 0.00001,
        flt_lasso_cv_length_of_alphas_regularization_path = 1., # (0.; +1.] # 1 by default (0.001); alpha_min / alpha_max
        int_lasso_cv_num_candidate_alphas_on_regularization_path = 0, # [0; +Inf] # 0 by default (100)
        int_lasso_cv_maximum_number_of_iterations = 1_000_000,
        int_lasso_cv_optimization_tolerance = 0.00001,
        int_lasso_cv_number_of_folds_in_cross_validation = 10,
        df_predicted_vote_counts_out = None,
        bool_estimate_model_parameters_diagnostics = False,
        ) :
    '''<function description>
    
    <calling functions>
    <called functions>

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
    (y_hat, coef,
     std_err, t_value, p_value,
     coef_low_25bp, coef_upp_25bp, coef_low_5bp, coef_upp_5bp) = \
        estimate_linear_model_with_summary(
            y = y,
            X = X,
            flt_lasso_alpha_l1_regularization_strength_term =
                flt_lasso_alpha_l1_regularization_strength_term,
            int_lasso_maximum_number_of_iterations =
                int_lasso_maximum_number_of_iterations,
            int_lasso_optimization_tolerance =
                int_lasso_optimization_tolerance,
            flt_lasso_cv_length_of_alphas_regularization_path =
                flt_lasso_cv_length_of_alphas_regularization_path,
            int_lasso_cv_num_candidate_alphas_on_regularization_path =
                int_lasso_cv_num_candidate_alphas_on_regularization_path,
            int_lasso_cv_maximum_number_of_iterations =
                int_lasso_cv_maximum_number_of_iterations,
            int_lasso_cv_optimization_tolerance =
                int_lasso_cv_optimization_tolerance,
            int_lasso_cv_number_of_folds_in_cross_validation =
                int_lasso_cv_number_of_folds_in_cross_validation,
            bool_estimate_model_parameters_diagnostics =
                bool_estimate_model_parameters_diagnostics,
            )
    if X is not None :
        lst_X_columns = list(X.columns)
    else :
        lst_X_columns = []
    if bool_estimate_model_parameters_diagnostics :
        lst_indep_var = [str_y_name] * (2 + len(lst_X_columns))
        lst_dep_var = ["LASSO_ALPHA", "INTERCEPT"] + lst_X_columns
        df_ols_coeffs_curr = pd.concat([
            pd.DataFrame(lst_indep_var),
            pd.DataFrame(lst_dep_var),
            pd.DataFrame(coef),
            pd.DataFrame(std_err),
            pd.DataFrame(t_value),
            pd.DataFrame(p_value),
            pd.DataFrame(coef_low_25bp), # CI = 0.95 = (1 - 2 * 0.025)
            pd.DataFrame(coef_upp_25bp), # CI = 0.95 = (1 - 2 * 0.025)
            pd.DataFrame(coef_low_5bp),  # CI = 0.99 = (1 - 2 * 0.005)
            pd.DataFrame(coef_upp_5bp),  # CI = 0.99 = (1 - 2 * 0.005)
            ],axis=1)
        df_ols_coeffs_curr.columns = [
            "indep_var", "dep_var", "coef", "std_err", "t_value", "p_value",
            "coef_low_25bp", "coef_upp_25bp", "coef_low_5bp", "coef_upp_5bp",]
    else :
        lst_indep_var = [str_y_name] * (2 + len(lst_X_columns))
        lst_dep_var = ["LASSO_ALPHA", "INTERCEPT"] + lst_X_columns
        df_ols_coeffs_curr = pd.concat([
            pd.DataFrame(lst_indep_var),
            pd.DataFrame(lst_dep_var),
            pd.DataFrame(coef),
            ], axis=1)
        df_ols_coeffs_curr.columns = ["indep_var", "dep_var", "coef",]
    if df_predicted_vote_counts_out is not None and str_y_name is not None:
        y_hat[y_hat < 0.] = 0.
        df_predicted_vote_counts_out[str_y_name] = y_hat
    return df_ols_coeffs_curr

###############################################################################

def replace_residual_choices_with_aggregate_choice(
        df_vote_counts,
        str_residual_vote_count_column_name,
        lst_str_residual_vote_count_column_names,
        ) :
    if str_residual_vote_count_column_name is not None and \
       len(lst_str_residual_vote_count_column_names) > 0 :
        if str_residual_vote_count_column_name in \
           lst_str_residual_vote_count_column_names :
            for str_col_name in lst_str_residual_vote_count_column_names :
                if str_col_name != str_residual_vote_count_column_name :
                    df_vote_counts.loc[:,str_residual_vote_count_column_name]+=\
                        df_vote_counts.loc[:,str_col_name]
                    df_vote_counts.drop(str_col_name, axis = 1, inplace = True)
        else :
            df_vote_counts[str_residual_vote_count_column_name] = 0
            for str_col_name in lst_str_residual_vote_count_column_names :
                df_vote_counts.loc[:,str_residual_vote_count_column_name] += \
                    df_vote_counts.loc[:,str_col_name]
            df_vote_counts.drop(
                lst_str_residual_vote_count_column_names,
                axis = 1, inplace = True)
    return df_vote_counts

###############################################################################

def generate_predicted_counts(
    str_actual_vote_counts_csv_file_name,
    lst_str_actual_all_vote_count_column_names = [],
    str_actual_residual_vote_count_column_name = None,
    lst_str_actual_residual_vote_count_column_names = [],
    #
    str_benchmark_vote_counts_csv_file_name = None,
    lst_str_benchmark_all_vote_count_column_names = [],
    str_benchmark_residual_vote_count_column_name = None,
    lst_str_benchmark_residual_vote_count_column_names = [],
    #
    bool_aggregate_vote_counts_by_county = False,
    bool_approximate_missing_benchmark_precincts_by_scaled_benchmark_county = False,
    bool_aggregate_missing_benchmark_precincts_into_new_residual_county = False,
    #
    bool_produce_actual_vote_counts = False,
    #
    flt_lasso_alpha_l1_regularization_strength_term = None, # [0.; 1.]. # None by default (1.)
    int_lasso_maximum_number_of_iterations = 1_000_000,
    int_lasso_optimization_tolerance = 0.00001,
    flt_lasso_cv_length_of_alphas_regularization_path = 1., # (0.; +1.] # 1 by default (0.001); alpha_min / alpha_max
    int_lasso_cv_num_candidate_alphas_on_regularization_path = 0, # [0; +Inf] # 0 by default (100)
    int_lasso_cv_maximum_number_of_iterations = 1_000_000,
    int_lasso_cv_optimization_tolerance = 0.00001,
    int_lasso_cv_number_of_folds_in_cross_validation = 10,
    #
    bool_estimate_precinct_model_on_aggregated_vote_counts_by_county = False,
    bool_approximate_predictions_for_missing_benchmark_precincts_by_actual_values = False,
    #
    bool_estimate_model_parameters_diagnostics = False,
    bool_save_estimated_models_parameters_to_csv_file = False,
    str_csv_file_name_for_saving_estimated_models_parameters = None,
    #
    bool_distribute_job_output_files_in_directories_by_type = True,
    bool_gather_all_job_files_in_one_directory = False,
    str_job_subdir_name = None,
    #
    lst_str_predicted_for_actual_vote_count_column_names = [],
    #
    lst_str_predicting_from_ln_of_incremented_actual_vote_count_column_names = [],
    lst_str_predicting_from_power_one_quarter_of_actual_vote_count_column_names = [],
    lst_str_predicting_from_square_root_of_actual_vote_count_column_names = [],
    lst_str_predicting_from_power_three_quarters_of_actual_vote_count_column_names = [],
    lst_str_predicting_from_actual_vote_count_column_names = [],
    lst_str_predicting_from_power_one_and_a_half_of_actual_vote_count_column_names = [],
    lst_str_predicting_from_squared_actual_vote_count_column_names = [],
    #
    lst_str_predicting_from_ln_of_incremented_benchmark_vote_count_column_names = [],
    lst_str_predicting_from_power_one_quarter_of_benchmark_vote_count_column_names = [],
    lst_str_predicting_from_square_root_of_benchmark_vote_count_column_names = [],
    lst_str_predicting_from_power_three_quarters_of_benchmark_vote_count_column_names = [],
    lst_str_predicting_from_benchmark_vote_count_column_names = [],
    lst_str_predicting_from_power_one_and_a_half_of_benchmark_vote_count_column_names = [],
    lst_str_predicting_from_squared_benchmark_vote_count_column_names = [],
    #
    # [(a,b) for a in lst_str_predicting_from_actual_vote_count_column_names
    #        for b in lst_str_predicting_from_actual_vote_count_column_names]
    lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_actual_vote_count_column_names = [],
    lst_pairs_str_predicting_from_square_root_of_actual_interactions_actual_vote_count_column_names = [],
    lst_pairs_str_predicting_from_actual_interactions_actual_vote_count_column_names = [],
    #
    # [(a,b) for a in lst_str_predicting_from_benchmark_vote_count_column_names
    #        for b in lst_str_predicting_from_benchmark_vote_count_column_names]
    lst_pairs_str_predicting_from_fourth_root_of_benchmark_interactions_benchmark_vote_count_column_names = [],
    lst_pairs_str_predicting_from_square_root_of_benchmark_interactions_benchmark_vote_count_column_names = [],
    lst_pairs_str_predicting_from_benchmark_interactions_benchmark_vote_count_column_names = [],
    #
    # [(a,b) for a in lst_str_predicting_from_actual_vote_count_column_names
    #        for b in lst_str_predicting_from_benchmark_vote_count_column_names]
    lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_benchmark_vote_count_column_names = [],
    lst_pairs_str_predicting_from_square_root_of_actual_interactions_benchmark_vote_count_column_names = [],
    lst_pairs_str_predicting_from_actual_interactions_benchmark_vote_count_column_names = [],
    #######################################################################
    bool_drop_predicted_variable_during_computation_of_predicting_tally = True,
    #######################################################################
    bool_predicting_from_ln_of_incremented_actual_tally = False,
    bool_predicting_from_power_one_quarter_of_actual_tally = False,
    bool_predicting_from_square_root_of_actual_tally = False,
    bool_predicting_from_power_three_quarters_of_actual_tally = False,
    bool_predicting_from_actual_tally = False,
    bool_predicting_from_power_one_and_a_half_of_actual_tally = False,
    bool_predicting_from_squared_actual_tally = False,
    #
    bool_predicting_from_ln_of_incremented_benchmark_tally = False,
    bool_predicting_from_power_one_quarter_of_benchmark_tally = False,
    bool_predicting_from_square_root_of_benchmark_tally = False,
    bool_predicting_from_power_three_quarters_of_benchmark_tally = False,
    bool_predicting_from_benchmark_tally = False,
    bool_predicting_from_power_one_and_a_half_of_benchmark_tally = False,
    bool_predicting_from_squared_benchmark_tally = False,
    #
    bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally = False,
    bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally = False,
    bool_predicting_from_actual_tally_interaction_benchmark_tally = False,
    #######################################################################
    bool_predicting_from_ln_of_incremented_actual_tally_interaction_county_indicator = False,
    bool_predicting_from_power_one_quarter_of_actual_tally_interaction_county_indicator = False,
    bool_predicting_from_square_root_of_actual_tally_interaction_county_indicator = False,
    bool_predicting_from_power_three_quarters_of_actual_tally_interaction_county_indicator = False,
    bool_predicting_from_actual_tally_interaction_county_indicator = False,
    bool_predicting_from_power_one_and_a_half_of_actual_tally_interaction_county_indicator = False,
    bool_predicting_from_squared_actual_tally_interaction_county_indicator = False,
    #
    bool_predicting_from_ln_of_incremented_benchmark_tally_interaction_county_indicator = False,
    bool_predicting_from_power_one_quarter_of_benchmark_tally_interaction_county_indicator = False,
    bool_predicting_from_square_root_of_benchmark_tally_interaction_county_indicator = False,
    bool_predicting_from_power_three_quarters_of_benchmark_tally_interaction_county_indicator = False,
    bool_predicting_from_benchmark_tally_interaction_county_indicator = False,
    bool_predicting_from_power_one_and_a_half_of_benchmark_tally_interaction_county_indicator = False,
    bool_predicting_from_squared_benchmark_tally_interaction_county_indicator = False,
    #
    bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator = False,
    bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator = False,
    bool_predicting_from_actual_tally_interaction_benchmark_tally_interaction_county_indicator = False,
    ) :
    '''<function description>
    
    <calling functions>
    <called functions>

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

    if bool_approximate_missing_benchmark_precincts_by_scaled_benchmark_county or \
       bool_aggregate_missing_benchmark_precincts_into_new_residual_county :

        df_actual_vote_counts = generate_merged_counts(
            str_actual_vote_counts_csv_file_name =
                str_actual_vote_counts_csv_file_name,
            lst_str_actual_all_vote_count_column_names =
                lst_str_actual_all_vote_count_column_names,
            str_actual_residual_vote_count_column_name =
                str_actual_residual_vote_count_column_name,
            lst_str_actual_residual_vote_count_column_names =
                lst_str_actual_residual_vote_count_column_names,
            #
            str_benchmark_vote_counts_csv_file_name =
                str_benchmark_vote_counts_csv_file_name,
            lst_str_benchmark_all_vote_count_column_names =
                lst_str_benchmark_all_vote_count_column_names,
            str_benchmark_residual_vote_count_column_name =
                str_benchmark_residual_vote_count_column_name,
            lst_str_benchmark_residual_vote_count_column_names =
                lst_str_benchmark_residual_vote_count_column_names,
            #
            bool_aggregate_vote_counts_by_county =
                bool_aggregate_vote_counts_by_county,
            bool_approximate_missing_benchmark_precincts_by_scaled_benchmark_county =
                bool_approximate_missing_benchmark_precincts_by_scaled_benchmark_county,
            bool_aggregate_missing_benchmark_precincts_into_new_residual_county =
                bool_aggregate_missing_benchmark_precincts_into_new_residual_county,)

        if str_benchmark_vote_counts_csv_file_name is not None and \
           lst_str_benchmark_all_vote_count_column_names is not None and \
           len(lst_str_benchmark_all_vote_count_column_names) > 0 :

            df_benchmark_vote_counts = df_actual_vote_counts[[
                STR_IS_AUDITED_COLUMN_NAME, STR_COUNTY_NAME_COLUMN_NAME,
                STR_PRECINCT_NAME_COLUMN_NAME, STR_PRECINCT_CODE_COLUMN_NAME] +
                lst_str_benchmark_all_vote_count_column_names]
        else :

            df_benchmark_vote_counts = None

        if lst_str_benchmark_all_vote_count_column_names is not None and \
           len(lst_str_benchmark_all_vote_count_column_names) > 0 :

            df_actual_vote_counts.drop(
                columns = lst_str_benchmark_all_vote_count_column_names,
                axis = 1,
                inplace = True,)
    else :
        df_actual_vote_counts = pd.read_csv(os.path.join(
            STR_INPUT_DATA_CSV_PATH,
            str_actual_vote_counts_csv_file_name + ".csv"))
        if str_actual_residual_vote_count_column_name is not None and \
           len(lst_str_actual_residual_vote_count_column_names) > 0 :
            df_actual_vote_counts = \
                replace_residual_choices_with_aggregate_choice(
                    df_vote_counts = df_actual_vote_counts,
                    str_residual_vote_count_column_name =
                        str_actual_residual_vote_count_column_name,
                    lst_str_residual_vote_count_column_names =
                        lst_str_actual_residual_vote_count_column_names,)

        if str_benchmark_vote_counts_csv_file_name is not None and \
           str_benchmark_vote_counts_csv_file_name != "" :
            df_benchmark_vote_counts = pd.read_csv(os.path.join(
                STR_INPUT_DATA_CSV_PATH,
                str_benchmark_vote_counts_csv_file_name + ".csv"))
            if str_benchmark_residual_vote_count_column_name is not None and \
               len(lst_str_benchmark_residual_vote_count_column_names) > 0 :
                df_benchmark_vote_counts = \
                    replace_residual_choices_with_aggregate_choice(
                        df_vote_counts = df_benchmark_vote_counts,
                        str_residual_vote_count_column_name =
                            str_benchmark_residual_vote_count_column_name,
                        lst_str_residual_vote_count_column_names =
                            lst_str_benchmark_residual_vote_count_column_names,)
        else :
            df_benchmark_vote_counts = None

    df_predicted_vote_counts = None
    df_ols_coeffs = None
    if bool_produce_actual_vote_counts :
        if bool_aggregate_vote_counts_by_county :
            df_actual_vote_counts_copy = df_actual_vote_counts.copy()
            df_actual_vote_counts_copy[STR_PRECINCT_NAME_COLUMN_NAME] = "Composite"
            df_actual_vote_counts_copy[STR_PRECINCT_CODE_COLUMN_NAME] = "Composite"
            
            dict_predicted_for_actual_col_name_to_agg_oper = {
                n : "sum" for n in lst_str_predicted_for_actual_vote_count_column_names}
            dict_predicted_for_actual_col_name_to_agg_oper[STR_IS_AUDITED_COLUMN_NAME] = "min"
            df_actual_vote_counts_aggr_by_county = df_actual_vote_counts_copy.groupby([
                STR_COUNTY_NAME_COLUMN_NAME, STR_PRECINCT_NAME_COLUMN_NAME, STR_PRECINCT_CODE_COLUMN_NAME],
                sort=True).agg(dict_predicted_for_actual_col_name_to_agg_oper).reset_index().rename(
                columns={'index':STR_COUNTY_NAME_COLUMN_NAME})
            df_actual_vote_counts_aggr_by_county.insert(
                0, STR_IS_AUDITED_COLUMN_NAME,
                df_actual_vote_counts_aggr_by_county.pop(STR_IS_AUDITED_COLUMN_NAME)) 

            df_actual_vote_counts_copy = None
            df_predicted_vote_counts = df_actual_vote_counts_aggr_by_county
        else :
            df_predicted_vote_counts = df_actual_vote_counts[
                [STR_IS_AUDITED_COLUMN_NAME, STR_COUNTY_NAME_COLUMN_NAME,
                 STR_PRECINCT_NAME_COLUMN_NAME, STR_PRECINCT_CODE_COLUMN_NAME] +
                lst_str_predicted_for_actual_vote_count_column_names].copy()

        for str_y_name in lst_str_predicted_for_actual_vote_count_column_names :
            X = df_predicted_vote_counts[[str_y_name]]
            y = X
            df_ols_coeffs_curr = estimate_linear_model_with_summary_in_dataframe(
                str_y_name = str_y_name,
                y = y,
                X = X,
                flt_lasso_alpha_l1_regularization_strength_term =
                    flt_lasso_alpha_l1_regularization_strength_term,
                int_lasso_maximum_number_of_iterations =
                    int_lasso_maximum_number_of_iterations,
                int_lasso_optimization_tolerance =
                    int_lasso_optimization_tolerance,
                flt_lasso_cv_length_of_alphas_regularization_path =
                    flt_lasso_cv_length_of_alphas_regularization_path,
                int_lasso_cv_num_candidate_alphas_on_regularization_path =
                    int_lasso_cv_num_candidate_alphas_on_regularization_path,
                int_lasso_cv_maximum_number_of_iterations =
                    int_lasso_cv_maximum_number_of_iterations,
                int_lasso_cv_optimization_tolerance =
                    int_lasso_cv_optimization_tolerance,
                int_lasso_cv_number_of_folds_in_cross_validation =
                    int_lasso_cv_number_of_folds_in_cross_validation,
                df_predicted_vote_counts_out = None,
                bool_estimate_model_parameters_diagnostics =
                    bool_estimate_model_parameters_diagnostics,
                )
            if df_ols_coeffs is None :
                df_ols_coeffs = df_ols_coeffs_curr
            else :
                df_ols_coeffs = pd.concat(
                    [df_ols_coeffs, df_ols_coeffs_curr],
                    axis = 0,
                    ignore_index = True,)
    else :
        if bool_estimate_precinct_model_on_aggregated_vote_counts_by_county :

            # Prepare output dataframe
            df_actual_vote_counts_copy = df_actual_vote_counts.copy()
            df_actual_vote_counts_copy[STR_PRECINCT_NAME_COLUMN_NAME] = "Composite"
            df_actual_vote_counts_copy[STR_PRECINCT_CODE_COLUMN_NAME] = "Composite"

            dict_predicted_for_actual_col_name_to_agg_oper = {
                n : "sum" for n in lst_str_predicted_for_actual_vote_count_column_names}
            dict_predicted_for_actual_col_name_to_agg_oper[STR_IS_AUDITED_COLUMN_NAME] = "min"
            df_predicted_vote_counts_by_county = df_actual_vote_counts_copy.groupby([
                STR_COUNTY_NAME_COLUMN_NAME, STR_PRECINCT_NAME_COLUMN_NAME, STR_PRECINCT_CODE_COLUMN_NAME],
                sort=True).agg(dict_predicted_for_actual_col_name_to_agg_oper).reset_index().rename(
                columns={'index':STR_COUNTY_NAME_COLUMN_NAME})
            df_predicted_vote_counts_by_county.insert(
                0, STR_IS_AUDITED_COLUMN_NAME, df_predicted_vote_counts_by_county.pop(STR_IS_AUDITED_COLUMN_NAME)) 

            df_actual_vote_counts_copy = None

            # Prepare input dataframes
            df_actual_vote_counts_aggr_by_county = df_actual_vote_counts.groupby(
                [STR_COUNTY_NAME_COLUMN_NAME], sort=True).agg(
                dict_predicted_for_actual_col_name_to_agg_oper).reset_index().rename(
                columns={'index':STR_COUNTY_NAME_COLUMN_NAME})
            df_actual_vote_counts_aggr_by_county.insert(
                0, STR_IS_AUDITED_COLUMN_NAME, df_actual_vote_counts_aggr_by_county.pop(STR_IS_AUDITED_COLUMN_NAME)) 
            if df_benchmark_vote_counts is not None :
                dict_predicting_from_benchmark_col_name_to_agg_oper = {
                    n : "sum" for n in lst_str_predicting_from_benchmark_vote_count_column_names}
                df_benchmark_vote_counts_aggr_by_county = df_benchmark_vote_counts.groupby(
                    [STR_COUNTY_NAME_COLUMN_NAME], sort=True).agg(
                    dict_predicting_from_benchmark_col_name_to_agg_oper).reset_index(
                        ).rename(columns={'index':STR_COUNTY_NAME_COLUMN_NAME})
            else :
                df_benchmark_vote_counts_aggr_by_county = None

            # Fit model for and predict one choice after the other.
            for str_y_name in lst_str_predicted_for_actual_vote_count_column_names :
                X = assemble_predictors(
                    str_predicted_for_actual_votes_counts_column_name = str_y_name,
                    df_actual_vote_counts = df_actual_vote_counts_aggr_by_county,
                    df_benchmark_vote_counts = df_benchmark_vote_counts_aggr_by_county,
                    ###########################################################
                    lst_str_actual_merge_keys = [STR_COUNTY_NAME_COLUMN_NAME],
                    lst_str_benchmark_merge_keys = [STR_COUNTY_NAME_COLUMN_NAME],
                    lst_str_actual_all_vote_count_column_names =
                        lst_str_actual_all_vote_count_column_names,
                    lst_str_benchmark_all_vote_count_column_names =
                        lst_str_benchmark_all_vote_count_column_names,
                    ###########################################################
                    lst_str_predicting_from_ln_of_incremented_actual_vote_count_column_names =
                        lst_str_predicting_from_ln_of_incremented_actual_vote_count_column_names,
                    lst_str_predicting_from_power_one_quarter_of_actual_vote_count_column_names =
                        lst_str_predicting_from_power_one_quarter_of_actual_vote_count_column_names,
                    lst_str_predicting_from_square_root_of_actual_vote_count_column_names =
                        lst_str_predicting_from_square_root_of_actual_vote_count_column_names,
                    lst_str_predicting_from_power_three_quarters_of_actual_vote_count_column_names =
                        lst_str_predicting_from_power_three_quarters_of_actual_vote_count_column_names,
                    lst_str_predicting_from_actual_vote_count_column_names =
                        lst_str_predicting_from_actual_vote_count_column_names,
                    lst_str_predicting_from_power_one_and_a_half_of_actual_vote_count_column_names =
                        lst_str_predicting_from_power_one_and_a_half_of_actual_vote_count_column_names,
                    lst_str_predicting_from_squared_actual_vote_count_column_names =
                        lst_str_predicting_from_squared_actual_vote_count_column_names,
                    #
                    lst_str_predicting_from_ln_of_incremented_benchmark_vote_count_column_names =
                        lst_str_predicting_from_ln_of_incremented_benchmark_vote_count_column_names,
                    lst_str_predicting_from_power_one_quarter_of_benchmark_vote_count_column_names =
                        lst_str_predicting_from_power_one_quarter_of_benchmark_vote_count_column_names,
                    lst_str_predicting_from_square_root_of_benchmark_vote_count_column_names =
                        lst_str_predicting_from_square_root_of_benchmark_vote_count_column_names,
                    lst_str_predicting_from_power_three_quarters_of_benchmark_vote_count_column_names =
                        lst_str_predicting_from_power_three_quarters_of_benchmark_vote_count_column_names,
                    lst_str_predicting_from_benchmark_vote_count_column_names =
                        lst_str_predicting_from_benchmark_vote_count_column_names,
                    lst_str_predicting_from_power_one_and_a_half_of_benchmark_vote_count_column_names =
                        lst_str_predicting_from_power_one_and_a_half_of_benchmark_vote_count_column_names,
                    lst_str_predicting_from_squared_benchmark_vote_count_column_names =
                        lst_str_predicting_from_squared_benchmark_vote_count_column_names,
                    #
                    lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_actual_vote_count_column_names =
                        lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_actual_vote_count_column_names,
                    lst_pairs_str_predicting_from_square_root_of_actual_interactions_actual_vote_count_column_names =
                        lst_pairs_str_predicting_from_square_root_of_actual_interactions_actual_vote_count_column_names,
                    lst_pairs_str_predicting_from_actual_interactions_actual_vote_count_column_names =
                        lst_pairs_str_predicting_from_actual_interactions_actual_vote_count_column_names,
                    #
                    lst_pairs_str_predicting_from_fourth_root_of_benchmark_interactions_benchmark_vote_count_column_names =
                        lst_pairs_str_predicting_from_fourth_root_of_benchmark_interactions_benchmark_vote_count_column_names,
                    lst_pairs_str_predicting_from_square_root_of_benchmark_interactions_benchmark_vote_count_column_names =
                        lst_pairs_str_predicting_from_square_root_of_benchmark_interactions_benchmark_vote_count_column_names,
                    lst_pairs_str_predicting_from_benchmark_interactions_benchmark_vote_count_column_names =
                        lst_pairs_str_predicting_from_benchmark_interactions_benchmark_vote_count_column_names,
                    #
                    lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_benchmark_vote_count_column_names =
                        lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_benchmark_vote_count_column_names,
                    lst_pairs_str_predicting_from_square_root_of_actual_interactions_benchmark_vote_count_column_names =
                        lst_pairs_str_predicting_from_square_root_of_actual_interactions_benchmark_vote_count_column_names,
                    lst_pairs_str_predicting_from_actual_interactions_benchmark_vote_count_column_names =
                        lst_pairs_str_predicting_from_actual_interactions_benchmark_vote_count_column_names,
                    ###########################################################
                    bool_drop_predicted_variable_during_computation_of_predicting_tally =
                        bool_drop_predicted_variable_during_computation_of_predicting_tally,
                    ###########################################################
                    bool_predicting_from_ln_of_incremented_actual_tally =
                        bool_predicting_from_ln_of_incremented_actual_tally,
                    bool_predicting_from_power_one_quarter_of_actual_tally =
                        bool_predicting_from_power_one_quarter_of_actual_tally,
                    bool_predicting_from_square_root_of_actual_tally =
                        bool_predicting_from_square_root_of_actual_tally,
                    bool_predicting_from_power_three_quarters_of_actual_tally =
                        bool_predicting_from_power_three_quarters_of_actual_tally,
                    bool_predicting_from_actual_tally =
                        bool_predicting_from_actual_tally,
                    bool_predicting_from_power_one_and_a_half_of_actual_tally =
                        bool_predicting_from_power_one_and_a_half_of_actual_tally,
                    bool_predicting_from_squared_actual_tally =
                        bool_predicting_from_squared_actual_tally,
                    #
                    bool_predicting_from_ln_of_incremented_benchmark_tally =
                        bool_predicting_from_ln_of_incremented_benchmark_tally,
                    bool_predicting_from_power_one_quarter_of_benchmark_tally =
                        bool_predicting_from_power_one_quarter_of_benchmark_tally,
                    bool_predicting_from_square_root_of_benchmark_tally =
                        bool_predicting_from_square_root_of_benchmark_tally,
                    bool_predicting_from_power_three_quarters_of_benchmark_tally =
                        bool_predicting_from_power_three_quarters_of_benchmark_tally,
                    bool_predicting_from_benchmark_tally =
                        bool_predicting_from_benchmark_tally,
                    bool_predicting_from_power_one_and_a_half_of_benchmark_tally =
                        bool_predicting_from_power_one_and_a_half_of_benchmark_tally,
                    bool_predicting_from_squared_benchmark_tally =
                        bool_predicting_from_squared_benchmark_tally,
                    #
                    bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally =
                        bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally,
                    bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally =
                        bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally,
                    bool_predicting_from_actual_tally_interaction_benchmark_tally =
                        bool_predicting_from_actual_tally_interaction_benchmark_tally,
                    ###########################################################
                    bool_predicting_from_ln_of_incremented_actual_tally_interaction_county_indicator =
                        bool_predicting_from_ln_of_incremented_actual_tally_interaction_county_indicator,
                    bool_predicting_from_power_one_quarter_of_actual_tally_interaction_county_indicator =
                        bool_predicting_from_power_one_quarter_of_actual_tally_interaction_county_indicator,
                    bool_predicting_from_square_root_of_actual_tally_interaction_county_indicator =
                        bool_predicting_from_square_root_of_actual_tally_interaction_county_indicator,
                    bool_predicting_from_power_three_quarters_of_actual_tally_interaction_county_indicator =
                        bool_predicting_from_power_three_quarters_of_actual_tally_interaction_county_indicator,
                    bool_predicting_from_actual_tally_interaction_county_indicator =
                        bool_predicting_from_actual_tally_interaction_county_indicator,
                    bool_predicting_from_power_one_and_a_half_of_actual_tally_interaction_county_indicator =
                        bool_predicting_from_power_one_and_a_half_of_actual_tally_interaction_county_indicator,
                    bool_predicting_from_squared_actual_tally_interaction_county_indicator =
                        bool_predicting_from_squared_actual_tally_interaction_county_indicator,
                    #
                    bool_predicting_from_ln_of_incremented_benchmark_tally_interaction_county_indicator =
                        bool_predicting_from_ln_of_incremented_benchmark_tally_interaction_county_indicator,
                    bool_predicting_from_power_one_quarter_of_benchmark_tally_interaction_county_indicator =
                        bool_predicting_from_power_one_quarter_of_benchmark_tally_interaction_county_indicator,
                    bool_predicting_from_square_root_of_benchmark_tally_interaction_county_indicator =
                        bool_predicting_from_square_root_of_benchmark_tally_interaction_county_indicator,
                    bool_predicting_from_power_three_quarters_of_benchmark_tally_interaction_county_indicator =
                        bool_predicting_from_power_three_quarters_of_benchmark_tally_interaction_county_indicator,
                    bool_predicting_from_benchmark_tally_interaction_county_indicator =
                        bool_predicting_from_benchmark_tally_interaction_county_indicator,
                    bool_predicting_from_power_one_and_a_half_of_benchmark_tally_interaction_county_indicator =
                        bool_predicting_from_power_one_and_a_half_of_benchmark_tally_interaction_county_indicator,
                    bool_predicting_from_squared_benchmark_tally_interaction_county_indicator =
                        bool_predicting_from_squared_benchmark_tally_interaction_county_indicator,
                    #
                    bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator =
                        bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator,
                    bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator =
                        bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator,
                    bool_predicting_from_actual_tally_interaction_benchmark_tally_interaction_county_indicator =
                        bool_predicting_from_actual_tally_interaction_benchmark_tally_interaction_county_indicator,
                    )
                y = df_predicted_vote_counts_by_county[[str_y_name]]
                df_ols_coeffs_curr = \
                    estimate_linear_model_with_summary_in_dataframe(
                        str_y_name = str_y_name,
                        y = y,
                        X = X,
                        flt_lasso_alpha_l1_regularization_strength_term =
                            flt_lasso_alpha_l1_regularization_strength_term,
                        int_lasso_maximum_number_of_iterations =
                            int_lasso_maximum_number_of_iterations,
                        int_lasso_optimization_tolerance =
                            int_lasso_optimization_tolerance,
                        flt_lasso_cv_length_of_alphas_regularization_path =
                            flt_lasso_cv_length_of_alphas_regularization_path,
                        int_lasso_cv_num_candidate_alphas_on_regularization_path =
                            int_lasso_cv_num_candidate_alphas_on_regularization_path,
                        int_lasso_cv_maximum_number_of_iterations =
                            int_lasso_cv_maximum_number_of_iterations,
                        int_lasso_cv_optimization_tolerance =
                            int_lasso_cv_optimization_tolerance,
                        int_lasso_cv_number_of_folds_in_cross_validation =
                            int_lasso_cv_number_of_folds_in_cross_validation,
                        df_predicted_vote_counts_out =
                            df_predicted_vote_counts_by_county,
                        bool_estimate_model_parameters_diagnostics =
                            bool_estimate_model_parameters_diagnostics,
                        )
                if df_ols_coeffs is None :
                    df_ols_coeffs = df_ols_coeffs_curr
                else :
                    df_ols_coeffs = pd.concat(
                        [df_ols_coeffs, df_ols_coeffs_curr],
                        axis=0, ignore_index=True)

            if not bool_aggregate_vote_counts_by_county :
                # Prepare output dataframe
                df_predicted_vote_counts_by_precinct = df_actual_vote_counts[
                    [STR_IS_AUDITED_COLUMN_NAME, STR_COUNTY_NAME_COLUMN_NAME,
                     STR_PRECINCT_NAME_COLUMN_NAME, STR_PRECINCT_CODE_COLUMN_NAME] +
                    lst_str_predicted_for_actual_vote_count_column_names].copy()

                # Compute and combine row totals for each precinct with county-level row totals on each precinct row.
                df_predicted_vote_counts_by_county['Total_Per_County'] = df_predicted_vote_counts_by_county[
                    lst_str_predicted_for_actual_vote_count_column_names].sum(axis=1)
                df_predicted_vote_counts_by_precinct['Total_Per_Precinct'] = df_predicted_vote_counts_by_precinct[
                    lst_str_predicted_for_actual_vote_count_column_names].sum(axis=1)
                df_predicted_vote_counts_by_precinct = df_predicted_vote_counts_by_precinct[
                    [STR_IS_AUDITED_COLUMN_NAME, STR_COUNTY_NAME_COLUMN_NAME,
                     STR_PRECINCT_NAME_COLUMN_NAME, STR_PRECINCT_CODE_COLUMN_NAME] +
                    ['Total_Per_Precinct']].merge(df_predicted_vote_counts_by_county[
                        [STR_COUNTY_NAME_COLUMN_NAME] + lst_str_predicted_for_actual_vote_count_column_names +
                        ['Total_Per_County']],
                    on=STR_COUNTY_NAME_COLUMN_NAME, how='inner',)

            if bool_aggregate_vote_counts_by_county :
                # Final cleaning and rounding on the county level
                df_predicted_vote_counts_by_county = df_predicted_vote_counts_by_county.round(0)
                df_predicted_vote_counts_by_county[lst_str_predicted_for_actual_vote_count_column_names] = \
                    df_predicted_vote_counts_by_county[lst_str_predicted_for_actual_vote_count_column_names].astype(int)
                df_predicted_vote_counts = df_predicted_vote_counts_by_county
            else :
                # Scaling the predicted values by the ratio "Precinct Size" over "County Size"
                df_predicted_vote_counts_by_precinct['Precinct_over_County_Ratio'] = \
                    df_predicted_vote_counts_by_precinct[['Total_Per_Precinct']].div(
                    df_predicted_vote_counts_by_precinct.Total_Per_County, axis=0)
                df_predicted_vote_counts_by_precinct[lst_str_predicted_for_actual_vote_count_column_names] = \
                    df_predicted_vote_counts_by_precinct[lst_str_predicted_for_actual_vote_count_column_names].mul(
                    df_predicted_vote_counts_by_precinct.Precinct_over_County_Ratio, axis=0)

                # Final cleaning and rounding on the precinct level
                df_predicted_vote_counts_by_precinct.drop(
                    columns=['Total_Per_County', 'Total_Per_Precinct', 'Precinct_over_County_Ratio'], axis=1, inplace=True)
                df_predicted_vote_counts_by_precinct = df_predicted_vote_counts_by_precinct.round(0)
                df_predicted_vote_counts_by_precinct[lst_str_predicted_for_actual_vote_count_column_names] = \
                    df_predicted_vote_counts_by_precinct[lst_str_predicted_for_actual_vote_count_column_names].astype(int)
                df_predicted_vote_counts = df_predicted_vote_counts_by_precinct
        else : # if bool_estimate_precinct_model_on_aggregated_vote_counts_by_county
            # Prepare output dataframe
            lst_str_merge_keys = [STR_COUNTY_NAME_COLUMN_NAME, STR_PRECINCT_NAME_COLUMN_NAME, STR_PRECINCT_CODE_COLUMN_NAME]
            if df_benchmark_vote_counts is not None :
                df_predicted_vote_counts_by_precinct = df_actual_vote_counts[
                    [STR_IS_AUDITED_COLUMN_NAME] + lst_str_merge_keys +
                    lst_str_predicted_for_actual_vote_count_column_names].merge(
                        df_benchmark_vote_counts[lst_str_merge_keys], on=lst_str_merge_keys, how='inner')
                df_predicted_vote_counts_by_precinct.insert(
                    0, STR_IS_AUDITED_COLUMN_NAME, df_predicted_vote_counts_by_precinct.pop(
                    STR_IS_AUDITED_COLUMN_NAME)) 
            else :
                df_predicted_vote_counts_by_precinct = df_actual_vote_counts[
                    [STR_IS_AUDITED_COLUMN_NAME] + lst_str_merge_keys +
                    lst_str_predicted_for_actual_vote_count_column_names].copy()
            if bool_approximate_predictions_for_missing_benchmark_precincts_by_actual_values :
                if df_benchmark_vote_counts is not None :
                    df_actual_as_predicted_vote_counts_by_precinct = df_actual_vote_counts[
                        [STR_IS_AUDITED_COLUMN_NAME] + lst_str_merge_keys +
                        lst_str_predicted_for_actual_vote_count_column_names].merge(
                            df_benchmark_vote_counts[lst_str_merge_keys], on=lst_str_merge_keys,
                            how='left', indicator=True)
                    df_actual_as_predicted_vote_counts_by_precinct = df_actual_as_predicted_vote_counts_by_precinct[
                        df_actual_as_predicted_vote_counts_by_precinct['_merge'] == 'left_only']
                    df_actual_as_predicted_vote_counts_by_precinct.drop(['_merge'], axis=1, inplace=True)
                else :
                    df_actual_as_predicted_vote_counts_by_precinct = df_actual_vote_counts[
                        [STR_IS_AUDITED_COLUMN_NAME] + lst_str_merge_keys +
                        lst_str_predicted_for_actual_vote_count_column_names].iloc[:0].copy()

            # Fit model for and predict one choice after the other.
            for str_y_name in lst_str_predicted_for_actual_vote_count_column_names :
                X = assemble_predictors(
                    str_predicted_for_actual_votes_counts_column_name = str_y_name,
                    df_actual_vote_counts = df_actual_vote_counts,
                    df_benchmark_vote_counts = df_benchmark_vote_counts,
                    ###########################################################
                    lst_str_actual_merge_keys = [
                        STR_COUNTY_NAME_COLUMN_NAME, STR_PRECINCT_NAME_COLUMN_NAME, STR_PRECINCT_CODE_COLUMN_NAME],
                    lst_str_benchmark_merge_keys = [
                        STR_COUNTY_NAME_COLUMN_NAME, STR_PRECINCT_NAME_COLUMN_NAME, STR_PRECINCT_CODE_COLUMN_NAME],
                    lst_str_actual_all_vote_count_column_names =
                        lst_str_actual_all_vote_count_column_names,
                    lst_str_benchmark_all_vote_count_column_names =
                        lst_str_benchmark_all_vote_count_column_names,
                    ###########################################################
                    lst_str_predicting_from_ln_of_incremented_actual_vote_count_column_names =
                        lst_str_predicting_from_ln_of_incremented_actual_vote_count_column_names,
                    lst_str_predicting_from_power_one_quarter_of_actual_vote_count_column_names =
                        lst_str_predicting_from_power_one_quarter_of_actual_vote_count_column_names,
                    lst_str_predicting_from_square_root_of_actual_vote_count_column_names =
                        lst_str_predicting_from_square_root_of_actual_vote_count_column_names,
                    lst_str_predicting_from_power_three_quarters_of_actual_vote_count_column_names =
                        lst_str_predicting_from_power_three_quarters_of_actual_vote_count_column_names,
                    lst_str_predicting_from_actual_vote_count_column_names =
                        lst_str_predicting_from_actual_vote_count_column_names,
                    lst_str_predicting_from_power_one_and_a_half_of_actual_vote_count_column_names =
                        lst_str_predicting_from_power_one_and_a_half_of_actual_vote_count_column_names,
                    lst_str_predicting_from_squared_actual_vote_count_column_names =
                        lst_str_predicting_from_squared_actual_vote_count_column_names,
                    #
                    lst_str_predicting_from_ln_of_incremented_benchmark_vote_count_column_names =
                        lst_str_predicting_from_ln_of_incremented_benchmark_vote_count_column_names,
                    lst_str_predicting_from_power_one_quarter_of_benchmark_vote_count_column_names =
                        lst_str_predicting_from_power_one_quarter_of_benchmark_vote_count_column_names,
                    lst_str_predicting_from_square_root_of_benchmark_vote_count_column_names =
                        lst_str_predicting_from_square_root_of_benchmark_vote_count_column_names,
                    lst_str_predicting_from_power_three_quarters_of_benchmark_vote_count_column_names =
                        lst_str_predicting_from_power_three_quarters_of_benchmark_vote_count_column_names,
                    lst_str_predicting_from_benchmark_vote_count_column_names =
                        lst_str_predicting_from_benchmark_vote_count_column_names,
                    lst_str_predicting_from_power_one_and_a_half_of_benchmark_vote_count_column_names =
                        lst_str_predicting_from_power_one_and_a_half_of_benchmark_vote_count_column_names,
                    lst_str_predicting_from_squared_benchmark_vote_count_column_names =
                        lst_str_predicting_from_squared_benchmark_vote_count_column_names,
                    #
                    lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_actual_vote_count_column_names =
                        lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_actual_vote_count_column_names,
                    lst_pairs_str_predicting_from_square_root_of_actual_interactions_actual_vote_count_column_names =
                        lst_pairs_str_predicting_from_square_root_of_actual_interactions_actual_vote_count_column_names,
                    lst_pairs_str_predicting_from_actual_interactions_actual_vote_count_column_names =
                        lst_pairs_str_predicting_from_actual_interactions_actual_vote_count_column_names,
                    #
                    lst_pairs_str_predicting_from_fourth_root_of_benchmark_interactions_benchmark_vote_count_column_names =
                        lst_pairs_str_predicting_from_fourth_root_of_benchmark_interactions_benchmark_vote_count_column_names,
                    lst_pairs_str_predicting_from_square_root_of_benchmark_interactions_benchmark_vote_count_column_names =
                        lst_pairs_str_predicting_from_square_root_of_benchmark_interactions_benchmark_vote_count_column_names,
                    lst_pairs_str_predicting_from_benchmark_interactions_benchmark_vote_count_column_names =
                        lst_pairs_str_predicting_from_benchmark_interactions_benchmark_vote_count_column_names,
                    #
                    lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_benchmark_vote_count_column_names =
                        lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_benchmark_vote_count_column_names,
                    lst_pairs_str_predicting_from_square_root_of_actual_interactions_benchmark_vote_count_column_names =
                        lst_pairs_str_predicting_from_square_root_of_actual_interactions_benchmark_vote_count_column_names,
                    lst_pairs_str_predicting_from_actual_interactions_benchmark_vote_count_column_names =
                        lst_pairs_str_predicting_from_actual_interactions_benchmark_vote_count_column_names,
                    ###########################################################
                    bool_drop_predicted_variable_during_computation_of_predicting_tally =
                        bool_drop_predicted_variable_during_computation_of_predicting_tally,
                    ###########################################################
                    bool_predicting_from_ln_of_incremented_actual_tally =
                        bool_predicting_from_ln_of_incremented_actual_tally,
                    bool_predicting_from_power_one_quarter_of_actual_tally =
                        bool_predicting_from_power_one_quarter_of_actual_tally,
                    bool_predicting_from_square_root_of_actual_tally =
                        bool_predicting_from_square_root_of_actual_tally,
                    bool_predicting_from_power_three_quarters_of_actual_tally =
                        bool_predicting_from_power_three_quarters_of_actual_tally,
                    bool_predicting_from_actual_tally =
                        bool_predicting_from_actual_tally,
                    bool_predicting_from_power_one_and_a_half_of_actual_tally =
                        bool_predicting_from_power_one_and_a_half_of_actual_tally,
                    bool_predicting_from_squared_actual_tally =
                        bool_predicting_from_squared_actual_tally,
                    #
                    bool_predicting_from_ln_of_incremented_benchmark_tally =
                        bool_predicting_from_ln_of_incremented_benchmark_tally,
                    bool_predicting_from_power_one_quarter_of_benchmark_tally =
                        bool_predicting_from_power_one_quarter_of_benchmark_tally,
                    bool_predicting_from_square_root_of_benchmark_tally =
                        bool_predicting_from_square_root_of_benchmark_tally,
                    bool_predicting_from_power_three_quarters_of_benchmark_tally =
                        bool_predicting_from_power_three_quarters_of_benchmark_tally,
                    bool_predicting_from_benchmark_tally =
                        bool_predicting_from_benchmark_tally,
                    bool_predicting_from_power_one_and_a_half_of_benchmark_tally =
                        bool_predicting_from_power_one_and_a_half_of_benchmark_tally,
                    bool_predicting_from_squared_benchmark_tally =
                        bool_predicting_from_squared_benchmark_tally,
                    #
                    bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally =
                        bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally,
                    bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally =
                        bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally,
                    bool_predicting_from_actual_tally_interaction_benchmark_tally =
                        bool_predicting_from_actual_tally_interaction_benchmark_tally,
                    ###########################################################
                    bool_predicting_from_ln_of_incremented_actual_tally_interaction_county_indicator =
                        bool_predicting_from_ln_of_incremented_actual_tally_interaction_county_indicator,
                    bool_predicting_from_power_one_quarter_of_actual_tally_interaction_county_indicator =
                        bool_predicting_from_power_one_quarter_of_actual_tally_interaction_county_indicator,
                    bool_predicting_from_square_root_of_actual_tally_interaction_county_indicator =
                        bool_predicting_from_square_root_of_actual_tally_interaction_county_indicator,
                    bool_predicting_from_power_three_quarters_of_actual_tally_interaction_county_indicator =
                        bool_predicting_from_power_three_quarters_of_actual_tally_interaction_county_indicator,
                    bool_predicting_from_actual_tally_interaction_county_indicator =
                        bool_predicting_from_actual_tally_interaction_county_indicator,
                    bool_predicting_from_power_one_and_a_half_of_actual_tally_interaction_county_indicator =
                        bool_predicting_from_power_one_and_a_half_of_actual_tally_interaction_county_indicator,
                    bool_predicting_from_squared_actual_tally_interaction_county_indicator =
                        bool_predicting_from_squared_actual_tally_interaction_county_indicator,
                    #
                    bool_predicting_from_ln_of_incremented_benchmark_tally_interaction_county_indicator =
                        bool_predicting_from_ln_of_incremented_benchmark_tally_interaction_county_indicator,
                    bool_predicting_from_power_one_quarter_of_benchmark_tally_interaction_county_indicator =
                        bool_predicting_from_power_one_quarter_of_benchmark_tally_interaction_county_indicator,
                    bool_predicting_from_square_root_of_benchmark_tally_interaction_county_indicator =
                        bool_predicting_from_square_root_of_benchmark_tally_interaction_county_indicator,
                    bool_predicting_from_power_three_quarters_of_benchmark_tally_interaction_county_indicator =
                        bool_predicting_from_power_three_quarters_of_benchmark_tally_interaction_county_indicator,
                    bool_predicting_from_benchmark_tally_interaction_county_indicator =
                        bool_predicting_from_benchmark_tally_interaction_county_indicator,
                    bool_predicting_from_power_one_and_a_half_of_benchmark_tally_interaction_county_indicator =
                        bool_predicting_from_power_one_and_a_half_of_benchmark_tally_interaction_county_indicator,
                    bool_predicting_from_squared_benchmark_tally_interaction_county_indicator =
                        bool_predicting_from_squared_benchmark_tally_interaction_county_indicator,
                    #
                    bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator =
                        bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator,
                    bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator =
                        bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator,
                    bool_predicting_from_actual_tally_interaction_benchmark_tally_interaction_county_indicator =
                        bool_predicting_from_actual_tally_interaction_benchmark_tally_interaction_county_indicator,
                    )
                y = df_predicted_vote_counts_by_precinct[[str_y_name]]
                df_ols_coeffs_curr = estimate_linear_model_with_summary_in_dataframe(
                    str_y_name = str_y_name,
                    y = y,
                    X = X,
                    flt_lasso_alpha_l1_regularization_strength_term =
                        flt_lasso_alpha_l1_regularization_strength_term,
                    int_lasso_maximum_number_of_iterations =
                        int_lasso_maximum_number_of_iterations,
                    int_lasso_optimization_tolerance =
                        int_lasso_optimization_tolerance,
                    flt_lasso_cv_length_of_alphas_regularization_path =
                        flt_lasso_cv_length_of_alphas_regularization_path,
                    int_lasso_cv_num_candidate_alphas_on_regularization_path =
                        int_lasso_cv_num_candidate_alphas_on_regularization_path,
                    int_lasso_cv_maximum_number_of_iterations =
                        int_lasso_cv_maximum_number_of_iterations,
                    int_lasso_cv_optimization_tolerance =
                        int_lasso_cv_optimization_tolerance,
                    int_lasso_cv_number_of_folds_in_cross_validation =
                        int_lasso_cv_number_of_folds_in_cross_validation,
                    df_predicted_vote_counts_out =
                        df_predicted_vote_counts_by_precinct,
                    bool_estimate_model_parameters_diagnostics =
                        bool_estimate_model_parameters_diagnostics,
                    )
                if df_ols_coeffs is None :
                    df_ols_coeffs = df_ols_coeffs_curr
                else :
                    df_ols_coeffs = pd.concat(
                        [df_ols_coeffs, df_ols_coeffs_curr],
                        axis=0, ignore_index=True)

            # Final cleaning and rounding on the precinct level
            df_predicted_vote_counts_by_precinct = df_predicted_vote_counts_by_precinct.round(0)
            df_predicted_vote_counts_by_precinct[lst_str_predicted_for_actual_vote_count_column_names] = \
                df_predicted_vote_counts_by_precinct[
                    lst_str_predicted_for_actual_vote_count_column_names].astype("int64")
            if bool_approximate_predictions_for_missing_benchmark_precincts_by_actual_values :
                df_predicted_vote_counts_by_precinct = pd.concat([
                    df_predicted_vote_counts_by_precinct, df_actual_as_predicted_vote_counts_by_precinct],
                    ignore_index=True, axis=0)
            df_predicted_vote_counts = df_predicted_vote_counts_by_precinct

    if bool_distribute_job_output_files_in_directories_by_type or \
       bool_gather_all_job_files_in_one_directory :
        if bool_save_estimated_models_parameters_to_csv_file and \
           str_csv_file_name_for_saving_estimated_models_parameters is not None :
            str_csv_file_name_ext = \
                str_csv_file_name_for_saving_estimated_models_parameters + ".csv"
            str_csv_full_file_name_ext = os.path.join(
                STR_OUTPUT_PARAMS_PARM_CSV_PATH, str_csv_file_name_ext)
            try :
                if os.path.exists(str_csv_full_file_name_ext) :
                    os.remove(str_csv_full_file_name_ext)
            except Exception as exception :
                print_exception_message(exception)
            try :
                df_ols_coeffs.to_csv(str_csv_full_file_name_ext, index = False)
            except Exception as exception :
                print_exception_message(exception)

            if bool_gather_all_job_files_in_one_directory and \
               str_job_subdir_name is not None :
                str_csv_full_file_name_ext_from = str_csv_full_file_name_ext
                str_csv_full_file_name_ext_to = os.path.join(
                    str_job_subdir_name, str_csv_file_name_ext)
                try :
                    if bool_distribute_job_output_files_in_directories_by_type :
                        shutil.copy(
                            str_csv_full_file_name_ext_from,
                            str_csv_full_file_name_ext_to)
                    else :
                        shutil.move(
                            str_csv_full_file_name_ext_from,
                            str_csv_full_file_name_ext_to)
                        pass
                except Exception as exception :
                    print_exception_message(exception)

    return (df_actual_vote_counts, df_predicted_vote_counts)

###############################################################################

def compute_outlier_score_for_choice(
        k, M, n, N,
        bool_use_decimal_type = False,
        int_decimal_computational_precision = 1024, # [0; +Inf). # 1024 by default
        int_decimal_reporting_precision = 16, # [0; +Inf). # 16 by default
        int_max_num_iters_for_exact_hypergeom = 1_000_000, # [1; +Inf). # 1_000_000 by default
        int_max_num_iters_for_lanczos_approx_hypergeom = 500_000, # [1; +Inf). # 500_000 by default
        int_max_num_iters_for_spouge_approx_hypergeom = 500_000, # [1; +Inf). # 500_000 by default
        int_min_sample_size_for_approx_normal = 1_000, # [0; +Inf). # 1_000 by default
        flt_max_sample_sz_frac_of_pop_sz_for_approx_normal = 0.1, # [0.; 1.]. # 0.1 by default
        flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal = 0.1, # [0.; 1.]. # 0.1 by default
        flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal = 0.0, # [0.; +Inf). # 0.0 by default.
            # z = 2.575829303549 is for 99% in range mu +/- z*sigma
        bool_save_outlier_score_stats_to_csv_file = False,
        dict_lst_outlier_score_stats = None,
        int_id_1 = None,
        int_id_2 = None,
        ) :
    '''<function description>
    
    Rules:
        0 <= M <= M_MAX;
        0 <= N <= M;
        0 <= n <= M;
        0 <= k <= N <= M;
        0 <= k <= n <= M;
        0 <= (n - k) <= (M - N).
    
    <calling functions>
    <called functions>

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
    if bool_save_outlier_score_stats_to_csv_file :
        int_time_start_ns = time.monotonic_ns()
    if bool_use_decimal_type :
        if 0 < M and 0 <= N <= M and 0 <= n <= M and 0 <= k <= N and \
           0 <= (n - k) <= (M - N) : # (0 <= k <= n) is implied from others
            flt_outlier_score = custom_hypergeom_min_cdf_sf(
                k=k, M=M, n=n, N=N,
                bool_split_into_coeff_and_base_ten_exponent = False,
                bool_use_decimal_type =
                    bool_use_decimal_type,
                int_decimal_computational_precision =
                    int_decimal_computational_precision,
                int_decimal_reporting_precision =
                    int_decimal_reporting_precision,
                int_max_num_iters_for_exact_hypergeom = 
                    int_max_num_iters_for_exact_hypergeom,
                int_max_num_iters_for_lanczos_approx_hypergeom =
                    int_max_num_iters_for_lanczos_approx_hypergeom,
                int_max_num_iters_for_spouge_approx_hypergeom =
                    int_max_num_iters_for_spouge_approx_hypergeom,
                int_min_sample_size_for_approx_normal =
                    int_min_sample_size_for_approx_normal,
                flt_max_sample_sz_frac_of_pop_sz_for_approx_normal =
                    flt_max_sample_sz_frac_of_pop_sz_for_approx_normal,
                flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal =
                    flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal,
                flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal =
                    flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal,)
        else :
            flt_outlier_score = Decimal(0)
    else :
        if 0 < M and 0 <= N <= M and 0 <= n <= M and 0 <= k <= N and \
           0 <= (n - k) <= (M - N) : # (0 <= k <= n) is implied from others
            flt_outlier_score = scipy_hypergeom_min_cdf_sf(
                k=k,M=M,n=n,N=N,
                bool_split_into_coeff_and_base_ten_exponent = False,
                bool_use_decimal_type = bool_use_decimal_type,
                int_decimal_reporting_precision = int_decimal_reporting_precision,)
        else :
            flt_outlier_score = 0.
    if bool_save_outlier_score_stats_to_csv_file :
        int_time_end_ns = time.monotonic_ns()

    if bool_save_outlier_score_stats_to_csv_file :
        int_bm_time_start_ns = time.monotonic_ns()
    if 0 < M and 0 <= N <= M and 0 <= n <= M and 0 <= k <= N and \
       0 <= (n - k) <= (M - N) : # (0 <= k <= n) is implied from others
        flt_bm_outlier_score = scipy_hypergeom_min_cdf_sf(
            k=k,M=M,n=n,N=N,
            bool_split_into_coeff_and_base_ten_exponent = False,
            bool_use_decimal_type = False,
            int_decimal_reporting_precision = int_decimal_reporting_precision,)
    else :
        flt_bm_outlier_score = 0.
    if bool_save_outlier_score_stats_to_csv_file :
        int_bm_time_end_ns = time.monotonic_ns()

    if bool_save_outlier_score_stats_to_csv_file and \
       dict_lst_outlier_score_stats is not None :

        int_runtime_ms = int((int_time_end_ns - int_time_start_ns) /
                FLT_NANOSECONDS_PER_MILLISECOND)
        int_bm_runtime_ms = int((int_bm_time_end_ns - int_bm_time_start_ns) /
                FLT_NANOSECONDS_PER_MILLISECOND)

        flt_cdf = hypergeom.cdf(k=k, M=M, n=n, N=N)
        flt_sf = hypergeom.sf(k=k, M=M, n=n, N=N)
        flt_pmf = hypergeom.pmf(k=k, M=M, n=n, N=N)

        dict_lst_outlier_score_stats["id_1"].append(int_id_1)
        dict_lst_outlier_score_stats["id_2"].append(int_id_2)
        dict_lst_outlier_score_stats["M"].append(M)
        dict_lst_outlier_score_stats["N"].append(N)
        dict_lst_outlier_score_stats["n"].append(n)
        dict_lst_outlier_score_stats["k"].append(k)
        dict_lst_outlier_score_stats["runtime_ms"].append(int_runtime_ms)
        dict_lst_outlier_score_stats["bm_runtime_ms"].append(
            int_bm_runtime_ms)
        dict_lst_outlier_score_stats["outlier_score"].append(
            flt_outlier_score)
        dict_lst_outlier_score_stats["bm_outlier_score"].append(
            flt_bm_outlier_score)
        dict_lst_outlier_score_stats["bm_cdf"].append(flt_cdf)
        dict_lst_outlier_score_stats["bm_sf"].append(flt_sf)
        dict_lst_outlier_score_stats["bm_pmf"].append(flt_pmf)
    return flt_outlier_score

###############################################################################

def compute_each_with_rest_outlier_score_for_precinct(
        df_actual_and_predicted_vote_counts,
        lst_str_predicted_for_actual_vote_count_column_names,
        index_of_precinct_row,
        bool_use_decimal_type = False,
        int_decimal_computational_precision = 1024, # [0; +Inf). # 1024 by default
        int_decimal_reporting_precision = 16, # [0; +Inf). # 16 by default
        int_max_num_iters_for_exact_hypergeom = 1_000_000, # [1; +Inf). # 1_000_000 by default
        int_max_num_iters_for_lanczos_approx_hypergeom = 500_000, # [1; +Inf). # 500_000 by default
        int_max_num_iters_for_spouge_approx_hypergeom = 500_000, # [1; +Inf). # 500_000 by default
        int_min_sample_size_for_approx_normal = 1_000, # [0; +Inf). # 1_000 by default
        flt_max_sample_sz_frac_of_pop_sz_for_approx_normal = 0.1, # [0.; 1.]. # 0.1 by default
        flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal = 0.1, # [0.; 1.]. # 0.1 by default
        flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal = 0.0, # [0.; +Inf). # 0.0 by default.
            # z = 2.575829303549 is for 99% in range mu +/- z*sigma
        bool_compute_outlier_score_level_1 = True,
        bool_compute_outlier_score_level_2 = True,
        bool_compute_outlier_score_level_3 = True,
        bool_save_outlier_score_stats_to_csv_file = False,
        dict_lst_outlier_score_stats = None,
        int_id_1 = None,
        ) :
    '''<function description>
    
    <calling functions>
    <called functions>

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
    if bool_use_decimal_type :
        one = Decimal(1)
        flt_outlier_score_level_1_for_precinct = one
        flt_outlier_score_level_2_for_precinct = one
        flt_outlier_score_level_3_for_precinct = one
    else :
        flt_outlier_score_level_1_for_precinct = 1.
        flt_outlier_score_level_2_for_precinct = 1.
        flt_outlier_score_level_3_for_precinct = 1.

    if df_actual_and_predicted_vote_counts.loc[index_of_precinct_row, STR_IS_AUDITED_COLUMN_NAME] == 0 :

        if bool_compute_outlier_score_level_1 :
            # N is the expectation of "actual" counts in the population based on the "predicted" counts.
            N = sum([df_actual_and_predicted_vote_counts.loc[index_of_precinct_row, str_votes_counts_columns_name + "_pre"]
                     for str_votes_counts_columns_name in lst_str_predicted_for_actual_vote_count_column_names])
            # M is the expectation of both "actual" and "predicted" counts in the population based on the "predicted" counts.
            M = N * 2
            for str_votes_counts_columns_name in lst_str_predicted_for_actual_vote_count_column_names :
                # n is the expectation of both "actual" and "predicted" counts in the sample based on the "predicted" counts.
                n = df_actual_and_predicted_vote_counts.loc[index_of_precinct_row, str_votes_counts_columns_name + "_pre"] * 2
                # k is the observed "actual" count in the sample.
                k = df_actual_and_predicted_vote_counts.loc[index_of_precinct_row, str_votes_counts_columns_name]
                flt_outlier_score_for_choice = compute_outlier_score_for_choice(
                    k=k, M=M, n=n, N=N,
                    bool_use_decimal_type =
                        bool_use_decimal_type,
                    int_decimal_computational_precision =
                        int_decimal_computational_precision,
                    int_decimal_reporting_precision =
                        int_decimal_reporting_precision,
                    int_max_num_iters_for_exact_hypergeom =
                        int_max_num_iters_for_exact_hypergeom,
                    int_max_num_iters_for_lanczos_approx_hypergeom =
                        int_max_num_iters_for_lanczos_approx_hypergeom,
                    int_max_num_iters_for_spouge_approx_hypergeom =
                        int_max_num_iters_for_spouge_approx_hypergeom,
                    int_min_sample_size_for_approx_normal =
                        int_min_sample_size_for_approx_normal,
                    flt_max_sample_sz_frac_of_pop_sz_for_approx_normal =
                        flt_max_sample_sz_frac_of_pop_sz_for_approx_normal,
                    flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal =
                        flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal,
                    flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal =
                        flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal,
                    bool_save_outlier_score_stats_to_csv_file =
                        bool_save_outlier_score_stats_to_csv_file,
                    dict_lst_outlier_score_stats = dict_lst_outlier_score_stats,
                    int_id_1 = int_id_1,
                    int_id_2 = 1,)
                if flt_outlier_score_level_1_for_precinct > flt_outlier_score_for_choice :
                    flt_outlier_score_level_1_for_precinct = flt_outlier_score_for_choice

        if bool_compute_outlier_score_level_2 :
            # M is the observed count of both "actual" and "predicted" counts in the population.
            M = sum([df_actual_and_predicted_vote_counts.loc[index_of_precinct_row, str_votes_counts_columns_name + "_sum"]
                for str_votes_counts_columns_name in lst_str_predicted_for_actual_vote_count_column_names])
            # N is the expectation of "actual" counts in the population based on both "actual" and "predicted" counts.
            N = int(round(M / 2.))
            for str_votes_counts_columns_name in lst_str_predicted_for_actual_vote_count_column_names :
                # n is the observed count of both "actual" + "predicted" in the sample.
                n = df_actual_and_predicted_vote_counts.loc[index_of_precinct_row, str_votes_counts_columns_name] + \
                    df_actual_and_predicted_vote_counts.loc[index_of_precinct_row, str_votes_counts_columns_name + "_pre"]
                # k is the observed "actual" count in the sample.
                k = df_actual_and_predicted_vote_counts.loc[index_of_precinct_row, str_votes_counts_columns_name]
                flt_outlier_score_for_choice = compute_outlier_score_for_choice(
                    k=k, M=M, n=n, N=N,
                    bool_use_decimal_type =
                        bool_use_decimal_type,
                    int_decimal_computational_precision =
                        int_decimal_computational_precision,
                    int_decimal_reporting_precision =
                        int_decimal_reporting_precision,
                    int_max_num_iters_for_exact_hypergeom =
                        int_max_num_iters_for_exact_hypergeom,
                    int_max_num_iters_for_lanczos_approx_hypergeom =
                        int_max_num_iters_for_lanczos_approx_hypergeom,
                    int_max_num_iters_for_spouge_approx_hypergeom =
                        int_max_num_iters_for_spouge_approx_hypergeom,
                    int_min_sample_size_for_approx_normal =
                        int_min_sample_size_for_approx_normal,
                    flt_max_sample_sz_frac_of_pop_sz_for_approx_normal =
                        flt_max_sample_sz_frac_of_pop_sz_for_approx_normal,
                    flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal =
                        flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal,
                    flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal =
                        flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal,
                    bool_save_outlier_score_stats_to_csv_file =
                        bool_save_outlier_score_stats_to_csv_file,
                    dict_lst_outlier_score_stats = dict_lst_outlier_score_stats,
                    int_id_1 = int_id_1,
                    int_id_2 = 2,)
                if flt_outlier_score_level_2_for_precinct > flt_outlier_score_for_choice :
                    flt_outlier_score_level_2_for_precinct = flt_outlier_score_for_choice

        if bool_compute_outlier_score_level_3 :
            # M is the observed count of both "actual" and "predicted" counts in the population.
            M = sum([df_actual_and_predicted_vote_counts.loc[index_of_precinct_row, str_votes_counts_columns_name + "_sum"]
                for str_votes_counts_columns_name in lst_str_predicted_for_actual_vote_count_column_names])
            # N is the observed count of "actual" counts in the population based 
            N = sum([df_actual_and_predicted_vote_counts.loc[index_of_precinct_row, str_votes_counts_columns_name]
                     for str_votes_counts_columns_name in lst_str_predicted_for_actual_vote_count_column_names])
            for str_votes_counts_columns_name in lst_str_predicted_for_actual_vote_count_column_names :
                # n is the observed count of both "actual" + "predicted" in the sample.
                n = df_actual_and_predicted_vote_counts.loc[index_of_precinct_row, str_votes_counts_columns_name] + \
                    df_actual_and_predicted_vote_counts.loc[index_of_precinct_row, str_votes_counts_columns_name + "_pre"]
                # k is the observed "actual" count in the sample.
                k = df_actual_and_predicted_vote_counts.loc[index_of_precinct_row, str_votes_counts_columns_name]
                flt_outlier_score_for_choice = compute_outlier_score_for_choice(
                    k=k, M=M, n=n, N=N,
                    bool_use_decimal_type =
                        bool_use_decimal_type,
                    int_decimal_computational_precision =
                        int_decimal_computational_precision,
                    int_decimal_reporting_precision =
                        int_decimal_reporting_precision,
                    int_max_num_iters_for_exact_hypergeom =
                        int_max_num_iters_for_exact_hypergeom,
                    int_max_num_iters_for_lanczos_approx_hypergeom =
                        int_max_num_iters_for_lanczos_approx_hypergeom,
                    int_max_num_iters_for_spouge_approx_hypergeom =
                        int_max_num_iters_for_spouge_approx_hypergeom,
                    int_min_sample_size_for_approx_normal =
                        int_min_sample_size_for_approx_normal,
                    flt_max_sample_sz_frac_of_pop_sz_for_approx_normal =
                        flt_max_sample_sz_frac_of_pop_sz_for_approx_normal,
                    flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal =
                        flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal,
                    flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal =
                        flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal,
                    bool_save_outlier_score_stats_to_csv_file =
                        bool_save_outlier_score_stats_to_csv_file,
                    dict_lst_outlier_score_stats = dict_lst_outlier_score_stats,
                    int_id_1 = int_id_1,
                    int_id_2 = 3,)
                if flt_outlier_score_level_3_for_precinct > flt_outlier_score_for_choice :
                    flt_outlier_score_level_3_for_precinct = flt_outlier_score_for_choice

    return (flt_outlier_score_level_1_for_precinct,
            flt_outlier_score_level_2_for_precinct,
            flt_outlier_score_level_3_for_precinct,)

###############################################################################

def compute_each_with_each_outlier_score_for_precinct(
        df_actual_and_predicted_vote_counts,
        lst_str_predicted_for_actual_vote_count_column_names,
        index_of_precinct_row,
        bool_use_decimal_type = False,
        int_decimal_computational_precision = 1024, # [0; +Inf). # 1024 by default
        int_decimal_reporting_precision = 16, # [0; +Inf). # 16 by default
        int_max_num_iters_for_exact_hypergeom = 1_000_000, # [1; +Inf). # 1_000_000 by default
        int_max_num_iters_for_lanczos_approx_hypergeom = 500_000, # [1; +Inf). # 500_000 by default
        int_max_num_iters_for_spouge_approx_hypergeom = 500_000, # [1; +Inf). # 500_000 by default
        int_min_sample_size_for_approx_normal = 1_000, # [0; +Inf). # 1_000 by default
        flt_max_sample_sz_frac_of_pop_sz_for_approx_normal = 0.1, # [0.; 1.]. # 0.1 by default
        flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal = 0.1, # [0.; 1.]. # 0.1 by default
        flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal = 0.0, # [0.; +Inf). # 0.0 by default.
            # z = 2.575829303549 is for 99% in range mu +/- z*sigma
        bool_compute_outlier_score_level_1 = True,
        bool_compute_outlier_score_level_2 = True,
        bool_compute_outlier_score_level_3 = True,
        bool_save_outlier_score_stats_to_csv_file = False,
        dict_lst_outlier_score_stats = None,
        int_id_1 = None,
        ) :
    '''<function description>
    
    <calling functions>
    <called functions>

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
    if bool_use_decimal_type :
        one = Decimal(1)
        flt_outlier_score_level_1_for_precinct = one
        flt_outlier_score_level_2_for_precinct = one
        flt_outlier_score_level_3_for_precinct = one
    else :
        flt_outlier_score_level_1_for_precinct = 1.
        flt_outlier_score_level_2_for_precinct = 1.
        flt_outlier_score_level_3_for_precinct = 1.

    if df_actual_and_predicted_vote_counts.loc[index_of_precinct_row, STR_IS_AUDITED_COLUMN_NAME] == 0 :

        if bool_compute_outlier_score_level_1 :
            for int_votes_counts_columns_index1 in range(len(lst_str_predicted_for_actual_vote_count_column_names)) :
                str_votes_counts_columns_name1 = lst_str_predicted_for_actual_vote_count_column_names[
                    int_votes_counts_columns_index1]
                # n is the expectation of both "actual" and "predicted" counts in the sample based on the "predicted" counts.
                n = df_actual_and_predicted_vote_counts.loc[
                    index_of_precinct_row, str_votes_counts_columns_name1 + "_pre"] * 2
                # k is the observed "actual" count in the sample.
                k = df_actual_and_predicted_vote_counts.loc[
                    index_of_precinct_row, str_votes_counts_columns_name1]
                for int_votes_counts_columns_index2 in range(len(lst_str_predicted_for_actual_vote_count_column_names)) :
                    if int_votes_counts_columns_index1 != int_votes_counts_columns_index2 :
                        str_votes_counts_columns_name2 = lst_str_predicted_for_actual_vote_count_column_names[
                            int_votes_counts_columns_index2]
                        # N is the expectation of "actual" counts in the population based on the "predicted" counts.
                        N = sum([df_actual_and_predicted_vote_counts.loc[
                            index_of_precinct_row, str_votes_counts_columns_name + "_pre"]
                                 for str_votes_counts_columns_name in
                                 [str_votes_counts_columns_name1, str_votes_counts_columns_name2]])
                        # M is the expectation of both "actual" and "predicted" counts in the population based on the
                        # "predicted" counts.
                        M = N * 2
                        flt_outlier_score_for_choice = compute_outlier_score_for_choice(
                            k=k, M=M, n=n, N=N,
                            bool_use_decimal_type =
                                bool_use_decimal_type,
                            int_decimal_computational_precision =
                                int_decimal_computational_precision,
                            int_decimal_reporting_precision =
                                int_decimal_reporting_precision,
                            int_max_num_iters_for_exact_hypergeom =
                                int_max_num_iters_for_exact_hypergeom,
                            int_max_num_iters_for_lanczos_approx_hypergeom =
                                int_max_num_iters_for_lanczos_approx_hypergeom,
                            int_max_num_iters_for_spouge_approx_hypergeom =
                                int_max_num_iters_for_spouge_approx_hypergeom,
                            int_min_sample_size_for_approx_normal =
                                int_min_sample_size_for_approx_normal,
                            flt_max_sample_sz_frac_of_pop_sz_for_approx_normal =
                                flt_max_sample_sz_frac_of_pop_sz_for_approx_normal,
                            flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal =
                                flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal,
                            flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal =
                                flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal,
                            bool_save_outlier_score_stats_to_csv_file =
                                bool_save_outlier_score_stats_to_csv_file,
                            dict_lst_outlier_score_stats = dict_lst_outlier_score_stats,
                            int_id_1 = int_id_1,
                            int_id_2 = 1,)
                        if flt_outlier_score_level_1_for_precinct > flt_outlier_score_for_choice :
                            flt_outlier_score_level_1_for_precinct = flt_outlier_score_for_choice

        if bool_compute_outlier_score_level_2 :
            for int_votes_counts_columns_index1 in range(len(lst_str_predicted_for_actual_vote_count_column_names)) :
                str_votes_counts_columns_name1 = lst_str_predicted_for_actual_vote_count_column_names[
                    int_votes_counts_columns_index1]
                # n is the observed count of both "actual" + "predicted" in the sample.
                n = df_actual_and_predicted_vote_counts.loc[index_of_precinct_row, str_votes_counts_columns_name1] + \
                    df_actual_and_predicted_vote_counts.loc[index_of_precinct_row, str_votes_counts_columns_name1 + "_pre"]
                # k is the observed "actual" count in the sample.
                k = df_actual_and_predicted_vote_counts.loc[index_of_precinct_row, str_votes_counts_columns_name1]
                for int_votes_counts_columns_index2 in range(len(lst_str_predicted_for_actual_vote_count_column_names)) :
                    if int_votes_counts_columns_index1 != int_votes_counts_columns_index2 :
                        str_votes_counts_columns_name2 = lst_str_predicted_for_actual_vote_count_column_names[
                            int_votes_counts_columns_index2]

                        # M is the observed count of both "actual" and "predicted" counts in the population.
                        M = sum([df_actual_and_predicted_vote_counts.loc[
                            index_of_precinct_row, str_votes_counts_columns_name + "_sum"]
                                for str_votes_counts_columns_name in
                                [str_votes_counts_columns_name1, str_votes_counts_columns_name2]])
                        # N is the expectation of "actual" counts in the population based on both "actual" and "predicted" counts.
                        N = int(round(M / 2.))
                        flt_outlier_score_for_choice = compute_outlier_score_for_choice(
                            k=k, M=M, n=n, N=N,
                            bool_use_decimal_type =
                                bool_use_decimal_type,
                            int_decimal_computational_precision =
                                int_decimal_computational_precision,
                            int_decimal_reporting_precision =
                                int_decimal_reporting_precision,
                            int_max_num_iters_for_exact_hypergeom =
                                int_max_num_iters_for_exact_hypergeom,
                            int_max_num_iters_for_lanczos_approx_hypergeom =
                                int_max_num_iters_for_lanczos_approx_hypergeom,
                            int_max_num_iters_for_spouge_approx_hypergeom =
                                int_max_num_iters_for_spouge_approx_hypergeom,
                            int_min_sample_size_for_approx_normal =
                                int_min_sample_size_for_approx_normal,
                            flt_max_sample_sz_frac_of_pop_sz_for_approx_normal =
                                flt_max_sample_sz_frac_of_pop_sz_for_approx_normal,
                            flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal =
                                flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal,
                            flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal =
                                flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal,
                            bool_save_outlier_score_stats_to_csv_file =
                                bool_save_outlier_score_stats_to_csv_file,
                            dict_lst_outlier_score_stats = dict_lst_outlier_score_stats,
                            int_id_1 = int_id_1,
                            int_id_2 = 2,)
                        if flt_outlier_score_level_2_for_precinct > flt_outlier_score_for_choice :
                            flt_outlier_score_level_2_for_precinct = flt_outlier_score_for_choice

        if bool_compute_outlier_score_level_3 :
            for int_votes_counts_columns_index1 in range(len(lst_str_predicted_for_actual_vote_count_column_names)) :
                str_votes_counts_columns_name1 = lst_str_predicted_for_actual_vote_count_column_names[
                    int_votes_counts_columns_index1]
                # n is the observed count of both "actual" + "predicted" in the sample.
                n = df_actual_and_predicted_vote_counts.loc[index_of_precinct_row, str_votes_counts_columns_name1] + \
                    df_actual_and_predicted_vote_counts.loc[index_of_precinct_row, str_votes_counts_columns_name1 + "_pre"]
                # k is the observed "actual" count in the sample.
                k = df_actual_and_predicted_vote_counts.loc[index_of_precinct_row, str_votes_counts_columns_name1]
                for int_votes_counts_columns_index2 in range(len(lst_str_predicted_for_actual_vote_count_column_names)) :
                    if int_votes_counts_columns_index1 != int_votes_counts_columns_index2 :
                        str_votes_counts_columns_name2 = lst_str_predicted_for_actual_vote_count_column_names[
                            int_votes_counts_columns_index2]

                        # M is the observed count of both "actual" and "predicted" counts in the population.
                        M = sum([df_actual_and_predicted_vote_counts.loc[
                            index_of_precinct_row, str_votes_counts_columns_name + "_sum"]
                                for str_votes_counts_columns_name in
                                [str_votes_counts_columns_name1, str_votes_counts_columns_name2]])
                        # N is the observed count of "actual" counts in the population based 
                        N = sum([df_actual_and_predicted_vote_counts.loc[
                            index_of_precinct_row, str_votes_counts_columns_name]
                                 for str_votes_counts_columns_name in
                                 [str_votes_counts_columns_name1, str_votes_counts_columns_name2]])
                        flt_outlier_score_for_choice = compute_outlier_score_for_choice(
                            k=k, M=M, n=n, N=N,
                            bool_use_decimal_type =
                                bool_use_decimal_type,
                            int_decimal_computational_precision =
                                int_decimal_computational_precision,
                            int_decimal_reporting_precision =
                                int_decimal_reporting_precision,
                            int_max_num_iters_for_exact_hypergeom =
                                int_max_num_iters_for_exact_hypergeom,
                            int_max_num_iters_for_lanczos_approx_hypergeom =
                                int_max_num_iters_for_lanczos_approx_hypergeom,
                            int_max_num_iters_for_spouge_approx_hypergeom =
                                int_max_num_iters_for_spouge_approx_hypergeom,
                            int_min_sample_size_for_approx_normal =
                                int_min_sample_size_for_approx_normal,
                            flt_max_sample_sz_frac_of_pop_sz_for_approx_normal =
                                flt_max_sample_sz_frac_of_pop_sz_for_approx_normal,
                            flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal =
                                flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal,
                            flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal =
                                flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal,
                            bool_save_outlier_score_stats_to_csv_file =
                                bool_save_outlier_score_stats_to_csv_file,
                            dict_lst_outlier_score_stats = dict_lst_outlier_score_stats,
                            int_id_1 = int_id_1,
                            int_id_2 = 3,)
                        if flt_outlier_score_level_3_for_precinct > flt_outlier_score_for_choice :
                            flt_outlier_score_level_3_for_precinct = flt_outlier_score_for_choice

    return (flt_outlier_score_level_1_for_precinct,
            flt_outlier_score_level_2_for_precinct,
            flt_outlier_score_level_3_for_precinct)

###############################################################################

def compute_outlier_score_for_precinct(
        df_actual_and_predicted_vote_counts,
        lst_str_predicted_for_actual_vote_count_column_names,
        index_of_precinct_row,
        bool_use_decimal_type = False,
        int_decimal_computational_precision = 1024, # [0; +Inf). # 1024 by default
        int_decimal_reporting_precision = 16, # [0; +Inf). # 16 by default
        int_max_num_iters_for_exact_hypergeom = 1_000_000, # [1; +Inf). # 1_000_000 by default
        int_max_num_iters_for_lanczos_approx_hypergeom = 500_000, # [1; +Inf). # 500_000 by default
        int_max_num_iters_for_spouge_approx_hypergeom = 500_000, # [1; +Inf). # 500_000 by default
        int_min_sample_size_for_approx_normal = 1_000, # [0; +Inf). # 1_000 by default
        flt_max_sample_sz_frac_of_pop_sz_for_approx_normal = 0.1, # [0.; 1.]. # 0.1 by default
        flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal = 0.1, # [0.; 1.]. # 0.1 by default
        flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal = 0.0, # [0.; +Inf). # 0.0 by default.
            # z = 2.575829303549 is for 99% in range mu +/- z*sigma
        bool_true_each_choice_with_all_other_choices__false_each_choice_with_each_choice = True,
        bool_compute_outlier_score_level_1 = True,
        bool_compute_outlier_score_level_2 = True,
        bool_compute_outlier_score_level_3 = True,
        bool_save_outlier_score_stats_to_csv_file = False,
        dict_lst_outlier_score_stats = None,
        int_id_1 = None,
        ) :
    '''<function description>
    
    <calling functions>
    <called functions>

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
    if bool_true_each_choice_with_all_other_choices__false_each_choice_with_each_choice :
        if bool_compute_outlier_score_level_1 or \
           bool_compute_outlier_score_level_2 or \
           bool_compute_outlier_score_level_3 :
            (flt_outlier_score_level_1_for_precinct,
             flt_outlier_score_level_2_for_precinct,
             flt_outlier_score_level_3_for_precinct) = \
                compute_each_with_rest_outlier_score_for_precinct(
                    df_actual_and_predicted_vote_counts =
                        df_actual_and_predicted_vote_counts,
                    lst_str_predicted_for_actual_vote_count_column_names =
                        lst_str_predicted_for_actual_vote_count_column_names,
                    index_of_precinct_row = index_of_precinct_row,
                    bool_use_decimal_type =
                        bool_use_decimal_type,
                    int_decimal_computational_precision =
                        int_decimal_computational_precision,
                    int_decimal_reporting_precision =
                        int_decimal_reporting_precision,
                    int_max_num_iters_for_exact_hypergeom =
                        int_max_num_iters_for_exact_hypergeom,
                    int_max_num_iters_for_lanczos_approx_hypergeom =
                        int_max_num_iters_for_lanczos_approx_hypergeom,
                    int_max_num_iters_for_spouge_approx_hypergeom =
                        int_max_num_iters_for_spouge_approx_hypergeom,
                    int_min_sample_size_for_approx_normal =
                        int_min_sample_size_for_approx_normal,
                    flt_max_sample_sz_frac_of_pop_sz_for_approx_normal =
                        flt_max_sample_sz_frac_of_pop_sz_for_approx_normal,
                    flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal =
                        flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal,
                    flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal =
                        flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal,
                    bool_compute_outlier_score_level_1 =
                        bool_compute_outlier_score_level_1,
                    bool_compute_outlier_score_level_2 =
                        bool_compute_outlier_score_level_2,
                    bool_compute_outlier_score_level_3 =
                        bool_compute_outlier_score_level_3,
                    bool_save_outlier_score_stats_to_csv_file =
                        bool_save_outlier_score_stats_to_csv_file,
                    dict_lst_outlier_score_stats = dict_lst_outlier_score_stats,
                    int_id_1 = int_id_1,)
        else :
            if bool_use_decimal_type :
                one = Decimal(1)
                (flt_outlier_score_level_1_for_precinct,
                 flt_outlier_score_level_2_for_precinct,
                 flt_outlier_score_level_3_for_precinct) = (one,one,one)
            else :
                (flt_outlier_score_level_1_for_precinct,
                 flt_outlier_score_level_2_for_precinct,
                 flt_outlier_score_level_3_for_precinct) = (1.,1.,1.)
    else :
        if bool_compute_outlier_score_level_1 or \
           bool_compute_outlier_score_level_2 or \
           bool_compute_outlier_score_level_3 :
            (flt_outlier_score_level_1_for_precinct,
             flt_outlier_score_level_2_for_precinct,
             flt_outlier_score_level_3_for_precinct) = \
                compute_each_with_each_outlier_score_for_precinct(
                    df_actual_and_predicted_vote_counts =
                        df_actual_and_predicted_vote_counts,
                    lst_str_predicted_for_actual_vote_count_column_names =
                        lst_str_predicted_for_actual_vote_count_column_names,
                    index_of_precinct_row = index_of_precinct_row,
                    bool_use_decimal_type =
                        bool_use_decimal_type,
                    int_decimal_computational_precision =
                        int_decimal_computational_precision,
                    int_decimal_reporting_precision =
                        int_decimal_reporting_precision,
                    int_max_num_iters_for_exact_hypergeom =
                        int_max_num_iters_for_exact_hypergeom,
                    int_max_num_iters_for_lanczos_approx_hypergeom =
                        int_max_num_iters_for_lanczos_approx_hypergeom,
                    int_max_num_iters_for_spouge_approx_hypergeom =
                        int_max_num_iters_for_spouge_approx_hypergeom,
                    int_min_sample_size_for_approx_normal =
                        int_min_sample_size_for_approx_normal,
                    flt_max_sample_sz_frac_of_pop_sz_for_approx_normal =
                        flt_max_sample_sz_frac_of_pop_sz_for_approx_normal,
                    flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal =
                        flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal,
                    flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal =
                        flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal,
                    bool_compute_outlier_score_level_1 =
                        bool_compute_outlier_score_level_1,
                    bool_compute_outlier_score_level_2 =
                        bool_compute_outlier_score_level_2,
                    bool_compute_outlier_score_level_3 =
                        bool_compute_outlier_score_level_3,
                    bool_save_outlier_score_stats_to_csv_file =
                        bool_save_outlier_score_stats_to_csv_file,
                    dict_lst_outlier_score_stats = dict_lst_outlier_score_stats,
                    int_id_1 = int_id_1,)
        else :
            if bool_use_decimal_type :
                one = Decimal(1)
                (flt_outlier_score_level_1_for_precinct,
                 flt_outlier_score_level_2_for_precinct,
                 flt_outlier_score_level_3_for_precinct) = (one,one,one)
            else :
                (flt_outlier_score_level_1_for_precinct,
                 flt_outlier_score_level_2_for_precinct,
                 flt_outlier_score_level_3_for_precinct) = (1.,1.,1.)
    return (flt_outlier_score_level_1_for_precinct,
            flt_outlier_score_level_2_for_precinct,
            flt_outlier_score_level_3_for_precinct)

###############################################################################

def save_model_diagnostics_to_csv_file(
        df_actual_vote_counts,
        df_predicted_vote_counts,
        lst_str_predicted_for_actual_vote_count_column_names,
        str_csv_file_name_for_saving_model_diagnostics,
        bool_distribute_job_output_files_in_directories_by_type = True,
        bool_gather_all_job_files_in_one_directory = False,
        str_job_subdir_name = None,
        ) :
    '''<function description>
    Do not set predicted counts to the actual counts for the audited precincts.
    The model diagnostics are computed across all precincts, and the predicted
    values are used from all precincts, including the audited ones.
    https://scikit-learn.org/stable/modules/linear_model.html#using-cross-validation
    https://scikit-learn.org/stable/modules/model_evaluation.html
    https://scikit-learn.org/stable/modules/cross_validation.html
    https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-logarithmic-error
    https://scikit-learn.org/stable/modules/model_evaluation.html#mean-poisson-gamma-and-tweedie-deviances
    
    <calling functions>
    <called functions>

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

    dict_lst_model_diagnostics = {}
    #
    dict_lst_model_diagnostics["vote_count_name"] = []
    #
    # metrics.r2_score
    dict_lst_model_diagnostics["r2_score"] = []
    # metrics.d2_absolute_error_score
    dict_lst_model_diagnostics["d2_absolute_error_score"] = []
    #
    # metrics.mean_absolute_error
    dict_lst_model_diagnostics["mean_absolute_error"] = []
    # metrics.median_absolute_error
    dict_lst_model_diagnostics["median_absolute_error"] = []
    #
    # The higher power the less weight is given to extreme deviations between
    # true and predicted targets.
    # metrics.root_mean_squared_error
    # sklearn.metrics.mean_tweedie_deviance with p = 0.
    dict_lst_model_diagnostics["root_mean_squared_error"] = []
    # metrics.mean_poisson_deviance
    # sklearn.metrics.mean_tweedie_deviance p = 1.
    dict_lst_model_diagnostics["mean_poisson_deviance"] = []
    # metrics.mean_gamma_deviance
    # sklearn.metrics.mean_tweedie_deviance p = 2.
    dict_lst_model_diagnostics["mean_gamma_deviance"] = []
    #
    # This metric penalizes an under-predicted estimate greater than an
    # over-predicted estimate. This metric is best to use when targets having
    # exponential growth, such as population counts, ...
    # metrics.root_mean_squared_log_error
    dict_lst_model_diagnostics["root_mean_squared_log_error"] = []
    #
    # The idea of this metric is to be sensitive to relative errors.
    # It is for example not changed by a global scaling of the target variable.
    # metrics.mean_absolute_percentage_error
    dict_lst_model_diagnostics["mean_absolute_percentage_error"] = []

    for str_vote_count_column_name in lst_str_predicted_for_actual_vote_count_column_names :

        arr_actual = df_actual_vote_counts.loc[:,str_vote_count_column_name].values #.tolist()
        arr_predicted = df_predicted_vote_counts.loc[:,str_vote_count_column_name].values #.tolist()

        flt_r2_score = r2_score(arr_actual, arr_predicted)
        flt_d2_absolute_error_score = d2_absolute_error_score(arr_actual, arr_predicted)
        flt_mean_absolute_error = mean_absolute_error(arr_actual, arr_predicted)
        flt_median_absolute_error = median_absolute_error(arr_actual, arr_predicted)
        # mean_tweedie_deviance, # with p = 0:
        flt_root_mean_squared_error = root_mean_squared_error(arr_actual, arr_predicted)
        # mean_tweedie_deviance, # with p = 1:
        # Mean Tweedie deviance error with power=1 can only be used on non-negative
        # y and strictly positive y_pred.
        try :
            flt_mean_poisson_deviance = mean_poisson_deviance(arr_actual+1., arr_predicted+1.)
        except ValueError :
            flt_mean_poisson_deviance = float("Nan")
        # mean_tweedie_deviance, # with p = 2:
        # Mean Tweedie deviance error with power=2 can only be used on strictly positive y and y_pred.
        try :
            flt_mean_gamma_deviance = mean_gamma_deviance(arr_actual+1, arr_predicted+1)
        except ValueError :
            flt_mean_gamma_deviance = float("Nan")
        flt_root_mean_squared_log_error = root_mean_squared_log_error(arr_actual, arr_predicted)
        flt_mean_absolute_percentage_error = mean_absolute_percentage_error(arr_actual, arr_predicted)

        dict_lst_model_diagnostics["vote_count_name"].append(str_vote_count_column_name)
        dict_lst_model_diagnostics["r2_score"].append(flt_r2_score)
        dict_lst_model_diagnostics["d2_absolute_error_score"].append(flt_d2_absolute_error_score)
        dict_lst_model_diagnostics["mean_absolute_error"].append(flt_mean_absolute_error)
        dict_lst_model_diagnostics["median_absolute_error"].append(flt_median_absolute_error)
        dict_lst_model_diagnostics["root_mean_squared_error"].append(flt_root_mean_squared_error)
        dict_lst_model_diagnostics["mean_poisson_deviance"].append(flt_mean_poisson_deviance)
        dict_lst_model_diagnostics["mean_gamma_deviance"].append(flt_mean_gamma_deviance)
        dict_lst_model_diagnostics["root_mean_squared_log_error"].append(flt_root_mean_squared_log_error)
        dict_lst_model_diagnostics["mean_absolute_percentage_error"].append(flt_mean_absolute_percentage_error)

    df_lst_model_diagnostics = pd.DataFrame.from_dict(dict_lst_model_diagnostics)

    if bool_distribute_job_output_files_in_directories_by_type or \
       bool_gather_all_job_files_in_one_directory :
        if str_csv_file_name_for_saving_model_diagnostics is not None :
            str_csv_file_name_ext = \
                str_csv_file_name_for_saving_model_diagnostics + ".csv"
            str_csv_full_file_name_ext = os.path.join(
                STR_OUTPUT_DIAGN_PARM_CSV_PATH, str_csv_file_name_ext)
            try :
                if os.path.exists(str_csv_full_file_name_ext) :
                    os.remove(str_csv_full_file_name_ext)
            except Exception as exception :
                print_exception_message(exception)
            try :
                df_lst_model_diagnostics.to_csv(
                    str_csv_full_file_name_ext, index = False)
            except Exception as exception :
                print_exception_message(exception)

            if bool_gather_all_job_files_in_one_directory and \
               str_job_subdir_name is not None :
                str_csv_full_file_name_ext_from = str_csv_full_file_name_ext
                str_csv_full_file_name_ext_to = os.path.join(
                    str_job_subdir_name, str_csv_file_name_ext)
                try :
                    if bool_distribute_job_output_files_in_directories_by_type :
                        shutil.copy(
                            str_csv_full_file_name_ext_from,
                            str_csv_full_file_name_ext_to)
                    else :
                        shutil.move(
                            str_csv_full_file_name_ext_from,
                            str_csv_full_file_name_ext_to)
                        pass
                except Exception as exception :
                    print_exception_message(exception)

###############################################################################

def sort_precincts_by_decreasing_predictability_for_parametric_model(
        df_actual_vote_counts,
        df_predicted_vote_counts,
        lst_str_predicted_for_actual_vote_count_column_names,
        int_random_number_generator_seed_for_sorting = 0,
        bool_use_decimal_type = False,
        int_decimal_computational_precision = 1024, # [0; +Inf). # 1024 by default
        int_decimal_reporting_precision = 16, # [0; +Inf). # 16 by default
        int_max_num_iters_for_exact_hypergeom = 1_000_000, # [1; +Inf). # 1_000_000 by default
        int_max_num_iters_for_lanczos_approx_hypergeom = 500_000, # [1; +Inf). # 500_000 by default
        int_max_num_iters_for_spouge_approx_hypergeom = 500_000, # [1; +Inf). # 500_000 by default
        int_min_sample_size_for_approx_normal = 1_000, # [0; +Inf). # 1_000 by default
        flt_max_sample_sz_frac_of_pop_sz_for_approx_normal = 0.1, # [0.; 1.]. # 0.1 by default
        flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal = 0.1, # [0.; 1.]. # 0.1 by default
        flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal = 0.0, # [0.; +Inf). # 0.0 by default.
            # z = 2.575829303549 is for 99% in range mu +/- z*sigma
        bool_true_each_choice_with_all_other_choices__false_each_choice_with_each_choice = True,
        bool_compute_outlier_score_level_1 = True,
        bool_compute_outlier_score_level_2 = True,
        bool_compute_outlier_score_level_3 = True,
        bool_retain_predicted_vote_counts_columns_in_sorted_precincts_file = False,
        bool_retain_sorting_criteria_columns = False,
        bool_compute_cumulative_actual_vote_count_fractions = False,
        bool_save_sorted_precincts_to_csv_file = False,
        str_csv_file_name_for_saving_sorted_precincts = None,
        bool_save_outlier_score_stats_to_csv_file = False,
        str_csv_file_name_for_saving_outlier_score_stats = None,
        bool_distribute_job_output_files_in_directories_by_type = True,
        bool_gather_all_job_files_in_one_directory = False,
        str_job_subdir_name = None,
        ) :
    '''<function description>
    
    <calling functions>
    <called functions>

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
    lst_str_actual_merge_keys = [
        STR_COUNTY_NAME_COLUMN_NAME,
        STR_PRECINCT_NAME_COLUMN_NAME,
        STR_PRECINCT_CODE_COLUMN_NAME]
    lst_str_predicted_merge_keys = [
        STR_COUNTY_NAME_COLUMN_NAME,
        STR_PRECINCT_NAME_COLUMN_NAME,
        STR_PRECINCT_CODE_COLUMN_NAME]
    lst_str_actual_vote_count_column_names = \
        lst_str_predicted_for_actual_vote_count_column_names
    lst_str_predicted_vote_count_column_names = \
        lst_str_predicted_for_actual_vote_count_column_names

    df_actual_and_predicted_vote_counts = df_actual_vote_counts[
        [STR_IS_AUDITED_COLUMN_NAME] +
        lst_str_actual_merge_keys +
        lst_str_actual_vote_count_column_names].merge(
            df_predicted_vote_counts[
                lst_str_predicted_merge_keys +
                lst_str_predicted_vote_count_column_names],
            left_on=lst_str_actual_merge_keys,
            right_on=lst_str_predicted_merge_keys,
            sort=True,
            suffixes=("", "_pre"))

    # For all audited precints, set predicted counts to the actual counts
    for str_votes_counts_columns_name in lst_str_predicted_for_actual_vote_count_column_names :
        lst_bool_filter = df_actual_and_predicted_vote_counts[STR_IS_AUDITED_COLUMN_NAME] == 1
        df_actual_and_predicted_vote_counts.loc[lst_bool_filter, str_votes_counts_columns_name + "_pre"] = \
            df_actual_and_predicted_vote_counts.loc[lst_bool_filter, str_votes_counts_columns_name]

    for str_votes_counts_columns_name in lst_str_predicted_for_actual_vote_count_column_names :
        df_actual_and_predicted_vote_counts[str_votes_counts_columns_name + "_sum"] = \
            df_actual_and_predicted_vote_counts[str_votes_counts_columns_name] + \
            df_actual_and_predicted_vote_counts[str_votes_counts_columns_name + "_pre"]

    if bool_save_outlier_score_stats_to_csv_file :
        dict_lst_outlier_score_stats = {}
        dict_lst_outlier_score_stats["id_1"] = []
        dict_lst_outlier_score_stats["id_2"] = []
        dict_lst_outlier_score_stats["M"] = []
        dict_lst_outlier_score_stats["N"] = []
        dict_lst_outlier_score_stats["n"] = []
        dict_lst_outlier_score_stats["k"] = []
        dict_lst_outlier_score_stats["runtime_ms"] = []
        dict_lst_outlier_score_stats["bm_runtime_ms"] = []
        dict_lst_outlier_score_stats["outlier_score"] = []
        dict_lst_outlier_score_stats["bm_outlier_score"] = []
        dict_lst_outlier_score_stats["bm_cdf"] = []
        dict_lst_outlier_score_stats["bm_sf"] = []
        dict_lst_outlier_score_stats["bm_pmf"] = []
    else :
        dict_lst_outlier_score_stats = None

    for index_of_precinct_row in df_actual_and_predicted_vote_counts.index :
        (flt_outlier_score_level_1_for_precinct,
         flt_outlier_score_level_2_for_precinct,
         flt_outlier_score_level_3_for_precinct) = \
            compute_outlier_score_for_precinct(
                df_actual_and_predicted_vote_counts =
                    df_actual_and_predicted_vote_counts,
                lst_str_predicted_for_actual_vote_count_column_names =
                    lst_str_predicted_for_actual_vote_count_column_names,
                index_of_precinct_row = index_of_precinct_row,
                bool_use_decimal_type =
                    bool_use_decimal_type,
                int_decimal_computational_precision =
                    int_decimal_computational_precision,
                int_decimal_reporting_precision =
                    int_decimal_reporting_precision,
                int_max_num_iters_for_exact_hypergeom =
                    int_max_num_iters_for_exact_hypergeom,
                int_max_num_iters_for_lanczos_approx_hypergeom =
                    int_max_num_iters_for_lanczos_approx_hypergeom,
                int_max_num_iters_for_spouge_approx_hypergeom =
                    int_max_num_iters_for_spouge_approx_hypergeom,
                int_min_sample_size_for_approx_normal =
                    int_min_sample_size_for_approx_normal,
                flt_max_sample_sz_frac_of_pop_sz_for_approx_normal =
                    flt_max_sample_sz_frac_of_pop_sz_for_approx_normal,
                flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal =
                    flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal,
                flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal =
                    flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal,
                bool_true_each_choice_with_all_other_choices__false_each_choice_with_each_choice =
                    bool_true_each_choice_with_all_other_choices__false_each_choice_with_each_choice,
                bool_compute_outlier_score_level_1 =
                    bool_compute_outlier_score_level_1,
                bool_compute_outlier_score_level_2 =
                    bool_compute_outlier_score_level_2,
                bool_compute_outlier_score_level_3 =
                    bool_compute_outlier_score_level_3,
                bool_save_outlier_score_stats_to_csv_file =
                    bool_save_outlier_score_stats_to_csv_file,
                dict_lst_outlier_score_stats = dict_lst_outlier_score_stats,
                int_id_1 = index_of_precinct_row,)
        df_actual_and_predicted_vote_counts.loc[
            index_of_precinct_row, "OUTLIER_SCORE_LEVEL_1"] = \
            flt_outlier_score_level_1_for_precinct
        df_actual_and_predicted_vote_counts.loc[
            index_of_precinct_row, "OUTLIER_SCORE_LEVEL_2"] = \
            flt_outlier_score_level_2_for_precinct
        df_actual_and_predicted_vote_counts.loc[
            index_of_precinct_row, "OUTLIER_SCORE_LEVEL_3"] = \
            flt_outlier_score_level_3_for_precinct

    if bool_save_outlier_score_stats_to_csv_file :
        df_outlier_score_stats = pd.DataFrame.from_dict(dict_lst_outlier_score_stats)

    generator_of_random_numbers = np.random.default_rng(int_random_number_generator_seed_for_sorting)
    df_actual_and_predicted_vote_counts["UNIFORM_RANDOM_NUMBER"] = generator_of_random_numbers.random(
        len(df_actual_and_predicted_vote_counts.index))
    df_actual_and_predicted_vote_counts.sort_values(
        by=[STR_IS_AUDITED_COLUMN_NAME, "OUTLIER_SCORE_LEVEL_1", "OUTLIER_SCORE_LEVEL_2",
            "OUTLIER_SCORE_LEVEL_3", "UNIFORM_RANDOM_NUMBER"], ascending=False, inplace=True)

    lst_str_sum_merge_values = [e + "_sum" for e in lst_str_predicted_for_actual_vote_count_column_names]
    df_actual_and_predicted_vote_counts.drop(lst_str_sum_merge_values, axis=1, inplace=True)
    if not bool_retain_predicted_vote_counts_columns_in_sorted_precincts_file :
        lst_str_predicted_merge_new_values = [e + "_pre" for e in lst_str_predicted_for_actual_vote_count_column_names]
        df_actual_and_predicted_vote_counts.drop(lst_str_predicted_merge_new_values, axis=1, inplace=True)

    if not bool_retain_sorting_criteria_columns :
        df_actual_and_predicted_vote_counts.drop(
            columns=["OUTLIER_SCORE_LEVEL_1", "OUTLIER_SCORE_LEVEL_2", "OUTLIER_SCORE_LEVEL_3",
                     "UNIFORM_RANDOM_NUMBER"], axis=1, inplace=True)

    if bool_compute_cumulative_actual_vote_count_fractions :
        series_total_actual_vote_counts_per_precinct = df_actual_and_predicted_vote_counts[
            lst_str_predicted_for_actual_vote_count_column_names].sum(axis=1)
        flt_total_actual_vote_counts = float(series_total_actual_vote_counts_per_precinct.sum())
        series_cum_total_actual_vote_counts_per_precinct = series_total_actual_vote_counts_per_precinct.cumsum()
        df_actual_and_predicted_vote_counts["CUMULATIVE_TOTAL_ACTUAL_VOTE_COUNT_FRACTION"] = \
            series_cum_total_actual_vote_counts_per_precinct / flt_total_actual_vote_counts
        for str_votes_counts_columns_name in lst_str_predicted_for_actual_vote_count_column_names :
            df_actual_and_predicted_vote_counts["CUMULATIVE_FRACTION_" + str_votes_counts_columns_name] = \
                df_actual_and_predicted_vote_counts[str_votes_counts_columns_name].cumsum().div(
                    series_cum_total_actual_vote_counts_per_precinct)

    if bool_distribute_job_output_files_in_directories_by_type or \
       bool_gather_all_job_files_in_one_directory :
        if bool_save_sorted_precincts_to_csv_file and \
           str_csv_file_name_for_saving_sorted_precincts is not None :
            str_csv_file_name_ext = \
                str_csv_file_name_for_saving_sorted_precincts + ".csv"
            str_csv_full_file_name_ext = os.path.join(
                STR_OUTPUT_DATA_PARM_CSV_PATH, str_csv_file_name_ext)
            try :
                if os.path.exists(str_csv_full_file_name_ext) :
                    os.remove(str_csv_full_file_name_ext)
            except Exception as exception :
                print_exception_message(exception)
            try :
                df_actual_and_predicted_vote_counts.to_csv(
                    str_csv_full_file_name_ext, index = False)
            except Exception as exception :
                print_exception_message(exception)

            if bool_gather_all_job_files_in_one_directory and \
               str_job_subdir_name is not None :
                str_csv_full_file_name_ext_from = str_csv_full_file_name_ext
                str_csv_full_file_name_ext_to = os.path.join(
                    str_job_subdir_name, str_csv_file_name_ext)
                try :
                    if bool_distribute_job_output_files_in_directories_by_type :
                        shutil.copy(
                            str_csv_full_file_name_ext_from,
                            str_csv_full_file_name_ext_to)
                    else :
                        shutil.move(
                            str_csv_full_file_name_ext_from,
                            str_csv_full_file_name_ext_to)
                        pass
                except Exception as exception :
                    print_exception_message(exception)

        if bool_save_outlier_score_stats_to_csv_file and \
           str_csv_file_name_for_saving_outlier_score_stats is not None :
            str_csv_file_name_ext = \
                str_csv_file_name_for_saving_outlier_score_stats + ".csv"
            str_csv_full_file_name_ext = os.path.join(
                STR_OUTPUT_SCORES_PARM_CSV_PATH, str_csv_file_name_ext)
            try :
                if os.path.exists(str_csv_full_file_name_ext) :
                    os.remove(str_csv_full_file_name_ext)
            except Exception as exception :
                print_exception_message(exception)
            try :
                df_outlier_score_stats.to_csv(
                    str_csv_full_file_name_ext, index = False)
            except Exception as exception :
                print_exception_message(exception)

            if bool_gather_all_job_files_in_one_directory and \
               str_job_subdir_name is not None :
                str_csv_full_file_name_ext_from = str_csv_full_file_name_ext
                str_csv_full_file_name_ext_to = os.path.join(
                    str_job_subdir_name, str_csv_file_name_ext)
                try :
                    if bool_distribute_job_output_files_in_directories_by_type :
                        shutil.copy(
                            str_csv_full_file_name_ext_from,
                            str_csv_full_file_name_ext_to)
                    else :
                        shutil.move(
                            str_csv_full_file_name_ext_from,
                            str_csv_full_file_name_ext_to)
                        pass
                except Exception as exception :
                    print_exception_message(exception)

    return df_actual_and_predicted_vote_counts

###############################################################################

def generate_merged_counts(
    str_actual_vote_counts_csv_file_name,
    lst_str_actual_all_vote_count_column_names = [],
    str_actual_residual_vote_count_column_name = None,
    lst_str_actual_residual_vote_count_column_names = [],
    #
    str_benchmark_vote_counts_csv_file_name = None,
    lst_str_benchmark_all_vote_count_column_names = [],
    str_benchmark_residual_vote_count_column_name = None,
    lst_str_benchmark_residual_vote_count_column_names = [],
    #
    bool_aggregate_vote_counts_by_county = False,
    bool_approximate_missing_benchmark_precincts_by_scaled_benchmark_county = False,
    bool_aggregate_missing_benchmark_precincts_into_new_residual_county = False,
    ) :
    '''<function description>
    
    <calling functions>
    <called functions>

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
    df_actual_vote_counts = pd.read_csv(os.path.join(
        STR_INPUT_DATA_CSV_PATH,
        str_actual_vote_counts_csv_file_name + ".csv"))
    if str_actual_residual_vote_count_column_name is not None and \
       len(lst_str_actual_residual_vote_count_column_names) > 0 :
        df_actual_vote_counts = \
            replace_residual_choices_with_aggregate_choice(
                df_vote_counts = df_actual_vote_counts,
                str_residual_vote_count_column_name =
                    str_actual_residual_vote_count_column_name,
                lst_str_residual_vote_count_column_names =
                    lst_str_actual_residual_vote_count_column_names,)

    if str_benchmark_vote_counts_csv_file_name is not None and \
       str_benchmark_vote_counts_csv_file_name != "" :
        df_benchmark_vote_counts = pd.read_csv(os.path.join(
            STR_INPUT_DATA_CSV_PATH,
            str_benchmark_vote_counts_csv_file_name + ".csv"))
        if str_benchmark_residual_vote_count_column_name is not None and \
           len(lst_str_benchmark_residual_vote_count_column_names) > 0 :
            df_benchmark_vote_counts = \
                replace_residual_choices_with_aggregate_choice(
                    df_vote_counts = df_benchmark_vote_counts,
                    str_residual_vote_count_column_name =
                        str_benchmark_residual_vote_count_column_name,
                    lst_str_residual_vote_count_column_names =
                        lst_str_benchmark_residual_vote_count_column_names,)

    else :
        df_benchmark_vote_counts = None

    lst_str_merge_keys = [
        STR_COUNTY_NAME_COLUMN_NAME,
        STR_PRECINCT_NAME_COLUMN_NAME,
        STR_PRECINCT_CODE_COLUMN_NAME]
    df_merged_vote_counts_by_precinct = None

    if bool_aggregate_vote_counts_by_county :

        # 1. Group "actual counts" by county.
        df_actual_vote_counts_copy = df_actual_vote_counts.copy()
        df_actual_vote_counts_copy[STR_PRECINCT_NAME_COLUMN_NAME] = "Composite"
        df_actual_vote_counts_copy[STR_PRECINCT_CODE_COLUMN_NAME] = "Composite"
        dict_act_col_name_to_agg_oper = \
            {n : "sum" for n in lst_str_actual_all_vote_count_column_names}
        dict_act_col_name_to_agg_oper[STR_IS_AUDITED_COLUMN_NAME] = "min"
        df_actual_vote_counts_aggr_by_county = \
            df_actual_vote_counts_copy.groupby(
                lst_str_merge_keys, sort = True).agg(
                    dict_act_col_name_to_agg_oper).reset_index(drop=True)
        df_actual_vote_counts_aggr_by_county.insert(
                0, STR_IS_AUDITED_COLUMN_NAME,
                df_actual_vote_counts_aggr_by_county.pop(
                    STR_IS_AUDITED_COLUMN_NAME)) 
        df_actual_vote_counts_copy = None

        if df_benchmark_vote_counts is not None :
            # 2. Group "benchmark counts" by county.
            df_benchmark_vote_counts_copy = df_benchmark_vote_counts.copy()
            df_benchmark_vote_counts_copy[STR_PRECINCT_NAME_COLUMN_NAME] = \
                "Composite"
            df_benchmark_vote_counts_copy[STR_PRECINCT_CODE_COLUMN_NAME] = \
                "Composite"

            dict_bench_col_name_to_agg_oper = \
                {n : "sum" for n in
                 lst_str_benchmark_all_vote_count_column_names}
            #dict_bench_col_name_to_agg_oper[STR_IS_AUDITED_COLUMN_NAME] = "min"
            df_benchmark_vote_counts_aggr_by_county = \
                df_benchmark_vote_counts_copy.groupby(
                    lst_str_merge_keys, sort = True).agg(
                        dict_bench_col_name_to_agg_oper).reset_index(drop=True)
            #df_benchmark_vote_counts_aggr_by_county.insert(
            #        0, STR_IS_AUDITED_COLUMN_NAME,
            #        df_benchmark_vote_counts_aggr_by_county.pop(
            #            STR_IS_AUDITED_COLUMN_NAME)) 
            df_benchmark_vote_counts_copy = None

            df_merged_vote_counts_by_precinct = \
                df_actual_vote_counts_aggr_by_county[
                    [STR_IS_AUDITED_COLUMN_NAME] +
                    lst_str_merge_keys +
                    lst_str_actual_all_vote_count_column_names].merge(
                        right = df_benchmark_vote_counts_aggr_by_county[
                            lst_str_merge_keys +
                            lst_str_benchmark_all_vote_count_column_names],
                        on = lst_str_merge_keys,
                        how = 'inner',
                        sort = True,)
            df_merged_vote_counts_by_precinct.insert(
                    0, STR_IS_AUDITED_COLUMN_NAME,
                    df_merged_vote_counts_by_precinct.pop(
                        STR_IS_AUDITED_COLUMN_NAME)) 
        else :
            df_merged_vote_counts_by_precinct = \
                df_actual_vote_counts_aggr_by_county
    else :
        if bool_approximate_missing_benchmark_precincts_by_scaled_benchmark_county :
            if df_benchmark_vote_counts is not None :

                # 1. Group "actual counts" by county.
                dict_act_col_name_to_agg_oper = \
                    {n : "sum" for n in lst_str_actual_all_vote_count_column_names}
                dict_act_col_name_to_agg_oper[STR_IS_AUDITED_COLUMN_NAME] = "min"
                df_actual_vote_counts_aggr_by_county = df_actual_vote_counts.groupby(
                    [STR_COUNTY_NAME_COLUMN_NAME], sort=True).agg(dict_act_col_name_to_agg_oper).reset_index()
                df_actual_vote_counts_aggr_by_county.insert(
                    0, STR_IS_AUDITED_COLUMN_NAME, df_actual_vote_counts_aggr_by_county.pop(STR_IS_AUDITED_COLUMN_NAME)) 

                # 2. Compute total "tally" for each county in the "actual counts".
                df_actual_vote_counts_aggr_by_county["ACTUAL_COUNTY_TALLY"] = df_actual_vote_counts_aggr_by_county[
                    lst_str_actual_all_vote_count_column_names].sum(axis=1)

                # 3. Group "benchmark counts" by county.
                dict_benchmark_all_vote_count_col_name_to_agg_oper = {
                    n : "sum" for n in lst_str_benchmark_all_vote_count_column_names}
                df_benchmark_vote_counts_aggr_by_county = df_benchmark_vote_counts.groupby(
                    [STR_COUNTY_NAME_COLUMN_NAME], sort=True).agg(
                    dict_benchmark_all_vote_count_col_name_to_agg_oper).reset_index().rename(
                    columns={'index':STR_COUNTY_NAME_COLUMN_NAME})

                # 4. Rename columns in the "benchmark counts".
                lst_str_benchmark_votes_county_counts_columns_names = []
                for str_benchmark_votes_counts_column_name in lst_str_benchmark_all_vote_count_column_names :
                    str_benchmark_votes_counts_column_new_name = str_benchmark_votes_counts_column_name + "_county"
                    lst_str_benchmark_votes_county_counts_columns_names += [str_benchmark_votes_counts_column_new_name]
                    df_benchmark_vote_counts_aggr_by_county.rename(columns={
                        str_benchmark_votes_counts_column_name : str_benchmark_votes_counts_column_new_name}, inplace=True)

                # 5. Compute total "tally" for each precinct in the "actual counts".
                df_merged_vote_counts_by_precinct = df_actual_vote_counts[
                    [STR_IS_AUDITED_COLUMN_NAME] + lst_str_merge_keys + lst_str_actual_all_vote_count_column_names]
                df_merged_vote_counts_by_precinct["ACTUAL_PRECINCT_TALLY"] = df_merged_vote_counts_by_precinct[
                    lst_str_actual_all_vote_count_column_names].sum(axis=1)

                # 6. For each precinct in the "actual counts", compute the
                #    "precinct scaling factor" = "precinct tally" / "this precinct's county tally".
                df_merged_vote_counts_by_precinct = df_merged_vote_counts_by_precinct.merge(
                        right = df_actual_vote_counts_aggr_by_county[[STR_COUNTY_NAME_COLUMN_NAME, "ACTUAL_COUNTY_TALLY"]],
                        how = 'inner', on = [STR_COUNTY_NAME_COLUMN_NAME], sort = True,)
                df_merged_vote_counts_by_precinct["ACTUAL_PRECINCT_TALLY_FRACTION_IN_COUNTY_TALLY"] = \
                    df_merged_vote_counts_by_precinct["ACTUAL_PRECINCT_TALLY"] / \
                    df_merged_vote_counts_by_precinct["ACTUAL_COUNTY_TALLY"]

                # 7. For each precinct in the "actual counts", compute the proxies of the precinct's "benchmark counts" as
                #    round("precinct scaling factor" * "this precinct's countywide benchmark choice count").
                # https://stackoverflow.com/questions/29530232/how-to-check-if-any-value-is-nan-in-a-pandas-dataframe
                df_merged_vote_counts_by_precinct = df_merged_vote_counts_by_precinct.merge(
                        right = df_benchmark_vote_counts[lst_str_merge_keys + lst_str_benchmark_all_vote_count_column_names],
                        how = 'left', on = lst_str_merge_keys, sort = True,)
                df_merged_vote_counts_by_precinct = df_merged_vote_counts_by_precinct.merge(
                        right = df_benchmark_vote_counts_aggr_by_county[[STR_COUNTY_NAME_COLUMN_NAME] +
                            lst_str_benchmark_votes_county_counts_columns_names],
                        how = 'inner', on = [STR_COUNTY_NAME_COLUMN_NAME], sort = True,)

                for str_benchmark_votes_counts_column_name in lst_str_benchmark_all_vote_count_column_names :
                    df_merged_vote_counts_by_precinct.loc[df_merged_vote_counts_by_precinct[
                        str_benchmark_votes_counts_column_name].isnull(), str_benchmark_votes_counts_column_name] = round(
                        df_merged_vote_counts_by_precinct.loc[df_merged_vote_counts_by_precinct[
                            str_benchmark_votes_counts_column_name].isnull(), "ACTUAL_PRECINCT_TALLY_FRACTION_IN_COUNTY_TALLY"] * \
                        df_merged_vote_counts_by_precinct.loc[df_merged_vote_counts_by_precinct[
                            str_benchmark_votes_counts_column_name].isnull(), str_benchmark_votes_counts_column_name + "_county"])
                    df_merged_vote_counts_by_precinct[str_benchmark_votes_counts_column_name] = \
                        df_merged_vote_counts_by_precinct[str_benchmark_votes_counts_column_name].astype('int')

                # 8. Cleanup
                df_merged_vote_counts_by_precinct.drop([
                    "ACTUAL_PRECINCT_TALLY", "ACTUAL_COUNTY_TALLY",
                    "ACTUAL_PRECINCT_TALLY_FRACTION_IN_COUNTY_TALLY"], axis=1, inplace=True)
                df_merged_vote_counts_by_precinct.drop(
                    (s for s in lst_str_benchmark_votes_county_counts_columns_names), axis=1, inplace=True)
            else :
                df_merged_vote_counts_by_precinct = df_actual_vote_counts[
                    [STR_IS_AUDITED_COLUMN_NAME] + lst_str_merge_keys + lst_str_actual_all_vote_count_column_names].copy()

        elif bool_aggregate_missing_benchmark_precincts_into_new_residual_county :

            if df_benchmark_vote_counts is not None :

                # 1. For those precincts that are both in "actual counts" and "benchmark counts" copy and use respective
                #    "benchmark counts".
                df_inner_merged_vote_counts_by_precinct = \
                    df_actual_vote_counts[[STR_IS_AUDITED_COLUMN_NAME] + lst_str_merge_keys +
                                          lst_str_actual_all_vote_count_column_names].merge(
                        right = df_benchmark_vote_counts[lst_str_merge_keys + lst_str_benchmark_all_vote_count_column_names],
                        on = lst_str_merge_keys, how = 'inner', sort = True,)
                df_inner_merged_vote_counts_by_precinct.insert(
                    0, STR_IS_AUDITED_COLUMN_NAME, df_inner_merged_vote_counts_by_precinct.pop(STR_IS_AUDITED_COLUMN_NAME)) 

                # 2. Select those precincts that are only in "actual counts" but cannot be matched/found in the "benchmark counts".
                df_actual_only_merged_vote_counts_by_precinct = df_actual_vote_counts[[STR_IS_AUDITED_COLUMN_NAME] +
                    lst_str_merge_keys + lst_str_actual_all_vote_count_column_names].merge(
                        right = df_benchmark_vote_counts[lst_str_merge_keys + lst_str_benchmark_all_vote_count_column_names],
                        on = lst_str_merge_keys, how = 'left', indicator = True, sort = True,)
                df_actual_only_merged_vote_counts_by_precinct = df_actual_only_merged_vote_counts_by_precinct[
                    df_actual_only_merged_vote_counts_by_precinct['_merge'] == 'left_only']
                df_actual_only_merged_vote_counts_by_precinct.drop(['_merge'], axis = 1, inplace = True)
                df_actual_only_merged_vote_counts_by_precinct.insert(
                    0, STR_IS_AUDITED_COLUMN_NAME, df_actual_only_merged_vote_counts_by_precinct.pop(STR_IS_AUDITED_COLUMN_NAME)) 

                if len(df_actual_only_merged_vote_counts_by_precinct.index) > 0 :
                    # 3. Aggregate these only-"actual counts" precincts into a special artificial new "county" named "Various" with
                    #    one precinct "Composite" only.
                    df_actual_only_merged_vote_counts_by_precinct[STR_COUNTY_NAME_COLUMN_NAME] = "Various"
                    df_actual_only_merged_vote_counts_by_precinct[STR_PRECINCT_NAME_COLUMN_NAME] = "Composite"
                    df_actual_only_merged_vote_counts_by_precinct[STR_PRECINCT_CODE_COLUMN_NAME] = "Composite"

                    dict_act_col_name_to_agg_oper = {n : "sum" for n in lst_str_actual_all_vote_count_column_names}
                    dict_act_col_name_to_agg_oper[STR_IS_AUDITED_COLUMN_NAME] = "min"
                    df_actual_only_merged_vote_counts_aggregated = df_actual_only_merged_vote_counts_by_precinct.groupby(
                        lst_str_merge_keys, sort=True).agg(dict_act_col_name_to_agg_oper).reset_index()
                    df_actual_only_merged_vote_counts_aggregated.insert(
                        0, STR_IS_AUDITED_COLUMN_NAME, df_actual_only_merged_vote_counts_aggregated.pop(
                            STR_IS_AUDITED_COLUMN_NAME)) 

                    # 4. Compute "actual count tally from Various county"
                    int_actual_various_county_tally = int(df_actual_only_merged_vote_counts_aggregated[
                        lst_str_actual_all_vote_count_column_names].sum(axis=1))

                    # 5. Group "actual counts" on the statewide level and compute total statewide "tally".
                    int_actual_statewide_tally = int(df_actual_vote_counts[
                        lst_str_actual_all_vote_count_column_names].sum().sum())

                    # 6. Group "benchmark counts" on the statewide level and compute total statewide "tally".
                    df_benchmark_vote_counts_statewide = df_benchmark_vote_counts.copy()
                    df_benchmark_vote_counts_statewide[STR_COUNTY_NAME_COLUMN_NAME] = "Various"
                    df_benchmark_vote_counts_statewide[STR_PRECINCT_NAME_COLUMN_NAME] = "Composite"
                    df_benchmark_vote_counts_statewide[STR_PRECINCT_CODE_COLUMN_NAME] = "Composite"

                    dict_bench_col_name_to_agg_oper = {n : "sum" for n in lst_str_benchmark_all_vote_count_column_names}
                    dict_bench_col_name_to_agg_oper[STR_IS_AUDITED_COLUMN_NAME] = "min"
                    df_benchmark_vote_counts_statewide = df_benchmark_vote_counts_statewide.groupby(
                        lst_str_merge_keys, sort=True).agg(dict_bench_col_name_to_agg_oper).reset_index()
                    df_benchmark_vote_counts_statewide.insert(
                        0, STR_IS_AUDITED_COLUMN_NAME, df_benchmark_vote_counts_statewide.pop(STR_IS_AUDITED_COLUMN_NAME)) 

                    # 7. Compute the "tally scaling factor" = "actual count tally from Various county" /
                    #    "actual count tally from the state".
                    flt_actual_various_county_tally_frac_in_actual_statewide_tally = \
                        float(int_actual_various_county_tally) / float(int_actual_statewide_tally)

                    # 8. For the precinct from the "Various" county in the "actual counts", compute the proxies of the precinct's
                    #    "benchmark counts" as round("tally scaling factor" * "statwide benchmark choice count").
                    df_benchmark_vote_counts_statewide_scaled = df_benchmark_vote_counts_statewide.copy()
                    df_benchmark_vote_counts_statewide_scaled[lst_str_benchmark_all_vote_count_column_names] = round(
                        df_benchmark_vote_counts_statewide[lst_str_benchmark_all_vote_count_column_names] *
                        flt_actual_various_county_tally_frac_in_actual_statewide_tally)
                    df_benchmark_vote_counts_statewide_scaled[lst_str_benchmark_all_vote_count_column_names] = \
                        df_benchmark_vote_counts_statewide_scaled[lst_str_benchmark_all_vote_count_column_names].astype('int')

                    # 9. Concatenate by columns for "Various": actual with scaled statwide benchmark
                    df_actual_benchmark_vote_counts_various_county = pd.concat([df_actual_only_merged_vote_counts_aggregated,
                        df_benchmark_vote_counts_statewide_scaled[lst_str_benchmark_all_vote_count_column_names]], axis=1)

                    # 10. Combine original merged precints "actual counts" with the special "Composite" precinct from county "Various"
                    df_merged_vote_counts_by_precinct = pd.concat(
                        [df_inner_merged_vote_counts_by_precinct, df_actual_benchmark_vote_counts_various_county],
                        axis=0).reset_index(drop=True)

                else :
                    df_merged_vote_counts_by_precinct = \
                        df_inner_merged_vote_counts_by_precinct
            else :
                df_merged_vote_counts_by_precinct = df_actual_vote_counts[
                    [STR_IS_AUDITED_COLUMN_NAME] + lst_str_merge_keys +
                    lst_str_actual_all_vote_count_column_names].copy()
        else :
            if df_benchmark_vote_counts is not None :
                df_merged_vote_counts_by_precinct = df_actual_vote_counts[
                    [STR_IS_AUDITED_COLUMN_NAME] + lst_str_merge_keys +
                    lst_str_actual_all_vote_count_column_names].merge(
                        right = df_benchmark_vote_counts[
                            lst_str_merge_keys +
                            lst_str_benchmark_all_vote_count_column_names],
                        on = lst_str_merge_keys, how = 'inner', sort = True,)
                df_merged_vote_counts_by_precinct.insert(
                    0, STR_IS_AUDITED_COLUMN_NAME,
                    df_merged_vote_counts_by_precinct.pop(
                        STR_IS_AUDITED_COLUMN_NAME))
            else :
                df_merged_vote_counts_by_precinct = df_actual_vote_counts[
                    [STR_IS_AUDITED_COLUMN_NAME] + lst_str_merge_keys +
                    lst_str_actual_all_vote_count_column_names].copy()

    return df_merged_vote_counts_by_precinct

###############################################################################

def sort_by_computed_mh_pmf(
    df_data_panel,
    lst_str_all_predicting_predicted_vote_count_column_names,
    generator_of_random_numbers,
    int_window_left_index,
    int_window_right_index,) :
    '''<function description>
    
    <calling functions>
    <called functions>

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
    # Extract data from sub-interval
    x=df_data_panel.iloc[int_window_left_index:(int_window_right_index+1)].loc[
        :, lst_str_all_predicting_predicted_vote_count_column_names].values.tolist()

    # Compute PMF for sub-interval
    df_data_panel.iloc[
        int_window_left_index:(int_window_right_index+1),
        df_data_panel.columns.get_loc("OUTLIER_SCORE")] = \
            multivariate_hypergeom.pmf(
                x=x,
                m=[[sum(lst) for lst in list(map(list, zip(*x)))]] *
                    (int_window_right_index - int_window_left_index + 1),
                n=[sum(lst) for lst in x],)

    # Generate random numbers for sub-interval.
    df_data_panel.iloc[
        int_window_left_index:(int_window_right_index+1),
        df_data_panel.columns.get_loc("UNIFORM_RANDOM_NUMBER")] = \
            generator_of_random_numbers.random(len(x))

    # Sort sub-interval
    df_data_panel = pd.concat([
        df_data_panel.iloc[:int_window_left_index],
        df_data_panel.iloc[int_window_left_index:(
            int_window_right_index+1)].sort_values(
            by=["OUTLIER_SCORE", "UNIFORM_RANDOM_NUMBER"], ascending = False),
        df_data_panel.iloc[(int_window_right_index+1):],], ignore_index = True)

    return df_data_panel

###############################################################################

def sort_precincts_by_decreasing_predictability_for_non_parametric_model(
        df_data_panel,
        lst_str_all_predicting_predicted_vote_count_column_names,
        int_max_num_two_ways_passes = 1, # integer in [1; +Inf]
        flt_window_size_scaling_factor = .5, # 0 < f < 1
        int_random_number_generator_seed_for_sorting = 0,
        bool_retain_sorting_criteria_columns = False,
        bool_compute_cumulative_actual_vote_count_fractions = False,
        bool_save_sorted_precincts_to_csv_file = False,
        str_csv_file_name_for_saving_sorted_precincts = None,
        bool_distribute_job_output_files_in_directories_by_type = True,
        bool_gather_all_job_files_in_one_directory = False,
        str_job_subdir_name = None,
        ) :
    '''<function description>
    
    <calling functions>
    <called functions>

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

    int_num_precincts = df_data_panel.shape[0]
    df_data_panel["OUTLIER_SCORE"] = math.nan
    df_data_panel["UNIFORM_RANDOM_NUMBER"] = math.nan
    generator_of_random_numbers = np.random.default_rng(
        int_random_number_generator_seed_for_sorting)

    flt_window_size_frac = 1. / flt_window_size_scaling_factor
    for int_dual_pass_index in range(int_max_num_two_ways_passes):
        flt_window_size_frac *= flt_window_size_scaling_factor
        flt_window_shift_frac = flt_window_size_frac / 2.
        flt_window_size_precincts = flt_window_size_frac * int_num_precincts
        flt_window_shift_precincts = flt_window_shift_frac * int_num_precincts
        if flt_window_size_precincts < 1.5 or flt_window_shift_precincts < 0.5:
            break
        # print(str(int_dual_pass_index + 1) + ". Window size: " +
        #       str(flt_window_size_precincts) + " ; Window shift: " +
        #       str(flt_window_shift_precincts))

        # Scan from left to right.
        int_shift_index = 0
        flt_cum_shift = int_shift_index * flt_window_shift_precincts
        int_window_left_index = min(int(round(
            flt_cum_shift)), int_num_precincts)
        int_window_right_index = int(round(
            flt_cum_shift + flt_window_size_precincts - 1.0))
        while True:
            if int_window_right_index >= int_num_precincts :
                break
            else :
                # print(str(int_window_left_index) + ";" +
                #       str(int_window_right_index))
                df_data_panel = sort_by_computed_mh_pmf(
                    df_data_panel = df_data_panel,
                    lst_str_all_predicting_predicted_vote_count_column_names =
                        lst_str_all_predicting_predicted_vote_count_column_names,
                    generator_of_random_numbers = generator_of_random_numbers,
                    int_window_left_index = int_window_left_index,
                    int_window_right_index = int_window_right_index,)
            int_shift_index += 1
            flt_cum_shift = int_shift_index * flt_window_shift_precincts
            int_window_left_index = min(int(round(
                flt_cum_shift)), int_num_precincts)
            int_window_right_index = int(round(
                flt_cum_shift + flt_window_size_precincts - 1.0))

        # Scan from right to left.
        if int_dual_pass_index > 0 :
            int_shift_index = 1
            flt_cum_shift = int_shift_index * flt_window_shift_precincts
            int_window_left_index = int(round(
                int_num_precincts - flt_cum_shift - flt_window_size_precincts))
            int_window_right_index = max(int(round(
                int_num_precincts - flt_cum_shift - 1.0)), 0)
            while True:
                if int_window_left_index < 0 :
                    break
                else :
                    #if True :
                    #    print(str(int_window_left_index) + ";" + str(int_window_right_index))
                    df_data_panel = sort_by_computed_mh_pmf(
                        df_data_panel = df_data_panel,
                        lst_str_all_predicting_predicted_vote_count_column_names =
                            lst_str_all_predicting_predicted_vote_count_column_names,
                        generator_of_random_numbers = generator_of_random_numbers,
                        int_window_left_index = int_window_left_index,
                        int_window_right_index = int_window_right_index,)
                int_shift_index += 1
                flt_cum_shift = int_shift_index * flt_window_shift_precincts
                int_window_left_index = int(round(
                    int_num_precincts - flt_cum_shift - flt_window_size_precincts))
                int_window_right_index = max(int(round(
                    int_num_precincts - flt_cum_shift - 1.0)), 0)

    flt_outlier_max_score_for_precinct = 1.
    df_data_panel.loc[df_data_panel[STR_IS_AUDITED_COLUMN_NAME] != 0,
                      "OUTLIER_SCORE"] = flt_outlier_max_score_for_precinct
    df_data_panel["INDEX_FOR_SEPARATION_OF_AUDITED_PRECINCTS"] = \
        [i for i in range(len(df_data_panel),0,-1)]

    # Separate audited from non-audited, preserve order in both groups.
    df_data_panel.sort_values(
            by=[STR_IS_AUDITED_COLUMN_NAME,
                "INDEX_FOR_SEPARATION_OF_AUDITED_PRECINCTS"],
            ascending = False, inplace = True),
    df_data_panel["INDEX_FOR_SHUFFLING_OF_AUDITED_PRECINCTS"] = df_data_panel[
        "INDEX_FOR_SEPARATION_OF_AUDITED_PRECINCTS"]
    df_data_panel.loc[df_data_panel[STR_IS_AUDITED_COLUMN_NAME] != 0,
                      "INDEX_FOR_SHUFFLING_OF_AUDITED_PRECINCTS"] = \
                      generator_of_random_numbers.random(sum(df_data_panel[
                          STR_IS_AUDITED_COLUMN_NAME] != 0))

    # Randomly shuffle just the audited precincts.
    df_data_panel.sort_values(
            by=[STR_IS_AUDITED_COLUMN_NAME,
                "INDEX_FOR_SHUFFLING_OF_AUDITED_PRECINCTS"],
            ascending = False, inplace = True),
    df_data_panel.drop(columns=[
        "INDEX_FOR_SEPARATION_OF_AUDITED_PRECINCTS",
        "INDEX_FOR_SHUFFLING_OF_AUDITED_PRECINCTS"], axis=1, inplace=True)

    if not bool_retain_sorting_criteria_columns :
        df_data_panel.drop(columns=[
            "OUTLIER_SCORE", "UNIFORM_RANDOM_NUMBER"], axis=1, inplace=True)

    if bool_compute_cumulative_actual_vote_count_fractions :
        series_total_actual_vote_counts_per_precinct = df_data_panel[
            lst_str_all_predicting_predicted_vote_count_column_names].sum(axis=1)
        flt_total_actual_vote_counts = float(
            series_total_actual_vote_counts_per_precinct.sum())
        series_cum_total_actual_vote_counts_per_precinct = \
            series_total_actual_vote_counts_per_precinct.cumsum()
        df_data_panel["CUMULATIVE_TOTAL_ACTUAL_VOTE_COUNT_FRACTION"] = \
            series_cum_total_actual_vote_counts_per_precinct / \
            flt_total_actual_vote_counts
        for str_votes_counts_columns_name in \
                lst_str_all_predicting_predicted_vote_count_column_names :
            df_data_panel[
                "CUMULATIVE_FRACTION_" + str_votes_counts_columns_name] = \
                    df_data_panel[str_votes_counts_columns_name].cumsum().div(
                        series_cum_total_actual_vote_counts_per_precinct)

    if bool_distribute_job_output_files_in_directories_by_type or \
       bool_gather_all_job_files_in_one_directory :
        if bool_save_sorted_precincts_to_csv_file and \
           str_csv_file_name_for_saving_sorted_precincts is not None :
            str_csv_file_name_ext = \
                str_csv_file_name_for_saving_sorted_precincts + ".csv"
            str_csv_full_file_name_ext = os.path.join(
                STR_OUTPUT_DATA_NPAR_CSV_PATH, str_csv_file_name_ext)
            try :
                if os.path.exists(str_csv_full_file_name_ext) :
                    os.remove(str_csv_full_file_name_ext)
            except Exception as exception :
                print_exception_message(exception)
            try :
                df_data_panel.to_csv(str_csv_full_file_name_ext, index = False)
            except Exception as exception :
                print_exception_message(exception)

            if bool_gather_all_job_files_in_one_directory and \
               str_job_subdir_name is not None :
                str_csv_full_file_name_ext_from = str_csv_full_file_name_ext
                str_csv_full_file_name_ext_to = os.path.join(
                    str_job_subdir_name, str_csv_file_name_ext)
                try :
                    if bool_distribute_job_output_files_in_directories_by_type :
                        shutil.copy(
                            str_csv_full_file_name_ext_from,
                            str_csv_full_file_name_ext_to)
                    else :
                        shutil.move(
                            str_csv_full_file_name_ext_from,
                            str_csv_full_file_name_ext_to)
                        pass
                except Exception as exception :
                    print_exception_message(exception)

    return df_data_panel

###############################################################################

def compute_cumulative_percent_values(
        df_sorted_data_panel,
        lst_str_vote_count_column_names_for_cumulative_tally_for_curves,
        lst_str_count_col_names_for_cumul_series,
        ):
    '''<function description>
    
    <calling functions>
    <called functions>

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
    STR_COL_NAME_FOR_TALLY_PER_ROW = "Row_Total"
    STR_COL_NAME_FOR_CUMUL_TALLY = "Row_Total_Cumul"
    STR_COL_NAME_POSTFIX_FOR_CUMUL_CHOICE_COUNTS = "_Cumul"

    df_sorted_data_panel_cumul_pct = df_sorted_data_panel.copy()
    df_sorted_data_panel_cumul_pct[STR_COL_NAME_FOR_TALLY_PER_ROW] = \
        df_sorted_data_panel_cumul_pct[
            lst_str_vote_count_column_names_for_cumulative_tally_for_curves].sum(
                axis=1, numeric_only=True)
    int_Grand_Total = df_sorted_data_panel_cumul_pct[
        STR_COL_NAME_FOR_TALLY_PER_ROW].sum(axis=0, numeric_only=True)
    df_sorted_data_panel_cumul_pct[STR_COL_NAME_FOR_CUMUL_TALLY] = \
        df_sorted_data_panel_cumul_pct['Row_Total'].cumsum(axis=0)
    for str_col_name in lst_str_count_col_names_for_cumul_series :
        df_sorted_data_panel_cumul_pct[
            str_col_name + STR_COL_NAME_POSTFIX_FOR_CUMUL_CHOICE_COUNTS] = \
                df_sorted_data_panel_cumul_pct[str_col_name].cumsum(axis=0)
    df_sorted_data_panel_cumul_pct[STR_COL_NAME_FOR_CUMUL_PLOT_X_AXIS] = \
        df_sorted_data_panel_cumul_pct[STR_COL_NAME_FOR_CUMUL_TALLY] / \
            int_Grand_Total * 100.
    for str_col_name in lst_str_count_col_names_for_cumul_series :
        df_sorted_data_panel_cumul_pct[
            str_col_name + STR_COL_NAME_POSTFIX_FOR_CUMUL_PLOT_Y_AXIS] = \
            df_sorted_data_panel_cumul_pct[
                str_col_name + STR_COL_NAME_POSTFIX_FOR_CUMUL_CHOICE_COUNTS] / \
            df_sorted_data_panel_cumul_pct[STR_COL_NAME_FOR_CUMUL_TALLY] * 100.
    df_sorted_data_panel_cumul_pct.drop(
        [STR_COL_NAME_FOR_TALLY_PER_ROW,
         STR_COL_NAME_FOR_CUMUL_TALLY] + [
             (str_col_name + STR_COL_NAME_POSTFIX_FOR_CUMUL_CHOICE_COUNTS)
            for str_col_name in lst_str_count_col_names_for_cumul_series],
             axis=1, inplace=True)
    return df_sorted_data_panel_cumul_pct

###############################################################################

def generate_cumulative_plot(
        bool_is_plot_for_parametric_model,
        bool_is_plot_for_parametric_lasso_model,
        df_sorted_data_panel,
        str_plot_title,
        str_x_axis_label,
        str_y_axis_label,
        #######################################################################
        lst_str_vote_count_column_names_for_cumulative_tally_for_curves,
        lst_str_vote_count_column_name_prefixes_to_use_for_curves,
        lst_str_legend_labels_for_curves,
        lst_str_color_names_for_curves,
        lst_str_linestyles_for_curves = ['-'],
        lst_flt_linewidths_for_curves = [1.],
        lst_bool_draw_right_tail_circle_for_curves = [False],
        lst_str_annotations_for_curves = [''],
        lst_int_annotation_font_sizes_for_curves = [6],
        lst_bool_display_final_cumulative_percent_of_tally_for_curves = [False],
        lst_int_final_cumulative_percent_of_tally_font_sizes_for_curves = [6],
        lst_int_final_cumulative_percent_of_tally_rounding_precision_for_curves = [2],
        #######################################################################
        int_plot_title_font_size = None,
        int_x_axis_label_font_size = None,
        int_y_axis_label_font_size = None,
        int_x_axis_tick_labels_font_size = None,
        int_y_axis_tick_labels_font_size = None,
        int_legend_font_size = None,
        str_legend_loc = 'lower left',
        lst_str_text_line_content = [],
        lst_str_text_line_color = [],
        int_text_line_font_size = 7,
        flt_text_line_x_loc_as_fraction = 0.1275,
        flt_text_line_lower_y_loc_as_fraction = 0.14,
        flt_text_line_upper_y_loc_as_fraction = 0.85,
        flt_text_line_abs_delta_y_loc_as_fraction = 0.03,
        str_text_line_start_y_loc = "upper",
        bool_display_chart_id = False,
        bool_is_chart_from_parametric_model = None,
        int_chart_id_font_size = 7,
        bool_include_mac_address_in_chart_id = False,
        #######################################################################
        lst_flt_min_x = [0.],
        lst_flt_max_x = [100.],
        lst_flt_xticks_incr = [5.],
        lst_flt_min_y = [0.],
        lst_flt_max_y = [100.],
        lst_flt_yticks_incr = [5.],
        lst_flt_plot_width_in_inches = [6.4],
        lst_flt_plot_height_in_inches = [4.8],
        lst_int_dots_per_inch_for_png_and_jpg_plots = [1200],
        lst_flt_right_tails_circles_radius_in_pct_pnts = [0.],
        lst_str_file_name_for_saving_plot = [None,],
        #######################################################################
        bool_show_plot = False,
        bool_save_plot_to_png_file = False,
        bool_save_plot_to_jpg_file = False,
        bool_save_plot_to_pdf_file = False,
        bool_save_plot_to_svg_file = False,
        bool_distribute_job_output_files_in_directories_by_type = True,
        bool_gather_all_job_files_in_one_directory = False,
        str_job_subdir_name = None,
        ) :
    '''<function description>
    
    <calling functions>
    <called functions>

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
    df_sorted_data_panel_cumul_pct = compute_cumulative_percent_values(
        df_sorted_data_panel = df_sorted_data_panel,
        lst_str_vote_count_column_names_for_cumulative_tally_for_curves =
            lst_str_vote_count_column_names_for_cumulative_tally_for_curves,
        lst_str_count_col_names_for_cumul_series =
            lst_str_vote_count_column_name_prefixes_to_use_for_curves,)
    int_min_num_cumul_plots = min(
        len(lst_flt_min_x),
        len(lst_flt_max_x),
        len(lst_flt_xticks_incr),
        len(lst_flt_min_y),
        len(lst_flt_max_y),
        len(lst_flt_yticks_incr),
        len(lst_flt_plot_width_in_inches),
        len(lst_flt_plot_height_in_inches),
        len(lst_int_dots_per_inch_for_png_and_jpg_plots),
        len(lst_flt_right_tails_circles_radius_in_pct_pnts),
        len(lst_str_file_name_for_saving_plot),)
    int_max_num_cumul_plots = max(
        len(lst_flt_min_x),
        len(lst_flt_max_x),
        len(lst_flt_xticks_incr),
        len(lst_flt_min_y),
        len(lst_flt_max_y),
        len(lst_flt_yticks_incr),
        len(lst_flt_plot_width_in_inches),
        len(lst_flt_plot_height_in_inches),
        len(lst_int_dots_per_inch_for_png_and_jpg_plots),
        len(lst_flt_right_tails_circles_radius_in_pct_pnts),
        len(lst_str_file_name_for_saving_plot),)
    if int_min_num_cumul_plots == 0 :
        return

    int_num_of_curves = len(
        lst_str_vote_count_column_name_prefixes_to_use_for_curves)

    if len(lst_str_linestyles_for_curves) == 0 :
        lst_str_linestyles_for_curves = ['-'] * int_num_of_curves
    elif len(lst_str_linestyles_for_curves) < int_num_of_curves :
        lst_str_linestyles_for_curves += \
            [lst_str_linestyles_for_curves[-1]] * (
                int_num_of_curves - len(lst_str_linestyles_for_curves))

    if len(lst_flt_linewidths_for_curves) == 0 :
        lst_flt_linewidths_for_curves = [1.] * int_num_of_curves
    elif len(lst_flt_linewidths_for_curves) < int_num_of_curves :
        lst_flt_linewidths_for_curves += \
            [lst_flt_linewidths_for_curves[-1]] * (
                int_num_of_curves - len(lst_flt_linewidths_for_curves))

    if len(lst_bool_draw_right_tail_circle_for_curves) == 0 :
        lst_bool_draw_right_tail_circle_for_curves = \
            [False] * int_num_of_curves
    elif len(lst_bool_draw_right_tail_circle_for_curves) < int_num_of_curves :
        lst_bool_draw_right_tail_circle_for_curves += \
            [lst_bool_draw_right_tail_circle_for_curves[-1]] * (
                int_num_of_curves -
                len(lst_bool_draw_right_tail_circle_for_curves))

    if len(lst_str_annotations_for_curves) == 0 :
        lst_str_annotations_for_curves = [""] * int_num_of_curves
    elif len(lst_str_annotations_for_curves) < int_num_of_curves :
        lst_str_annotations_for_curves += \
            [lst_str_annotations_for_curves[-1]] * (
                int_num_of_curves - len(lst_str_annotations_for_curves))

    if len(lst_int_annotation_font_sizes_for_curves) == 0 :
        lst_int_annotation_font_sizes_for_curves = [6] * int_num_of_curves
    elif len(lst_int_annotation_font_sizes_for_curves) < int_num_of_curves :
        lst_int_annotation_font_sizes_for_curves += \
            [lst_int_annotation_font_sizes_for_curves[-1]] * (
                int_num_of_curves -
                len(lst_int_annotation_font_sizes_for_curves))

    if len(lst_bool_display_final_cumulative_percent_of_tally_for_curves) == 0 :
        lst_bool_display_final_cumulative_percent_of_tally_for_curves = \
            [False] * int_num_of_curves
    elif len(lst_bool_display_final_cumulative_percent_of_tally_for_curves) < \
         int_num_of_curves :
        lst_bool_display_final_cumulative_percent_of_tally_for_curves += \
            [lst_bool_display_final_cumulative_percent_of_tally_for_curves[-1]] * (
            int_num_of_curves -
            len(lst_bool_display_final_cumulative_percent_of_tally_for_curves))

    if len(lst_int_final_cumulative_percent_of_tally_font_sizes_for_curves) == 0 :
        lst_int_final_cumulative_percent_of_tally_font_sizes_for_curves = \
            [6] * int_num_of_curves
    elif len(lst_int_final_cumulative_percent_of_tally_font_sizes_for_curves) < \
         int_num_of_curves :
        lst_int_final_cumulative_percent_of_tally_font_sizes_for_curves += \
            [lst_int_final_cumulative_percent_of_tally_font_sizes_for_curves[-1]] * (
            int_num_of_curves -
            len(lst_int_final_cumulative_percent_of_tally_font_sizes_for_curves))

    if len(lst_int_final_cumulative_percent_of_tally_rounding_precision_for_curves) == 0 :
        lst_int_final_cumulative_percent_of_tally_rounding_precision_for_curves = \
            [2] * int_num_of_curves
    elif len(lst_int_final_cumulative_percent_of_tally_rounding_precision_for_curves) < \
         int_num_of_curves :
        lst_int_final_cumulative_percent_of_tally_rounding_precision_for_curves += \
            [lst_int_final_cumulative_percent_of_tally_rounding_precision_for_curves[-1]] * (
            int_num_of_curves -
            len(lst_int_final_cumulative_percent_of_tally_rounding_precision_for_curves))

    plt.ioff()

    for int_cumul_plot_index in range(int_max_num_cumul_plots) :

        if int_cumul_plot_index < len(lst_flt_min_x) :
            flt_min_x = lst_flt_min_x[int_cumul_plot_index]
        if int_cumul_plot_index < len(lst_flt_max_x) :
            flt_max_x = lst_flt_max_x[int_cumul_plot_index]
        if int_cumul_plot_index < len(lst_flt_xticks_incr) :
            flt_xticks_incr = lst_flt_xticks_incr[int_cumul_plot_index]
        if int_cumul_plot_index < len(lst_flt_min_y) :
            flt_min_y = lst_flt_min_y[int_cumul_plot_index]
        if int_cumul_plot_index < len(lst_flt_max_y) :
            flt_max_y = lst_flt_max_y[int_cumul_plot_index]
        if int_cumul_plot_index < len(lst_flt_yticks_incr) :
            flt_yticks_incr = lst_flt_yticks_incr[int_cumul_plot_index]
        if int_cumul_plot_index < len(lst_flt_plot_width_in_inches) :
            flt_plot_width_in_inches = \
                lst_flt_plot_width_in_inches[int_cumul_plot_index]
        if int_cumul_plot_index < len(lst_flt_plot_height_in_inches) :
            flt_plot_height_in_inches = \
                lst_flt_plot_height_in_inches[int_cumul_plot_index]
        if int_cumul_plot_index < len(lst_int_dots_per_inch_for_png_and_jpg_plots) :
            int_dots_per_inch_for_png_and_jpg_plots = \
                lst_int_dots_per_inch_for_png_and_jpg_plots[int_cumul_plot_index]
        if int_cumul_plot_index < len(
                lst_flt_right_tails_circles_radius_in_pct_pnts) :
            flt_right_tails_circles_radius_in_pct_pnts = \
                lst_flt_right_tails_circles_radius_in_pct_pnts[
                    int_cumul_plot_index]
        if int_cumul_plot_index < len(lst_str_file_name_for_saving_plot) :
            str_file_name_for_saving_plot = \
                lst_str_file_name_for_saving_plot[int_cumul_plot_index]

        fig, ax = plt.subplots(figsize = (
            flt_plot_width_in_inches, flt_plot_height_in_inches),)

        if bool_is_plot_for_parametric_model :
            if bool_is_plot_for_parametric_lasso_model :
                str_plot_title_with_model_type = str_plot_title + \
                    STR_PLOT_TITLE_PARM_LASSO_GAM_SUFFIX
            else :
                str_plot_title_with_model_type = str_plot_title + \
                    STR_PLOT_TITLE_PARM_OLS_GAM_SUFFIX
        else :
            str_plot_title_with_model_type = str_plot_title + \
                STR_PLOT_TITLE_NPAR_SUFFIX
        if int_plot_title_font_size is None :
            plt.suptitle(str_plot_title_with_model_type,)
        else :
            plt.suptitle(str_plot_title_with_model_type,
                         fontsize = int_plot_title_font_size)

        if int_x_axis_label_font_size is None :
            plt.xlabel(str_x_axis_label)
        else :
            plt.xlabel(str_x_axis_label, fontsize=int_x_axis_label_font_size)
        
        if int_y_axis_label_font_size is None :
            plt.ylabel(str_y_axis_label)
        else :
            plt.ylabel(str_y_axis_label, fontsize=int_y_axis_label_font_size)

        if str_text_line_start_y_loc == 'upper' or \
           str_text_line_start_y_loc == 'lower' :
            if str_text_line_start_y_loc == 'upper' :
                flt_text_line_curr_y_loc_as_fraction = \
                    flt_text_line_upper_y_loc_as_fraction
                flt_text_line_delta_y_loc_as_fraction = \
                    -math.fabs(flt_text_line_abs_delta_y_loc_as_fraction)
                lst_str_text_line_content_ordered = \
                    lst_str_text_line_content
                lst_str_text_line_color_ordered = \
                    lst_str_text_line_color
            elif str_text_line_start_y_loc == 'lower' :
                flt_text_line_curr_y_loc_as_fraction = \
                    flt_text_line_lower_y_loc_as_fraction
                flt_text_line_delta_y_loc_as_fraction = \
                    +math.fabs(flt_text_line_abs_delta_y_loc_as_fraction)
                lst_str_text_line_content_ordered = [
                    str_text_line_content for str_text_line_content
                    in reversed(lst_str_text_line_content)]
                lst_str_text_line_color_ordered = [
                    tr_text_line_color for tr_text_line_color
                    in reversed(lst_str_text_line_color)]
            str_text_line_color = "black"
            for int_text_line_index in range(len(
                    lst_str_text_line_content_ordered)) :
                str_text_line_content = lst_str_text_line_content_ordered[
                    int_text_line_index]
                if int_text_line_index < len(lst_str_text_line_color_ordered) :
                    str_text_line_color = lst_str_text_line_color_ordered[
                        int_text_line_index]
                plt.figtext(
                    flt_text_line_x_loc_as_fraction,
                    flt_text_line_curr_y_loc_as_fraction,
                    str_text_line_content,
                    size = int_text_line_font_size,
                    color = str_text_line_color)
                flt_text_line_curr_y_loc_as_fraction += \
                    flt_text_line_delta_y_loc_as_fraction
                if not (flt_text_line_lower_y_loc_as_fraction <=
                        flt_text_line_curr_y_loc_as_fraction <=
                        flt_text_line_upper_y_loc_as_fraction) :
                    break

        if bool_is_chart_from_parametric_model is None :
            str_chart_id = "Chart ID {"
        elif bool_is_chart_from_parametric_model :
            str_chart_id = "Parametric Model Chart ID {"
        else :
            str_chart_id = "Non-parametric Model Chart ID {"
        if bool_display_chart_id and bool_include_mac_address_in_chart_id :
            str_chart_id += ':'.join(re.findall(
                '..', '%012x' % uuid.getnode())) + "; "
        if bool_display_chart_id :
            str_curr_gmt_date_time = str(datetime.datetime.now().astimezone(
                pytz.timezone('GMT'))) + "}"
            str_chart_id += str_curr_gmt_date_time
        if bool_display_chart_id :
            plt.figtext(0.1275, 0.89, str_chart_id,
                        size = int_chart_id_font_size, color = 'black')

        if lst_str_color_names_for_curves is None :
            lst_str_color_names_for_curves = [None] * \
                len(lst_str_legend_labels_for_curves)
        #if flt_right_tails_circles_radius_in_pct_pnts > 0. :
        flt_max_y += (flt_max_y - flt_min_y) * 0.000001
        flt_max_x += (flt_max_x - flt_min_x) * 0.000001
        flt_y_to_x_axes_ratio = (flt_max_y - flt_min_y) / \
            (flt_max_x - flt_min_x) * \
            (flt_plot_width_in_inches / flt_plot_height_in_inches) * \
            (1.4 / (4./3.))
        for (str_col_name_prefix,
             str_col_label,
             str_color_name,
             str_linestyle,
             flt_linewidth,
             bool_draw_right_tail_circle,
             str_annotation,
             int_annotation_font_size,
             bool_display_final_cumulative_percent_of_tally,
             int_final_cumulative_percent_of_tally_font_size,
             int_final_cumulative_percent_of_tally_rounding_precision,) in zip(
                 lst_str_vote_count_column_name_prefixes_to_use_for_curves,
                 lst_str_legend_labels_for_curves,
                 lst_str_color_names_for_curves,
                 lst_str_linestyles_for_curves,
                 lst_flt_linewidths_for_curves,
                 lst_bool_draw_right_tail_circle_for_curves,
                 lst_str_annotations_for_curves,
                 lst_int_annotation_font_sizes_for_curves,
                 lst_bool_display_final_cumulative_percent_of_tally_for_curves,
                 lst_int_final_cumulative_percent_of_tally_font_sizes_for_curves,
                 lst_int_final_cumulative_percent_of_tally_rounding_precision_for_curves,) : 
            # linestyle:
            # '-' solid line style
            # '--' dashed line style
            # '-.' dash-dot line style
            # ':' dotted line style
            plt.plot(
                df_sorted_data_panel_cumul_pct[
                    STR_COL_NAME_FOR_CUMUL_PLOT_X_AXIS],
                df_sorted_data_panel_cumul_pct[
                    str_col_name_prefix +
                    STR_COL_NAME_POSTFIX_FOR_CUMUL_PLOT_Y_AXIS],
                label = str_col_label,
                color = str_color_name,
                linestyle = str_linestyle,
                linewidth = flt_linewidth,)
            if bool_draw_right_tail_circle and \
               flt_right_tails_circles_radius_in_pct_pnts > 0. :
                flt_ellipse_x = df_sorted_data_panel_cumul_pct.iloc[-1][
                    STR_COL_NAME_FOR_CUMUL_PLOT_X_AXIS] - \
                    flt_right_tails_circles_radius_in_pct_pnts
                flt_ellipse_y = df_sorted_data_panel_cumul_pct.iloc[-1][
                    str_col_name_prefix +
                    STR_COL_NAME_POSTFIX_FOR_CUMUL_PLOT_Y_AXIS]
                flt_ellipse_w = 2.*flt_right_tails_circles_radius_in_pct_pnts
                flt_ellipse_h = 2.*flt_right_tails_circles_radius_in_pct_pnts*\
                    flt_y_to_x_axes_ratio
                ax.add_patch(Ellipse(
                    xy = (flt_ellipse_x, flt_ellipse_y),
                    width = flt_ellipse_w,
                    height = flt_ellipse_h,
                    edgecolor = 'gray',
                    facecolor='None',
                    linestyle=':',))
            if str_annotation != '' and int_annotation_font_size > 0 :
                flt_annotate_tip_x = df_sorted_data_panel_cumul_pct.iloc[-1][
                    STR_COL_NAME_FOR_CUMUL_PLOT_X_AXIS] - \
                    flt_right_tails_circles_radius_in_pct_pnts
                flt_annotate_tip_y = df_sorted_data_panel_cumul_pct.iloc[-1][
                    str_col_name_prefix +
                    STR_COL_NAME_POSTFIX_FOR_CUMUL_PLOT_Y_AXIS]
                flt_annotate_text_x = flt_annotate_tip_x - \
                    flt_right_tails_circles_radius_in_pct_pnts
                flt_annotate_text_y = flt_annotate_tip_y + \
                    flt_right_tails_circles_radius_in_pct_pnts * \
                        flt_y_to_x_axes_ratio
                ax.annotate(str_annotation,
                            xy = (flt_annotate_tip_x, flt_annotate_tip_y),
                            xycoords = 'data',
                            xytext = (flt_annotate_text_x,flt_annotate_text_y),
                            textcoords = 'data',
                            fontsize = int_annotation_font_size,
                            horizontalalignment = 'right',
                            verticalalignment='bottom',
                            arrowprops = dict(
                                width = 0,
                                headlength = 10,
                                headwidth = 3,
                                linestyle = 'solid',
                                fill = True,
                                edgecolor = 'gray',
                                facecolor = 'gray',
                                shrink = 0.))
            if bool_display_final_cumulative_percent_of_tally and \
               int_final_cumulative_percent_of_tally_font_size > 0 and \
               flt_max_x >= 100. :
                flt_final_cumulative_percent_of_tally_text_loc_x = \
                    df_sorted_data_panel_cumul_pct.iloc[-1][
                        STR_COL_NAME_FOR_CUMUL_PLOT_X_AXIS]
                flt_final_cumulative_percent_of_tally_text_loc_y = \
                    df_sorted_data_panel_cumul_pct.iloc[-1][
                        str_col_name_prefix +
                        STR_COL_NAME_POSTFIX_FOR_CUMUL_PLOT_Y_AXIS]
                if flt_min_y <= flt_final_cumulative_percent_of_tally_text_loc_y <= flt_max_y :
                    plt.figtext(
                        flt_final_cumulative_percent_of_tally_text_loc_x,
                        flt_final_cumulative_percent_of_tally_text_loc_y,
                        (" {0:0." +
                         str(int_final_cumulative_percent_of_tally_rounding_precision) +
                         "f}%").format(round(
                             flt_final_cumulative_percent_of_tally_text_loc_y,
                             int_final_cumulative_percent_of_tally_rounding_precision)),
                        transform = ax.transData,
                        size = int_final_cumulative_percent_of_tally_font_size,
                        color = 'black',)

        plt.axis((flt_min_x, flt_max_x, flt_min_y, flt_max_y))

        if int_x_axis_tick_labels_font_size is None :
            plt.xticks(np.arange(flt_min_x, flt_max_x, flt_xticks_incr),)
        else :
            plt.xticks(np.arange(flt_min_x, flt_max_x, flt_xticks_incr),
                       fontsize=int_x_axis_tick_labels_font_size,)

        if int_y_axis_tick_labels_font_size is None :
            plt.yticks(np.arange(flt_min_y, flt_max_y, flt_yticks_incr),)
        else :
            plt.yticks(np.arange(flt_min_y, flt_max_y, flt_yticks_incr),
                       fontsize=int_y_axis_tick_labels_font_size,)

        plt.grid(True)

        if int_legend_font_size is None :
            plt.legend(loc = str_legend_loc,)
        else :
            plt.legend(loc = str_legend_loc, prop = {
                "size" : int_legend_font_size },)

        if (bool_distribute_job_output_files_in_directories_by_type or
            bool_gather_all_job_files_in_one_directory) and \
            str_file_name_for_saving_plot is not None :

            if int_cumul_plot_index < len(lst_str_file_name_for_saving_plot) :
                str_indexed_file_name_for_saving_plot = \
                    str_file_name_for_saving_plot
            else :
                str_indexed_file_name_for_saving_plot = \
                    str_file_name_for_saving_plot + '_' + \
                    str(len(lst_str_file_name_for_saving_plot) - \
                        int_cumul_plot_index + 1).zfill(3)

            if bool_save_plot_to_png_file :
                if bool_is_plot_for_parametric_model :
                    str_output_plots_png_path = STR_OUTPUT_PLOTS_PARM_PNG_PATH
                else :
                    str_output_plots_png_path = STR_OUTPUT_PLOTS_NPAR_PNG_PATH
                str_png_file_name_ext = \
                    str_indexed_file_name_for_saving_plot + ".png"
                str_png_full_file_name_ext = os.path.join(
                    str_output_plots_png_path, str_png_file_name_ext)
                try :
                    if os.path.exists(str_png_full_file_name_ext) :
                        os.remove(str_png_full_file_name_ext)
                except Exception as exception :
                    print_exception_message(exception)
                try :
                    plt.savefig(
                        str_png_full_file_name_ext,
                        format = "png",
                        bbox_inches = "tight",
                        dpi = int_dots_per_inch_for_png_and_jpg_plots)
                except Exception as exception :
                    print_exception_message(exception)

                if bool_gather_all_job_files_in_one_directory and \
                   str_job_subdir_name is not None :
                    str_png_full_file_name_ext_from = str_png_full_file_name_ext
                    str_png_full_file_name_ext_to = os.path.join(
                        str_job_subdir_name, str_png_file_name_ext)
                    try :
                        if bool_distribute_job_output_files_in_directories_by_type :
                            shutil.copy(
                                str_png_full_file_name_ext_from,
                                str_png_full_file_name_ext_to)
                        else :
                            shutil.move(
                                str_png_full_file_name_ext_from,
                                str_png_full_file_name_ext_to)
                            pass
                    except Exception as exception :
                        print_exception_message(exception)

            if bool_save_plot_to_jpg_file :
                if bool_is_plot_for_parametric_model :
                    str_output_plots_jpg_path = STR_OUTPUT_PLOTS_PARM_JPG_PATH
                else :
                    str_output_plots_jpg_path = STR_OUTPUT_PLOTS_NPAR_JPG_PATH
                str_jpg_file_name_ext = \
                    str_indexed_file_name_for_saving_plot + ".jpg"
                str_jpg_full_file_name_ext = os.path.join(
                    str_output_plots_jpg_path, str_jpg_file_name_ext)
                try :
                    if os.path.exists(str_jpg_full_file_name_ext) :
                        os.remove(str_jpg_full_file_name_ext)
                except Exception as exception :
                    print_exception_message(exception)
                try :
                    plt.savefig(
                        str_jpg_full_file_name_ext,
                        format = "jpg",
                        bbox_inches = "tight",
                        dpi = int_dots_per_inch_for_png_and_jpg_plots)
                except Exception as exception :
                    print_exception_message(exception)

                if bool_gather_all_job_files_in_one_directory and \
                   str_job_subdir_name is not None :
                    str_jpg_full_file_name_ext_from = str_jpg_full_file_name_ext
                    str_jpg_full_file_name_ext_to = os.path.join(
                        str_job_subdir_name, str_jpg_file_name_ext)
                    try :
                        if bool_distribute_job_output_files_in_directories_by_type :
                            shutil.copy(
                                str_jpg_full_file_name_ext_from,
                                str_jpg_full_file_name_ext_to)
                        else :
                            shutil.move(
                                str_jpg_full_file_name_ext_from,
                                str_jpg_full_file_name_ext_to)
                            pass
                    except Exception as exception :
                        print_exception_message(exception)

            if bool_save_plot_to_pdf_file :
                if bool_is_plot_for_parametric_model :
                    str_output_plots_pdf_path = STR_OUTPUT_PLOTS_PARM_PDF_PATH
                else :
                    str_output_plots_pdf_path = STR_OUTPUT_PLOTS_NPAR_PDF_PATH
                str_pdf_file_name_ext = \
                    str_indexed_file_name_for_saving_plot + ".pdf"
                str_pdf_full_file_name_ext = os.path.join(
                    str_output_plots_pdf_path, str_pdf_file_name_ext)
                try :
                    if os.path.exists(str_pdf_full_file_name_ext) :
                        os.remove(str_pdf_full_file_name_ext)
                except Exception as exception :
                    print_exception_message(exception)
                try :
                    plt.savefig(
                        str_pdf_full_file_name_ext,
                        format = "pdf",
                        bbox_inches = "tight",)
                except Exception as exception :
                    print_exception_message(exception)

                if bool_gather_all_job_files_in_one_directory and \
                   str_job_subdir_name is not None :
                    str_pdf_full_file_name_ext_from = str_pdf_full_file_name_ext
                    str_pdf_full_file_name_ext_to = os.path.join(
                        str_job_subdir_name, str_pdf_file_name_ext)
                    try :
                        if bool_distribute_job_output_files_in_directories_by_type :
                            shutil.copy(
                                str_pdf_full_file_name_ext_from,
                                str_pdf_full_file_name_ext_to)
                        else :
                            shutil.move(
                                str_pdf_full_file_name_ext_from,
                                str_pdf_full_file_name_ext_to)
                            pass
                    except Exception as exception:
                        print_exception_message(exception)

            if bool_save_plot_to_svg_file :
                if bool_is_plot_for_parametric_model :
                    str_output_plots_svg_path = STR_OUTPUT_PLOTS_PARM_SVG_PATH
                else :
                    str_output_plots_svg_path = STR_OUTPUT_PLOTS_NPAR_SVG_PATH
                str_svg_file_name_ext = \
                    str_indexed_file_name_for_saving_plot + ".svg"
                str_svg_full_file_name_ext = os.path.join(
                    str_output_plots_svg_path, str_svg_file_name_ext)
                try :
                    if os.path.exists(str_svg_full_file_name_ext) :
                        os.remove(str_svg_full_file_name_ext)
                except Exception as exception :
                    print_exception_message(exception)
                try :
                    plt.savefig(
                        str_svg_full_file_name_ext,
                        format = "svg",
                        bbox_inches = "tight",)
                except Exception as exception :
                    print_exception_message(exception)

                if bool_gather_all_job_files_in_one_directory and \
                   str_job_subdir_name is not None :
                    str_svg_full_file_name_ext_from = str_svg_full_file_name_ext
                    str_svg_full_file_name_ext_to = os.path.join(
                        str_job_subdir_name, str_svg_file_name_ext)
                    try :
                        if bool_distribute_job_output_files_in_directories_by_type :
                            shutil.copy(
                                str_svg_full_file_name_ext_from,
                                str_svg_full_file_name_ext_to)
                        else :
                            shutil.move(
                                str_svg_full_file_name_ext_from,
                                str_svg_full_file_name_ext_to)
                            pass
                    except Exception as exception :
                        print_exception_message(exception)

        if bool_show_plot :
            plt.show()

        plt.close()

###############################################################################

def apply_parametric_model(
        #######################################################################
        # generate_predicted_counts
        #######################################################################
        str_actual_vote_counts_csv_file_name,
        lst_str_actual_all_vote_count_column_names = [],
        str_actual_residual_vote_count_column_name = None,
        lst_str_actual_residual_vote_count_column_names = [],
        #
        str_benchmark_vote_counts_csv_file_name = None,
        lst_str_benchmark_all_vote_count_column_names = [],
        str_benchmark_residual_vote_count_column_name = None,
        lst_str_benchmark_residual_vote_count_column_names = [],
        #
        bool_aggregate_vote_counts_by_county = False,
        bool_approximate_missing_benchmark_precincts_by_scaled_benchmark_county = False,
        bool_aggregate_missing_benchmark_precincts_into_new_residual_county = False,

        #######################################################################
        # sort_precincts_by_decreasing_predictability_for_parametric_model
        #######################################################################
        int_random_number_generator_seed_for_sorting = 0,
        bool_retain_sorting_criteria_columns = False,
        bool_compute_cumulative_actual_vote_count_fractions = False,
        bool_save_sorted_precincts_to_csv_file = False,
        str_csv_file_name_for_saving_sorted_precincts = None,
        #
        bool_distribute_job_output_files_in_directories_by_type = True,
        bool_gather_all_job_files_in_one_directory = False,
        str_job_subdir_name = None,

        #######################################################################
        bool_use_decimal_type = False,
        int_decimal_computational_precision = 1024, # [0; +Inf). # 1024 by default
        int_decimal_reporting_precision = 16, # [0; +Inf). # 16 by default
        int_max_num_iters_for_exact_hypergeom = 1_000_000, # [1; +Inf). # 1_000_000 by default
        int_max_num_iters_for_lanczos_approx_hypergeom = 500_000, # [1; +Inf). # 500_000 by default
        int_max_num_iters_for_spouge_approx_hypergeom = 500_000, # [1; +Inf). # 500_000 by default
        int_min_sample_size_for_approx_normal = 1_000, # [0; +Inf). # 1_000 by default
        flt_max_sample_sz_frac_of_pop_sz_for_approx_normal = 0.1, # [0.; 1.]. # 0.1 by default
        flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal = 0.1, # [0.; 1.]. # 0.1 by default
        flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal = 0.0, # [0.; +Inf). # 0.0 by default.
            # z = 2.575829303549 is for 99% in range mu +/- z*sigma
        #
        flt_lasso_alpha_l1_regularization_strength_term = None, # [0.; 1.]. # None by default (1.)
        int_lasso_maximum_number_of_iterations = 1_000_000,
        int_lasso_optimization_tolerance = 0.00001,
        flt_lasso_cv_length_of_alphas_regularization_path = 1., # (0.; +1.] # 1 by default (0.001); alpha_min / alpha_max
        int_lasso_cv_num_candidate_alphas_on_regularization_path = 0, # [0; +Inf] # 0 by default (100)
        int_lasso_cv_maximum_number_of_iterations = 1_000_000,
        int_lasso_cv_optimization_tolerance = 0.00001,
        int_lasso_cv_number_of_folds_in_cross_validation = 10,
        #
        bool_estimate_precinct_model_on_aggregated_vote_counts_by_county = False,
        bool_approximate_predictions_for_missing_benchmark_precincts_by_actual_values = False,
        bool_true_each_choice_with_all_other_choices__false_each_choice_with_each_choice = True,
        bool_compute_outlier_score_level_1 = True,
        bool_compute_outlier_score_level_2 = True,
        bool_compute_outlier_score_level_3 = True,
        #
        bool_estimate_model_parameters_diagnostics = False,
        bool_save_estimated_models_parameters_to_csv_file = False,
        str_csv_file_name_for_saving_estimated_models_parameters = None,
        bool_retain_predicted_vote_counts_columns_in_sorted_precincts_file = False,
        bool_save_outlier_score_stats_to_csv_file = False,
        str_csv_file_name_for_saving_outlier_score_stats = None,
        bool_save_model_diagnostics_to_csv_file = False,
        str_csv_file_name_for_saving_model_diagnostics = None,
        #
        lst_str_predicted_for_actual_vote_count_column_names = [],
        #
        lst_str_predicting_from_ln_of_incremented_actual_vote_count_column_names = [],
        lst_str_predicting_from_power_one_quarter_of_actual_vote_count_column_names = [],
        lst_str_predicting_from_square_root_of_actual_vote_count_column_names = [],
        lst_str_predicting_from_power_three_quarters_of_actual_vote_count_column_names = [],
        lst_str_predicting_from_actual_vote_count_column_names = [],
        lst_str_predicting_from_power_one_and_a_half_of_actual_vote_count_column_names = [],
        lst_str_predicting_from_squared_actual_vote_count_column_names = [],
        #
        lst_str_predicting_from_ln_of_incremented_benchmark_vote_count_column_names = [],
        lst_str_predicting_from_power_one_quarter_of_benchmark_vote_count_column_names = [],
        lst_str_predicting_from_square_root_of_benchmark_vote_count_column_names = [],
        lst_str_predicting_from_power_three_quarters_of_benchmark_vote_count_column_names = [],
        lst_str_predicting_from_benchmark_vote_count_column_names = [],
        lst_str_predicting_from_power_one_and_a_half_of_benchmark_vote_count_column_names = [],
        lst_str_predicting_from_squared_benchmark_vote_count_column_names = [],
        #
        # [(a,b) for a in lst_str_predicting_from_actual_vote_count_column_names
        #        for b in lst_str_predicting_from_actual_vote_count_column_names]
        lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_actual_vote_count_column_names = [],
        lst_pairs_str_predicting_from_square_root_of_actual_interactions_actual_vote_count_column_names = [],
        lst_pairs_str_predicting_from_actual_interactions_actual_vote_count_column_names = [],
        #
        # [(a,b) for a in lst_str_predicting_from_benchmark_vote_count_column_names
        #        for b in lst_str_predicting_from_benchmark_vote_count_column_names]
        lst_pairs_str_predicting_from_fourth_root_of_benchmark_interactions_benchmark_vote_count_column_names = [],
        lst_pairs_str_predicting_from_square_root_of_benchmark_interactions_benchmark_vote_count_column_names = [],
        lst_pairs_str_predicting_from_benchmark_interactions_benchmark_vote_count_column_names = [],
        #
        # [(a,b) for a in lst_str_predicting_from_actual_vote_count_column_names
        #        for b in lst_str_predicting_from_benchmark_vote_count_column_names]
        lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_benchmark_vote_count_column_names = [],
        lst_pairs_str_predicting_from_square_root_of_actual_interactions_benchmark_vote_count_column_names = [],
        lst_pairs_str_predicting_from_actual_interactions_benchmark_vote_count_column_names = [],
        #######################################################################
        bool_drop_predicted_variable_during_computation_of_predicting_tally = True,
        #######################################################################
        bool_predicting_from_ln_of_incremented_actual_tally = False,
        bool_predicting_from_power_one_quarter_of_actual_tally = False,
        bool_predicting_from_square_root_of_actual_tally = False,
        bool_predicting_from_power_three_quarters_of_actual_tally = False,
        bool_predicting_from_actual_tally = False,
        bool_predicting_from_power_one_and_a_half_of_actual_tally = False,
        bool_predicting_from_squared_actual_tally = False,
        #
        bool_predicting_from_ln_of_incremented_benchmark_tally = False,
        bool_predicting_from_power_one_quarter_of_benchmark_tally = False,
        bool_predicting_from_square_root_of_benchmark_tally = False,
        bool_predicting_from_power_three_quarters_of_benchmark_tally = False,
        bool_predicting_from_benchmark_tally = False,
        bool_predicting_from_power_one_and_a_half_of_benchmark_tally = False,
        bool_predicting_from_squared_benchmark_tally = False,
        #
        bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally = False,
        bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally = False,
        bool_predicting_from_actual_tally_interaction_benchmark_tally = False,
        #######################################################################
        bool_predicting_from_ln_of_incremented_actual_tally_interaction_county_indicator = False,
        bool_predicting_from_power_one_quarter_of_actual_tally_interaction_county_indicator = False,
        bool_predicting_from_square_root_of_actual_tally_interaction_county_indicator = False,
        bool_predicting_from_power_three_quarters_of_actual_tally_interaction_county_indicator = False,
        bool_predicting_from_actual_tally_interaction_county_indicator = False,
        bool_predicting_from_power_one_and_a_half_of_actual_tally_interaction_county_indicator = False,
        bool_predicting_from_squared_actual_tally_interaction_county_indicator = False,
        #
        bool_predicting_from_ln_of_incremented_benchmark_tally_interaction_county_indicator = False,
        bool_predicting_from_power_one_quarter_of_benchmark_tally_interaction_county_indicator = False,
        bool_predicting_from_square_root_of_benchmark_tally_interaction_county_indicator = False,
        bool_predicting_from_power_three_quarters_of_benchmark_tally_interaction_county_indicator = False,
        bool_predicting_from_benchmark_tally_interaction_county_indicator = False,
        bool_predicting_from_power_one_and_a_half_of_benchmark_tally_interaction_county_indicator = False,
        bool_predicting_from_squared_benchmark_tally_interaction_county_indicator = False,
        #
        bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator = False,
        bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator = False,
        bool_predicting_from_actual_tally_interaction_benchmark_tally_interaction_county_indicator = False,

        #######################################################################
        # generate_cumulative_plot
        #######################################################################
        str_plot_title = None,
        str_x_axis_label =
            "Cumulative Headcount Percent for Localities Sorted in Descending"+
            " Order of Predictability.",
        str_y_axis_label =
            "Contest Choice Percent. Right Tails Show the Impact of Outliers.",
        #######################################################################
        lst_str_vote_count_column_names_for_cumulative_tally_for_curves = None,
        lst_str_vote_count_column_name_prefixes_to_use_for_curves = None,
        lst_str_legend_labels_for_curves = None,
        lst_str_color_names_for_curves = None,
        lst_str_linestyles_for_curves = ['-'],
        lst_flt_linewidths_for_curves = [1.],
        lst_bool_draw_right_tail_circle_for_curves = [False],
        lst_str_annotations_for_curves = [''],
        lst_int_annotation_font_sizes_for_curves = [6],
        lst_bool_display_final_cumulative_percent_of_tally_for_curves = [False],
        lst_int_final_cumulative_percent_of_tally_font_sizes_for_curves = [6],
        lst_int_final_cumulative_percent_of_tally_rounding_precision_for_curves = [2],
        #######################################################################
        int_plot_title_font_size = None,
        int_x_axis_label_font_size = None,
        int_y_axis_label_font_size = None,
        int_x_axis_tick_labels_font_size = None,
        int_y_axis_tick_labels_font_size = None,
        int_legend_font_size = None,
        str_legend_loc = 'lower left',
        lst_str_text_line_content = [],
        lst_str_text_line_color = [],
        int_text_line_font_size = 7,
        flt_text_line_x_loc_as_fraction = 0.1275,
        flt_text_line_lower_y_loc_as_fraction = 0.14,
        flt_text_line_upper_y_loc_as_fraction = 0.85,
        flt_text_line_abs_delta_y_loc_as_fraction = 0.03,
        str_text_line_start_y_loc = "upper",
        bool_display_chart_id = False,
        int_chart_id_font_size = 7,
        bool_include_mac_address_in_chart_id = False,
        #######################################################################
        lst_flt_min_x = [0.],
        lst_flt_max_x = [100.],
        lst_flt_xticks_incr = [5.],
        lst_flt_min_y = [0.],
        lst_flt_max_y = [100.],
        lst_flt_yticks_incr = [5.],
        lst_flt_plot_width_in_inches = [6.4],
        lst_flt_plot_height_in_inches = [4.8],
        lst_int_dots_per_inch_for_png_and_jpg_plots = [1200],
        lst_flt_right_tails_circles_radius_in_pct_pnts = [0.],
        lst_str_file_name_for_saving_plot = [None,],
        #######################################################################
        bool_show_plot = False,
        bool_save_plot_to_png_file = False,
        bool_save_plot_to_jpg_file = False,
        bool_save_plot_to_pdf_file = False,
        bool_save_plot_to_svg_file = False,
        ) :
    '''<function description>
    
    <calling functions>
    <called functions>

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
    (df_actual_vote_counts,df_predicted_vote_counts)=generate_predicted_counts(
        str_actual_vote_counts_csv_file_name =
            str_actual_vote_counts_csv_file_name,
        lst_str_actual_all_vote_count_column_names =
            lst_str_actual_all_vote_count_column_names,
        str_actual_residual_vote_count_column_name =
            str_actual_residual_vote_count_column_name,
        lst_str_actual_residual_vote_count_column_names =
            lst_str_actual_residual_vote_count_column_names,
        #
        str_benchmark_vote_counts_csv_file_name =
            str_benchmark_vote_counts_csv_file_name,
        lst_str_benchmark_all_vote_count_column_names =
            lst_str_benchmark_all_vote_count_column_names,
        str_benchmark_residual_vote_count_column_name =
            str_benchmark_residual_vote_count_column_name,
        lst_str_benchmark_residual_vote_count_column_names =
            lst_str_benchmark_residual_vote_count_column_names,
        #
        bool_aggregate_vote_counts_by_county =
            bool_aggregate_vote_counts_by_county,
        bool_approximate_missing_benchmark_precincts_by_scaled_benchmark_county =
            bool_approximate_missing_benchmark_precincts_by_scaled_benchmark_county,
        bool_aggregate_missing_benchmark_precincts_into_new_residual_county =
            bool_aggregate_missing_benchmark_precincts_into_new_residual_county,
        #
        bool_produce_actual_vote_counts = False,
        #
        flt_lasso_alpha_l1_regularization_strength_term =
            flt_lasso_alpha_l1_regularization_strength_term,
        int_lasso_maximum_number_of_iterations =
            int_lasso_maximum_number_of_iterations,
        int_lasso_optimization_tolerance =
            int_lasso_optimization_tolerance,
        flt_lasso_cv_length_of_alphas_regularization_path =
            flt_lasso_cv_length_of_alphas_regularization_path,
        int_lasso_cv_num_candidate_alphas_on_regularization_path =
            int_lasso_cv_num_candidate_alphas_on_regularization_path,
        int_lasso_cv_maximum_number_of_iterations =
            int_lasso_cv_maximum_number_of_iterations,
        int_lasso_cv_optimization_tolerance =
            int_lasso_cv_optimization_tolerance,
        int_lasso_cv_number_of_folds_in_cross_validation =
            int_lasso_cv_number_of_folds_in_cross_validation,
        #
        bool_estimate_precinct_model_on_aggregated_vote_counts_by_county =
            bool_estimate_precinct_model_on_aggregated_vote_counts_by_county,
        bool_approximate_predictions_for_missing_benchmark_precincts_by_actual_values =
            bool_approximate_predictions_for_missing_benchmark_precincts_by_actual_values,
        #
        bool_estimate_model_parameters_diagnostics =
            bool_estimate_model_parameters_diagnostics,
        bool_save_estimated_models_parameters_to_csv_file =
            bool_save_estimated_models_parameters_to_csv_file,
        str_csv_file_name_for_saving_estimated_models_parameters =
            str_csv_file_name_for_saving_estimated_models_parameters,
        #
        bool_distribute_job_output_files_in_directories_by_type =
            bool_distribute_job_output_files_in_directories_by_type,
        bool_gather_all_job_files_in_one_directory =
            bool_gather_all_job_files_in_one_directory,
        str_job_subdir_name = str_job_subdir_name,
        #
        lst_str_predicted_for_actual_vote_count_column_names =
            lst_str_predicted_for_actual_vote_count_column_names,
        #
        lst_str_predicting_from_ln_of_incremented_actual_vote_count_column_names =
            lst_str_predicting_from_ln_of_incremented_actual_vote_count_column_names,
        lst_str_predicting_from_power_one_quarter_of_actual_vote_count_column_names =
            lst_str_predicting_from_power_one_quarter_of_actual_vote_count_column_names,
        lst_str_predicting_from_square_root_of_actual_vote_count_column_names =
            lst_str_predicting_from_square_root_of_actual_vote_count_column_names,
        lst_str_predicting_from_power_three_quarters_of_actual_vote_count_column_names =
            lst_str_predicting_from_power_three_quarters_of_actual_vote_count_column_names,
        lst_str_predicting_from_actual_vote_count_column_names =
            lst_str_predicting_from_actual_vote_count_column_names,
        lst_str_predicting_from_power_one_and_a_half_of_actual_vote_count_column_names =
            lst_str_predicting_from_power_one_and_a_half_of_actual_vote_count_column_names,
        lst_str_predicting_from_squared_actual_vote_count_column_names =
            lst_str_predicting_from_squared_actual_vote_count_column_names,
        #
        lst_str_predicting_from_ln_of_incremented_benchmark_vote_count_column_names =
            lst_str_predicting_from_ln_of_incremented_benchmark_vote_count_column_names,
        lst_str_predicting_from_power_one_quarter_of_benchmark_vote_count_column_names =
            lst_str_predicting_from_power_one_quarter_of_benchmark_vote_count_column_names,
        lst_str_predicting_from_square_root_of_benchmark_vote_count_column_names =
            lst_str_predicting_from_square_root_of_benchmark_vote_count_column_names,
        lst_str_predicting_from_power_three_quarters_of_benchmark_vote_count_column_names =
            lst_str_predicting_from_power_three_quarters_of_benchmark_vote_count_column_names,
        lst_str_predicting_from_benchmark_vote_count_column_names =
            lst_str_predicting_from_benchmark_vote_count_column_names,
        lst_str_predicting_from_power_one_and_a_half_of_benchmark_vote_count_column_names =
            lst_str_predicting_from_power_one_and_a_half_of_benchmark_vote_count_column_names,
        lst_str_predicting_from_squared_benchmark_vote_count_column_names =
            lst_str_predicting_from_squared_benchmark_vote_count_column_names,
        #
        # [(a,b) for a in lst_str_predicting_from_actual_vote_count_column_names
        #        for b in lst_str_predicting_from_actual_vote_count_column_names]
        lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_actual_vote_count_column_names =
            lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_actual_vote_count_column_names,
        lst_pairs_str_predicting_from_square_root_of_actual_interactions_actual_vote_count_column_names =
            lst_pairs_str_predicting_from_square_root_of_actual_interactions_actual_vote_count_column_names,
        lst_pairs_str_predicting_from_actual_interactions_actual_vote_count_column_names =
            lst_pairs_str_predicting_from_actual_interactions_actual_vote_count_column_names,
        #
        # [(a,b) for a in lst_str_predicting_from_benchmark_vote_count_column_names
        #        for b in lst_str_predicting_from_benchmark_vote_count_column_names]
        lst_pairs_str_predicting_from_fourth_root_of_benchmark_interactions_benchmark_vote_count_column_names =
            lst_pairs_str_predicting_from_fourth_root_of_benchmark_interactions_benchmark_vote_count_column_names,
        lst_pairs_str_predicting_from_square_root_of_benchmark_interactions_benchmark_vote_count_column_names =
            lst_pairs_str_predicting_from_square_root_of_benchmark_interactions_benchmark_vote_count_column_names,
        lst_pairs_str_predicting_from_benchmark_interactions_benchmark_vote_count_column_names =
            lst_pairs_str_predicting_from_benchmark_interactions_benchmark_vote_count_column_names,
        #
        # [(a,b) for a in lst_str_predicting_from_actual_vote_count_column_names
        #        for b in lst_str_predicting_from_benchmark_vote_count_column_names]
        lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_benchmark_vote_count_column_names =
            lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_benchmark_vote_count_column_names,
        lst_pairs_str_predicting_from_square_root_of_actual_interactions_benchmark_vote_count_column_names =
            lst_pairs_str_predicting_from_square_root_of_actual_interactions_benchmark_vote_count_column_names,
        lst_pairs_str_predicting_from_actual_interactions_benchmark_vote_count_column_names =
            lst_pairs_str_predicting_from_actual_interactions_benchmark_vote_count_column_names,
        #######################################################################
        bool_drop_predicted_variable_during_computation_of_predicting_tally =
            bool_drop_predicted_variable_during_computation_of_predicting_tally,
        #######################################################################
        bool_predicting_from_ln_of_incremented_actual_tally =
            bool_predicting_from_ln_of_incremented_actual_tally,
        bool_predicting_from_power_one_quarter_of_actual_tally =
            bool_predicting_from_power_one_quarter_of_actual_tally,
        bool_predicting_from_square_root_of_actual_tally =
            bool_predicting_from_square_root_of_actual_tally,
        bool_predicting_from_power_three_quarters_of_actual_tally =
            bool_predicting_from_power_three_quarters_of_actual_tally,
        bool_predicting_from_actual_tally =
            bool_predicting_from_actual_tally,
        bool_predicting_from_power_one_and_a_half_of_actual_tally =
            bool_predicting_from_power_one_and_a_half_of_actual_tally,
        bool_predicting_from_squared_actual_tally =
            bool_predicting_from_squared_actual_tally,
        #
        bool_predicting_from_ln_of_incremented_benchmark_tally =
            bool_predicting_from_ln_of_incremented_benchmark_tally,
        bool_predicting_from_power_one_quarter_of_benchmark_tally =
            bool_predicting_from_power_one_quarter_of_benchmark_tally,
        bool_predicting_from_square_root_of_benchmark_tally =
            bool_predicting_from_square_root_of_benchmark_tally,
        bool_predicting_from_power_three_quarters_of_benchmark_tally =
            bool_predicting_from_power_three_quarters_of_benchmark_tally,
        bool_predicting_from_benchmark_tally =
            bool_predicting_from_benchmark_tally,
        bool_predicting_from_power_one_and_a_half_of_benchmark_tally =
            bool_predicting_from_power_one_and_a_half_of_benchmark_tally,
        bool_predicting_from_squared_benchmark_tally =
            bool_predicting_from_squared_benchmark_tally,
        #
        bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally =
            bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally,
        bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally =
            bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally,
        bool_predicting_from_actual_tally_interaction_benchmark_tally =
            bool_predicting_from_actual_tally_interaction_benchmark_tally,
        #######################################################################
        bool_predicting_from_ln_of_incremented_actual_tally_interaction_county_indicator =
            bool_predicting_from_ln_of_incremented_actual_tally_interaction_county_indicator,
        bool_predicting_from_power_one_quarter_of_actual_tally_interaction_county_indicator =
            bool_predicting_from_power_one_quarter_of_actual_tally_interaction_county_indicator,
        bool_predicting_from_square_root_of_actual_tally_interaction_county_indicator =
            bool_predicting_from_square_root_of_actual_tally_interaction_county_indicator,
        bool_predicting_from_power_three_quarters_of_actual_tally_interaction_county_indicator =
            bool_predicting_from_power_three_quarters_of_actual_tally_interaction_county_indicator,
        bool_predicting_from_actual_tally_interaction_county_indicator =
            bool_predicting_from_actual_tally_interaction_county_indicator,
        bool_predicting_from_power_one_and_a_half_of_actual_tally_interaction_county_indicator =
            bool_predicting_from_power_one_and_a_half_of_actual_tally_interaction_county_indicator,
        bool_predicting_from_squared_actual_tally_interaction_county_indicator =
            bool_predicting_from_squared_actual_tally_interaction_county_indicator,
        #
        bool_predicting_from_ln_of_incremented_benchmark_tally_interaction_county_indicator =
            bool_predicting_from_ln_of_incremented_benchmark_tally_interaction_county_indicator,
        bool_predicting_from_power_one_quarter_of_benchmark_tally_interaction_county_indicator =
            bool_predicting_from_power_one_quarter_of_benchmark_tally_interaction_county_indicator,
        bool_predicting_from_square_root_of_benchmark_tally_interaction_county_indicator =
            bool_predicting_from_square_root_of_benchmark_tally_interaction_county_indicator,
        bool_predicting_from_power_three_quarters_of_benchmark_tally_interaction_county_indicator =
            bool_predicting_from_power_three_quarters_of_benchmark_tally_interaction_county_indicator,
        bool_predicting_from_benchmark_tally_interaction_county_indicator =
            bool_predicting_from_benchmark_tally_interaction_county_indicator,
        bool_predicting_from_power_one_and_a_half_of_benchmark_tally_interaction_county_indicator =
            bool_predicting_from_power_one_and_a_half_of_benchmark_tally_interaction_county_indicator,
        bool_predicting_from_squared_benchmark_tally_interaction_county_indicator =
            bool_predicting_from_squared_benchmark_tally_interaction_county_indicator,
        #
        bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator =
            bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator,
        bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator =
            bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator,
        bool_predicting_from_actual_tally_interaction_benchmark_tally_interaction_county_indicator =
            bool_predicting_from_actual_tally_interaction_benchmark_tally_interaction_county_indicator,)

    if bool_save_model_diagnostics_to_csv_file :
        save_model_diagnostics_to_csv_file(
            df_actual_vote_counts = df_actual_vote_counts,
            df_predicted_vote_counts = df_predicted_vote_counts,
            lst_str_predicted_for_actual_vote_count_column_names =
                lst_str_predicted_for_actual_vote_count_column_names,
            str_csv_file_name_for_saving_model_diagnostics =
                str_csv_file_name_for_saving_model_diagnostics,
            bool_distribute_job_output_files_in_directories_by_type =
                bool_distribute_job_output_files_in_directories_by_type,
            bool_gather_all_job_files_in_one_directory =
                bool_gather_all_job_files_in_one_directory,
            str_job_subdir_name = str_job_subdir_name,)

    df_sorted_data_panel = \
        sort_precincts_by_decreasing_predictability_for_parametric_model(
            df_actual_vote_counts = df_actual_vote_counts,
            df_predicted_vote_counts = df_predicted_vote_counts,
            lst_str_predicted_for_actual_vote_count_column_names =
                lst_str_predicted_for_actual_vote_count_column_names,
            int_random_number_generator_seed_for_sorting =
                int_random_number_generator_seed_for_sorting,
            bool_use_decimal_type = bool_use_decimal_type,
            int_decimal_computational_precision =
                int_decimal_computational_precision,
            int_decimal_reporting_precision =
                int_decimal_reporting_precision,
            int_max_num_iters_for_exact_hypergeom =
                int_max_num_iters_for_exact_hypergeom,
            int_max_num_iters_for_lanczos_approx_hypergeom =
                int_max_num_iters_for_lanczos_approx_hypergeom,
            int_max_num_iters_for_spouge_approx_hypergeom =
                int_max_num_iters_for_spouge_approx_hypergeom,
            int_min_sample_size_for_approx_normal =
                int_min_sample_size_for_approx_normal,
            flt_max_sample_sz_frac_of_pop_sz_for_approx_normal =
                flt_max_sample_sz_frac_of_pop_sz_for_approx_normal,
            flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal =
                flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal,
            flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal =
                flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal,
            bool_true_each_choice_with_all_other_choices__false_each_choice_with_each_choice =
                bool_true_each_choice_with_all_other_choices__false_each_choice_with_each_choice,
            bool_compute_outlier_score_level_1 =
                bool_compute_outlier_score_level_1,
            bool_compute_outlier_score_level_2 =
                bool_compute_outlier_score_level_2,
            bool_compute_outlier_score_level_3 =
                bool_compute_outlier_score_level_3,
            bool_retain_predicted_vote_counts_columns_in_sorted_precincts_file =
                bool_retain_predicted_vote_counts_columns_in_sorted_precincts_file,
            bool_retain_sorting_criteria_columns =
                bool_retain_sorting_criteria_columns,
            bool_compute_cumulative_actual_vote_count_fractions =
                bool_compute_cumulative_actual_vote_count_fractions,
            bool_save_sorted_precincts_to_csv_file =
                bool_save_sorted_precincts_to_csv_file,
            str_csv_file_name_for_saving_sorted_precincts =
                str_csv_file_name_for_saving_sorted_precincts,
            bool_save_outlier_score_stats_to_csv_file =
                bool_save_outlier_score_stats_to_csv_file,
            str_csv_file_name_for_saving_outlier_score_stats =
                str_csv_file_name_for_saving_outlier_score_stats,
            bool_distribute_job_output_files_in_directories_by_type =
                bool_distribute_job_output_files_in_directories_by_type,
            bool_gather_all_job_files_in_one_directory =
                bool_gather_all_job_files_in_one_directory,
            str_job_subdir_name = str_job_subdir_name,)

    if str_plot_title is None :
        str_plot_title = str_actual_vote_counts_csv_file_name
    if lst_str_vote_count_column_names_for_cumulative_tally_for_curves is None :
        lst_str_vote_count_column_names_for_cumulative_tally_for_curves = \
            lst_str_predicted_for_actual_vote_count_column_names
    if lst_str_vote_count_column_name_prefixes_to_use_for_curves is None :
        lst_str_vote_count_column_name_prefixes_to_use_for_curves = \
            lst_str_vote_count_column_names_for_cumulative_tally_for_curves
    if lst_str_legend_labels_for_curves is None :
        lst_str_legend_labels_for_curves = \
            lst_str_vote_count_column_name_prefixes_to_use_for_curves
    if lst_str_color_names_for_curves is None :
        lst_str_color_names_for_curves = \
            ["black"] * len(
                lst_str_vote_count_column_name_prefixes_to_use_for_curves)

    if (flt_lasso_alpha_l1_regularization_strength_term is None and
        flt_lasso_cv_length_of_alphas_regularization_path == 1. and
        int_lasso_cv_num_candidate_alphas_on_regularization_path == 0) or \
       (flt_lasso_alpha_l1_regularization_strength_term is not None and
        flt_lasso_alpha_l1_regularization_strength_term == 0. and
        flt_lasso_cv_length_of_alphas_regularization_path == 1. and
        int_lasso_cv_num_candidate_alphas_on_regularization_path == 0) :
        bool_is_plot_for_parametric_lasso_model = False
    else :
        bool_is_plot_for_parametric_lasso_model = True

    generate_cumulative_plot(
        bool_is_plot_for_parametric_model = True,
        bool_is_plot_for_parametric_lasso_model =
            bool_is_plot_for_parametric_lasso_model,
        df_sorted_data_panel = df_sorted_data_panel,
        str_plot_title = str_plot_title,
        str_x_axis_label = str_x_axis_label,
        str_y_axis_label = str_y_axis_label,
        #######################################################################
        lst_str_vote_count_column_names_for_cumulative_tally_for_curves =
            lst_str_vote_count_column_names_for_cumulative_tally_for_curves,
        lst_str_vote_count_column_name_prefixes_to_use_for_curves =
            lst_str_vote_count_column_name_prefixes_to_use_for_curves,
        lst_str_legend_labels_for_curves = lst_str_legend_labels_for_curves,
        lst_str_color_names_for_curves = lst_str_color_names_for_curves,
        lst_str_linestyles_for_curves = lst_str_linestyles_for_curves,
        lst_flt_linewidths_for_curves = lst_flt_linewidths_for_curves,
        lst_bool_draw_right_tail_circle_for_curves =
            lst_bool_draw_right_tail_circle_for_curves,
        lst_str_annotations_for_curves = lst_str_annotations_for_curves,
        lst_int_annotation_font_sizes_for_curves =
            lst_int_annotation_font_sizes_for_curves,
        lst_bool_display_final_cumulative_percent_of_tally_for_curves =
            lst_bool_display_final_cumulative_percent_of_tally_for_curves,
        lst_int_final_cumulative_percent_of_tally_font_sizes_for_curves =
            lst_int_final_cumulative_percent_of_tally_font_sizes_for_curves,
        lst_int_final_cumulative_percent_of_tally_rounding_precision_for_curves =
            lst_int_final_cumulative_percent_of_tally_rounding_precision_for_curves,
        #######################################################################
        int_plot_title_font_size = int_plot_title_font_size,
        int_x_axis_label_font_size = int_x_axis_label_font_size,
        int_y_axis_label_font_size = int_y_axis_label_font_size,
        int_x_axis_tick_labels_font_size = int_x_axis_tick_labels_font_size,
        int_y_axis_tick_labels_font_size = int_y_axis_tick_labels_font_size,
        int_legend_font_size = int_legend_font_size,
        str_legend_loc = str_legend_loc,
        lst_str_text_line_content = lst_str_text_line_content,
        lst_str_text_line_color = lst_str_text_line_color,
        int_text_line_font_size = int_text_line_font_size,
        flt_text_line_x_loc_as_fraction = flt_text_line_x_loc_as_fraction,
        flt_text_line_lower_y_loc_as_fraction =
            flt_text_line_lower_y_loc_as_fraction,
        flt_text_line_upper_y_loc_as_fraction =
            flt_text_line_upper_y_loc_as_fraction,
        flt_text_line_abs_delta_y_loc_as_fraction =
            flt_text_line_abs_delta_y_loc_as_fraction,
        str_text_line_start_y_loc = str_text_line_start_y_loc,
        bool_display_chart_id = bool_display_chart_id,
        bool_is_chart_from_parametric_model = True,
        int_chart_id_font_size = int_chart_id_font_size,
        bool_include_mac_address_in_chart_id =
            bool_include_mac_address_in_chart_id,
        #######################################################################
        lst_flt_min_x = lst_flt_min_x,
        lst_flt_max_x = lst_flt_max_x,
        lst_flt_xticks_incr = lst_flt_xticks_incr,
        lst_flt_min_y = lst_flt_min_y,
        lst_flt_max_y = lst_flt_max_y,
        lst_flt_yticks_incr = lst_flt_yticks_incr,
        lst_flt_plot_width_in_inches = lst_flt_plot_width_in_inches,
        lst_flt_plot_height_in_inches = lst_flt_plot_height_in_inches,
        lst_int_dots_per_inch_for_png_and_jpg_plots =
            lst_int_dots_per_inch_for_png_and_jpg_plots,
        lst_flt_right_tails_circles_radius_in_pct_pnts =
            lst_flt_right_tails_circles_radius_in_pct_pnts,
        lst_str_file_name_for_saving_plot =
            lst_str_file_name_for_saving_plot,
        #######################################################################
        bool_show_plot = bool_show_plot,
        bool_save_plot_to_png_file = bool_save_plot_to_png_file,
        bool_save_plot_to_jpg_file = bool_save_plot_to_jpg_file,
        bool_save_plot_to_pdf_file = bool_save_plot_to_pdf_file,
        bool_save_plot_to_svg_file = bool_save_plot_to_svg_file,
        bool_distribute_job_output_files_in_directories_by_type =
            bool_distribute_job_output_files_in_directories_by_type,
        bool_gather_all_job_files_in_one_directory =
            bool_gather_all_job_files_in_one_directory,
        str_job_subdir_name = str_job_subdir_name,)

###############################################################################

def apply_non_parametric_model(
        #######################################################################
        # generate_merged_counts
        #######################################################################
        str_actual_vote_counts_csv_file_name,
        lst_str_actual_all_vote_count_column_names = [],
        str_actual_residual_vote_count_column_name = None,
        lst_str_actual_residual_vote_count_column_names = [],
        #
        str_benchmark_vote_counts_csv_file_name = None,
        lst_str_benchmark_all_vote_count_column_names = [],
        str_benchmark_residual_vote_count_column_name = None,
        lst_str_benchmark_residual_vote_count_column_names = [],
        #
        bool_aggregate_vote_counts_by_county = False,
        bool_approximate_missing_benchmark_precincts_by_scaled_benchmark_county = False,
        bool_aggregate_missing_benchmark_precincts_into_new_residual_county = False,
        #
        int_random_number_generator_seed_for_sorting = 0,
        bool_retain_sorting_criteria_columns = False,
        bool_compute_cumulative_actual_vote_count_fractions = False,
        bool_save_sorted_precincts_to_csv_file = False,
        #
        str_csv_file_name_for_saving_sorted_precincts = None,
        #
        bool_distribute_job_output_files_in_directories_by_type = True,
        bool_gather_all_job_files_in_one_directory = False,
        #
        str_job_subdir_name = None,

        #######################################################################
        # sort_precincts_by_decreasing_predictability_for_non_parametric_model
        #######################################################################
        int_max_num_two_ways_passes = 1, # integer in [1; +Inf]
        flt_window_size_scaling_factor = .5, # 0 < f < 1
        lst_str_all_predicting_predicted_vote_count_column_names = [],

        #######################################################################
        # generate_cumulative_plot
        #######################################################################
        str_plot_title = None,
        str_x_axis_label =
            "Cumulative Headcount Percent for Localities Sorted in Descending"+
            " Order of Predictability.",
        str_y_axis_label =
            "Contest Choice Percent. Right Tails Show the Impact of Outliers.",
        #######################################################################
        lst_str_vote_count_column_names_for_cumulative_tally_for_curves = None,
        lst_str_vote_count_column_name_prefixes_to_use_for_curves = None,
        lst_str_legend_labels_for_curves = None,
        lst_str_color_names_for_curves = None,
        lst_str_linestyles_for_curves = ['-'],
        lst_flt_linewidths_for_curves = [1.],
        lst_bool_draw_right_tail_circle_for_curves = [False],
        lst_str_annotations_for_curves = [''],
        lst_int_annotation_font_sizes_for_curves = [6],
        lst_bool_display_final_cumulative_percent_of_tally_for_curves = [False],
        lst_int_final_cumulative_percent_of_tally_font_sizes_for_curves = [6],
        lst_int_final_cumulative_percent_of_tally_rounding_precision_for_curves = [2],
        #######################################################################
        int_plot_title_font_size = None,
        int_x_axis_label_font_size = None,
        int_y_axis_label_font_size = None,
        int_x_axis_tick_labels_font_size = None,
        int_y_axis_tick_labels_font_size = None,
        int_legend_font_size = None,
        str_legend_loc = 'lower left',
        lst_str_text_line_content = [],
        lst_str_text_line_color = [],
        int_text_line_font_size = 7,
        flt_text_line_x_loc_as_fraction = 0.1275,
        flt_text_line_lower_y_loc_as_fraction = 0.14,
        flt_text_line_upper_y_loc_as_fraction = 0.85,
        flt_text_line_abs_delta_y_loc_as_fraction = 0.03,
        str_text_line_start_y_loc = "upper",
        bool_display_chart_id = False,
        int_chart_id_font_size = 7,
        bool_include_mac_address_in_chart_id = False,
        #######################################################################
        lst_flt_min_x = [0.],
        lst_flt_max_x = [100.],
        lst_flt_xticks_incr = [5.],
        lst_flt_min_y = [0.],
        lst_flt_max_y = [100.],
        lst_flt_yticks_incr = [5.],
        lst_flt_plot_width_in_inches = [6.4],
        lst_flt_plot_height_in_inches = [4.8],
        lst_int_dots_per_inch_for_png_and_jpg_plots = [1200],
        lst_flt_right_tails_circles_radius_in_pct_pnts = [0.],
        lst_str_file_name_for_saving_plot = [None,],
        #######################################################################
        bool_show_plot = False,
        bool_save_plot_to_png_file = False,
        bool_save_plot_to_jpg_file = False,
        bool_save_plot_to_pdf_file = False,
        bool_save_plot_to_svg_file = False,
        ) :
    '''<function description>
    
    <calling functions>
    <called functions>

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
    df_merged_data_panel = generate_merged_counts(
        str_actual_vote_counts_csv_file_name =
            str_actual_vote_counts_csv_file_name,
        lst_str_actual_all_vote_count_column_names =
            lst_str_actual_all_vote_count_column_names,
        str_actual_residual_vote_count_column_name =
            str_actual_residual_vote_count_column_name,
        lst_str_actual_residual_vote_count_column_names =
            lst_str_actual_residual_vote_count_column_names,
        #
        str_benchmark_vote_counts_csv_file_name =
            str_benchmark_vote_counts_csv_file_name,
        lst_str_benchmark_all_vote_count_column_names =
            lst_str_benchmark_all_vote_count_column_names,
        str_benchmark_residual_vote_count_column_name =
            str_benchmark_residual_vote_count_column_name,
        lst_str_benchmark_residual_vote_count_column_names =
            lst_str_benchmark_residual_vote_count_column_names,
        #
        bool_aggregate_vote_counts_by_county =
            bool_aggregate_vote_counts_by_county,
        bool_approximate_missing_benchmark_precincts_by_scaled_benchmark_county =
            bool_approximate_missing_benchmark_precincts_by_scaled_benchmark_county,
        bool_aggregate_missing_benchmark_precincts_into_new_residual_county =
            bool_aggregate_missing_benchmark_precincts_into_new_residual_county,
        )

    if lst_str_all_predicting_predicted_vote_count_column_names is None or \
       len(lst_str_all_predicting_predicted_vote_count_column_names) == 0 :
        lst_str_all_predicting_predicted_vote_count_column_names = \
            lst_str_actual_all_vote_count_column_names + \
                lst_str_benchmark_all_vote_count_column_names

    df_sorted_data_panel = \
        sort_precincts_by_decreasing_predictability_for_non_parametric_model(
            df_data_panel = df_merged_data_panel,
            lst_str_all_predicting_predicted_vote_count_column_names =
                lst_str_all_predicting_predicted_vote_count_column_names,
            int_max_num_two_ways_passes = int_max_num_two_ways_passes,
            flt_window_size_scaling_factor =
                flt_window_size_scaling_factor,
            int_random_number_generator_seed_for_sorting =
                int_random_number_generator_seed_for_sorting,
            bool_retain_sorting_criteria_columns =
                bool_retain_sorting_criteria_columns,
            bool_compute_cumulative_actual_vote_count_fractions =
                bool_compute_cumulative_actual_vote_count_fractions,
            bool_save_sorted_precincts_to_csv_file =
                bool_save_sorted_precincts_to_csv_file,
            str_csv_file_name_for_saving_sorted_precincts =
                str_csv_file_name_for_saving_sorted_precincts,
            bool_distribute_job_output_files_in_directories_by_type =
                bool_distribute_job_output_files_in_directories_by_type,
            bool_gather_all_job_files_in_one_directory =
                bool_gather_all_job_files_in_one_directory,
            str_job_subdir_name = str_job_subdir_name,)

    if str_plot_title is None :
        str_plot_title = str_actual_vote_counts_csv_file_name
    if lst_str_vote_count_column_names_for_cumulative_tally_for_curves is None:
        lst_str_vote_count_column_names_for_cumulative_tally_for_curves = \
            lst_str_actual_all_vote_count_column_names
    if lst_str_vote_count_column_name_prefixes_to_use_for_curves is None :
        lst_str_vote_count_column_name_prefixes_to_use_for_curves = \
            lst_str_vote_count_column_names_for_cumulative_tally_for_curves
    if lst_str_legend_labels_for_curves is None :
        lst_str_legend_labels_for_curves = \
            lst_str_vote_count_column_name_prefixes_to_use_for_curves
    if lst_str_color_names_for_curves is None :
        lst_str_color_names_for_curves = \
            ["black"] * len(
                lst_str_vote_count_column_name_prefixes_to_use_for_curves)

    generate_cumulative_plot(
        bool_is_plot_for_parametric_model = False,
        bool_is_plot_for_parametric_lasso_model = False,
        df_sorted_data_panel = df_sorted_data_panel,
        str_plot_title = str_plot_title,
        str_x_axis_label = str_x_axis_label,
        str_y_axis_label = str_y_axis_label,
        #######################################################################
        lst_str_vote_count_column_names_for_cumulative_tally_for_curves =
            lst_str_vote_count_column_names_for_cumulative_tally_for_curves,
        lst_str_vote_count_column_name_prefixes_to_use_for_curves =
           lst_str_vote_count_column_name_prefixes_to_use_for_curves,
        lst_str_legend_labels_for_curves = lst_str_legend_labels_for_curves,
        lst_str_color_names_for_curves = lst_str_color_names_for_curves,
        lst_str_linestyles_for_curves = lst_str_linestyles_for_curves,
        lst_flt_linewidths_for_curves = lst_flt_linewidths_for_curves,
        lst_bool_draw_right_tail_circle_for_curves =
            lst_bool_draw_right_tail_circle_for_curves,
        lst_str_annotations_for_curves = lst_str_annotations_for_curves,
        lst_int_annotation_font_sizes_for_curves =
            lst_int_annotation_font_sizes_for_curves,
        lst_bool_display_final_cumulative_percent_of_tally_for_curves =
            lst_bool_display_final_cumulative_percent_of_tally_for_curves,
        lst_int_final_cumulative_percent_of_tally_font_sizes_for_curves =
            lst_int_final_cumulative_percent_of_tally_font_sizes_for_curves,
        lst_int_final_cumulative_percent_of_tally_rounding_precision_for_curves =
            lst_int_final_cumulative_percent_of_tally_rounding_precision_for_curves,
        #######################################################################
        int_plot_title_font_size = int_plot_title_font_size,
        int_x_axis_label_font_size = int_x_axis_label_font_size,
        int_y_axis_label_font_size = int_y_axis_label_font_size,
        int_x_axis_tick_labels_font_size = int_x_axis_tick_labels_font_size,
        int_y_axis_tick_labels_font_size = int_y_axis_tick_labels_font_size,
        int_legend_font_size = int_legend_font_size,
        str_legend_loc = str_legend_loc,
        lst_str_text_line_content = lst_str_text_line_content,
        lst_str_text_line_color = lst_str_text_line_color,
        int_text_line_font_size = int_text_line_font_size,
        flt_text_line_x_loc_as_fraction = flt_text_line_x_loc_as_fraction,
        flt_text_line_lower_y_loc_as_fraction =
            flt_text_line_lower_y_loc_as_fraction,
        flt_text_line_upper_y_loc_as_fraction =
            flt_text_line_upper_y_loc_as_fraction,
        flt_text_line_abs_delta_y_loc_as_fraction =
            flt_text_line_abs_delta_y_loc_as_fraction,
        str_text_line_start_y_loc = str_text_line_start_y_loc,
        bool_display_chart_id = bool_display_chart_id,
        bool_is_chart_from_parametric_model = False,
        int_chart_id_font_size = int_chart_id_font_size,
        bool_include_mac_address_in_chart_id =
            bool_include_mac_address_in_chart_id,
        #######################################################################
        lst_flt_min_x = lst_flt_min_x,
        lst_flt_max_x = lst_flt_max_x,
        lst_flt_xticks_incr = lst_flt_xticks_incr,
        lst_flt_min_y = lst_flt_min_y,
        lst_flt_max_y = lst_flt_max_y,
        lst_flt_yticks_incr = lst_flt_yticks_incr,
        lst_flt_plot_width_in_inches = lst_flt_plot_width_in_inches,
        lst_flt_plot_height_in_inches = lst_flt_plot_height_in_inches,
        lst_int_dots_per_inch_for_png_and_jpg_plots =
            lst_int_dots_per_inch_for_png_and_jpg_plots,
        lst_flt_right_tails_circles_radius_in_pct_pnts =
            lst_flt_right_tails_circles_radius_in_pct_pnts,
        lst_str_file_name_for_saving_plot =
            lst_str_file_name_for_saving_plot,
        #######################################################################
        bool_show_plot = bool_show_plot,
        bool_save_plot_to_png_file = bool_save_plot_to_png_file,
        bool_save_plot_to_jpg_file = bool_save_plot_to_jpg_file,
        bool_save_plot_to_pdf_file = bool_save_plot_to_pdf_file,
        bool_save_plot_to_svg_file = bool_save_plot_to_svg_file,
        bool_distribute_job_output_files_in_directories_by_type =
            bool_distribute_job_output_files_in_directories_by_type,
        bool_gather_all_job_files_in_one_directory =
            bool_gather_all_job_files_in_one_directory,
        str_job_subdir_name = str_job_subdir_name,)

###############################################################################

def copy_input_files_to_job_directory(
        str_input_batch_bat_file_name_ext,
        str_input_batch_csv_file_name_ext,
        str_input_job_json_file_name,
        str_actual_vote_counts_csv_file_name,
        str_benchmark_vote_counts_csv_file_name,
        bool_gather_all_job_files_in_one_directory,) :
    '''<function description>
    
    <calling functions>
    <called functions>

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
    if bool_gather_all_job_files_in_one_directory :
        str_job_subdir_name = os.path.join(
            ".", STR_OUTPUT_JOBS_DIR_PATH, str_input_job_json_file_name)

        # Clean or create job-specific directory.
        if os.path.exists(str_job_subdir_name) and \
           os.path.isdir(str_job_subdir_name):
            # Clean the directory
            for str_dir_elem_name in os.listdir(str_job_subdir_name) :
                str_dir_elem_path = os.path.join(
                    str_job_subdir_name, str_dir_elem_name)
                try :
                    if os.path.isfile(str_dir_elem_path) or \
                       os.path.islink(str_dir_elem_path) :
                        os.unlink(str_dir_elem_path)
                    elif os.path.isdir(str_dir_elem_path) :
                        shutil.rmtree(str_dir_elem_path)
                except Exception as exception :
                    print_exception_message(exception)
        else :
            # Create the directory.
            if not os.path.exists(str_job_subdir_name) :
                os.makedirs(str_job_subdir_name)

        # Copy Batch BAT file to the job-specific output directory.
        if str_input_batch_bat_file_name_ext is not None :
            if str_input_batch_bat_file_name_ext.startswith(".") or \
               "/" in str_input_batch_bat_file_name_ext or \
               "\\" in str_input_batch_bat_file_name_ext or \
               ":" in str_input_batch_bat_file_name_ext :
                # Already an absolute or relative path with name, not just file name.
                str_input_batch_bat_full_file_name_ext_from = \
                    str_input_batch_bat_file_name_ext
            else :
                str_input_batch_bat_full_file_name_ext_from = os.path.join(
                    ".", STR_INPUT_BATCHES_BAT_PATH,
                    str_input_batch_bat_file_name_ext)
            str_input_batch_bat_file_basename_ext = os.path.basename(
                str_input_batch_bat_file_name_ext)
            str_input_batch_bat_full_file_name_ext1_to = os.path.join(
                str_job_subdir_name, str_input_batch_bat_file_basename_ext)
            str_input_batch_bat_full_file_name_ext2_to = os.path.join(
                STR_OUTPUT_JOBS_DIR_PATH, str_input_batch_bat_file_basename_ext)
            try :
                if os.path.exists(str_input_batch_bat_full_file_name_ext_from) :
                    shutil.copy(
                        str_input_batch_bat_full_file_name_ext_from,
                        str_input_batch_bat_full_file_name_ext1_to)
                    shutil.copy(
                        str_input_batch_bat_full_file_name_ext_from,
                        str_input_batch_bat_full_file_name_ext2_to)
            except Exception as exception :
                print_exception_message(exception)

        # Copy Batch CSV file to the job-specific output directory.
        if str_input_batch_csv_file_name_ext is not None :
            if str_input_batch_csv_file_name_ext.startswith(".") or \
               "/" in str_input_batch_csv_file_name_ext or \
               "\\" in str_input_batch_csv_file_name_ext or \
               ":" in str_input_batch_csv_file_name_ext :
                # Already an absolute or relative path with name, not just file name.
                str_input_batch_csv_full_file_name_ext_from = \
                    str_input_batch_csv_file_name_ext
            else :
                str_input_batch_csv_full_file_name_ext_from = os.path.join(
                    ".", STR_INPUT_BATCHES_CSV_PATH,
                    str_input_batch_csv_file_name_ext)
            str_input_batch_csv_full_file_name_ext_to = os.path.join(
                str_job_subdir_name,
                os.path.basename(str_input_batch_csv_file_name_ext))
            try :
                shutil.copy(
                    str_input_batch_csv_full_file_name_ext_from,
                    str_input_batch_csv_full_file_name_ext_to)
            except Exception as exception :
                print_exception_message(exception)

        # Copy Job Configuration JSON file to the job-specific
        # output directory.
        str_input_job_json_file_name_ext = \
            str_input_job_json_file_name + ".json"
        str_input_job_json_full_file_name_ext_from = os.path.join(
            ".", STR_INPUT_JOBS_JSON_PATH,
            str_input_job_json_file_name_ext)
        str_input_job_json_full_file_name_ext_to = os.path.join(
            str_job_subdir_name,
            str_input_job_json_file_name_ext)
        try :
            shutil.copy(
                str_input_job_json_full_file_name_ext_from,
                str_input_job_json_full_file_name_ext_to)
        except Exception as exception :
            print_exception_message(exception)

        # Copy Actual Vote Count Data CSV file to the job-specific
        # output directory.
        str_input_actual_data_csv_file_name_ext = \
            str_actual_vote_counts_csv_file_name + ".csv"
        str_input_actual_data_csv_full_file_name_ext_from = os.path.join(
            ".", STR_INPUT_DATA_CSV_PATH,
            str_input_actual_data_csv_file_name_ext)
        str_input_actual_data_csv_full_file_name_ext_to = os.path.join(
            str_job_subdir_name,
            str_input_actual_data_csv_file_name_ext)
        try :
            shutil.copy(
                str_input_actual_data_csv_full_file_name_ext_from,
                str_input_actual_data_csv_full_file_name_ext_to)
        except Exception as exception :
            print_exception_message(exception)

        if str_benchmark_vote_counts_csv_file_name is not None and \
           len(str_benchmark_vote_counts_csv_file_name) > 0 and \
           str_benchmark_vote_counts_csv_file_name != \
               str_actual_vote_counts_csv_file_name :
            # Copy Benchmark Vote Count Data CSV file to the job-specific
            # output directory.
            str_input_benchmark_data_csv_file_name_ext = \
                str_benchmark_vote_counts_csv_file_name + ".csv"
            str_input_benchmark_data_csv_full_file_name_ext_from = os.path.join(
                ".", STR_INPUT_DATA_CSV_PATH,
                str_input_benchmark_data_csv_file_name_ext)
            str_input_benchmark_data_csv_full_file_name_ext_to = os.path.join(
                str_job_subdir_name,
                str_input_benchmark_data_csv_file_name_ext)
            try :
                shutil.copy(
                    str_input_benchmark_data_csv_full_file_name_ext_from,
                    str_input_benchmark_data_csv_full_file_name_ext_to)
            except Exception as exception :
                print_exception_message(exception)
    else :
        str_job_subdir_name = None

    return str_job_subdir_name

###############################################################################

def run_election_outliers_job(
        #######################################################################
        # Common inputs (input and data processing-related)
        #######################################################################
        str_input_batch_bat_file_name_ext,
        str_input_batch_csv_file_name_ext,
        str_input_job_json_file_name,
        bool_apply_non_parametric_model,
        bool_apply_parametric_model,
        #
        str_actual_vote_counts_csv_file_name,
        lst_str_actual_all_vote_count_column_names = [],
        str_actual_residual_vote_count_column_name = None,
        lst_str_actual_residual_vote_count_column_names = [],
        #
        str_benchmark_vote_counts_csv_file_name = None,
        lst_str_benchmark_all_vote_count_column_names = [],
        str_benchmark_residual_vote_count_column_name = None,
        lst_str_benchmark_residual_vote_count_column_names = [],
        #
        bool_aggregate_vote_counts_by_county = False,
        bool_approximate_missing_benchmark_precincts_by_scaled_benchmark_county = False,
        bool_aggregate_missing_benchmark_precincts_into_new_residual_county = False,
        #
        int_random_number_generator_seed_for_sorting = 0,
        bool_retain_sorting_criteria_columns = False,
        bool_compute_cumulative_actual_vote_count_fractions = False,
        bool_save_sorted_precincts_to_csv_file = False,
        #
        bool_distribute_job_output_files_in_directories_by_type = True,
        bool_gather_all_job_files_in_one_directory = False,
        #######################################################################

        #######################################################################
        # Non-parametric-specific inputs
        #######################################################################
        int_max_num_two_ways_passes = 1, # integer in [1; +Inf]
        flt_window_size_scaling_factor = .5, # 0 < f < 1
        lst_str_all_predicting_predicted_vote_count_column_names = [],
        #######################################################################

        #######################################################################
        # Parametric-specific inputs
        #######################################################################
        bool_use_decimal_type = False,
        int_decimal_computational_precision = 1024, # [0; +Inf). # 1024 by default
        int_decimal_reporting_precision = 16, # [0; +Inf). # 16 by default
        int_max_num_iters_for_exact_hypergeom = 1_000_000, # [1; +Inf). # 1_000_000 by default
        int_max_num_iters_for_lanczos_approx_hypergeom = 500_000, # [1; +Inf). # 500_000 by default
        int_max_num_iters_for_spouge_approx_hypergeom = 500_000, # [1; +Inf). # 500_000 by default
        int_min_sample_size_for_approx_normal = 1_000, # [0; +Inf). # 1_000 by default
        flt_max_sample_sz_frac_of_pop_sz_for_approx_normal = 0.1, # [0.; 1.]. # 0.1 by default
        flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal = 0.1, # [0.; 1.]. # 0.1 by default
        flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal = 0.0, # [0.; +Inf). # 0.0 by default.
            # z = 2.575829303549 is for 99% in range mu +/- z*sigma
        #
        flt_lasso_alpha_l1_regularization_strength_term = None, # [0.; 1.]. # None by default (1.)
        int_lasso_maximum_number_of_iterations = 1_000_000,
        int_lasso_optimization_tolerance = 0.00001,
        flt_lasso_cv_length_of_alphas_regularization_path = 1., # (0.; +1.] # 1 by default (0.001); alpha_min / alpha_max
        int_lasso_cv_num_candidate_alphas_on_regularization_path = 0, # [0; +Inf] # 0 by default (100)
        int_lasso_cv_maximum_number_of_iterations = 1_000_000,
        int_lasso_cv_optimization_tolerance = 0.00001,
        int_lasso_cv_number_of_folds_in_cross_validation = 10,
        #
        bool_estimate_precinct_model_on_aggregated_vote_counts_by_county = False,
        bool_approximate_predictions_for_missing_benchmark_precincts_by_actual_values = False,
        bool_true_each_choice_with_all_other_choices__false_each_choice_with_each_choice = True,
        bool_compute_outlier_score_level_1 = True,
        bool_compute_outlier_score_level_2 = True,
        bool_compute_outlier_score_level_3 = True,
        #
        bool_estimate_model_parameters_diagnostics = False,
        bool_save_estimated_models_parameters_to_csv_file = False,
        bool_retain_predicted_vote_counts_columns_in_sorted_precincts_file = True,
        bool_save_outlier_score_stats_to_csv_file = False,
        bool_save_model_diagnostics_to_csv_file = False,
        #
        lst_str_predicted_for_actual_vote_count_column_names = [],
        #
        lst_str_predicting_from_ln_of_incremented_actual_vote_count_column_names = [],
        lst_str_predicting_from_power_one_quarter_of_actual_vote_count_column_names = [],
        lst_str_predicting_from_square_root_of_actual_vote_count_column_names = [],
        lst_str_predicting_from_power_three_quarters_of_actual_vote_count_column_names = [],
        lst_str_predicting_from_actual_vote_count_column_names = [],
        lst_str_predicting_from_power_one_and_a_half_of_actual_vote_count_column_names = [],
        lst_str_predicting_from_squared_actual_vote_count_column_names = [],
        #
        lst_str_predicting_from_ln_of_incremented_benchmark_vote_count_column_names = [],
        lst_str_predicting_from_power_one_quarter_of_benchmark_vote_count_column_names = [],
        lst_str_predicting_from_square_root_of_benchmark_vote_count_column_names = [],
        lst_str_predicting_from_power_three_quarters_of_benchmark_vote_count_column_names = [],
        lst_str_predicting_from_benchmark_vote_count_column_names = [],
        lst_str_predicting_from_power_one_and_a_half_of_benchmark_vote_count_column_names = [],
        lst_str_predicting_from_squared_benchmark_vote_count_column_names = [],
        #
        # [(a,b) for a in lst_str_predicting_from_actual_vote_count_column_names
        #        for b in lst_str_predicting_from_actual_vote_count_column_names]
        lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_actual_vote_count_column_names = [],
        lst_pairs_str_predicting_from_square_root_of_actual_interactions_actual_vote_count_column_names = [],
        lst_pairs_str_predicting_from_actual_interactions_actual_vote_count_column_names = [],
        #
        # [(a,b) for a in lst_str_predicting_from_benchmark_vote_count_column_names
        #        for b in lst_str_predicting_from_benchmark_vote_count_column_names]
        lst_pairs_str_predicting_from_fourth_root_of_benchmark_interactions_benchmark_vote_count_column_names = [],
        lst_pairs_str_predicting_from_square_root_of_benchmark_interactions_benchmark_vote_count_column_names = [],
        lst_pairs_str_predicting_from_benchmark_interactions_benchmark_vote_count_column_names = [],
        #
        # [(a,b) for a in lst_str_predicting_from_actual_vote_count_column_names
        #        for b in lst_str_predicting_from_benchmark_vote_count_column_names]
        lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_benchmark_vote_count_column_names = [],
        lst_pairs_str_predicting_from_square_root_of_actual_interactions_benchmark_vote_count_column_names = [],
        lst_pairs_str_predicting_from_actual_interactions_benchmark_vote_count_column_names = [],
        #######################################################################
        bool_drop_predicted_variable_during_computation_of_predicting_tally = True,
        #######################################################################
        bool_predicting_from_ln_of_incremented_actual_tally = False,
        bool_predicting_from_power_one_quarter_of_actual_tally = False,
        bool_predicting_from_square_root_of_actual_tally = False,
        bool_predicting_from_power_three_quarters_of_actual_tally = False,
        bool_predicting_from_actual_tally = False,
        bool_predicting_from_power_one_and_a_half_of_actual_tally = False,
        bool_predicting_from_squared_actual_tally = False,
        #
        bool_predicting_from_ln_of_incremented_benchmark_tally = False,
        bool_predicting_from_power_one_quarter_of_benchmark_tally = False,
        bool_predicting_from_square_root_of_benchmark_tally = False,
        bool_predicting_from_power_three_quarters_of_benchmark_tally = False,
        bool_predicting_from_benchmark_tally = False,
        bool_predicting_from_power_one_and_a_half_of_benchmark_tally = False,
        bool_predicting_from_squared_benchmark_tally = False,
        #
        bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally = False,
        bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally = False,
        bool_predicting_from_actual_tally_interaction_benchmark_tally = False,
        #######################################################################
        bool_predicting_from_ln_of_incremented_actual_tally_interaction_county_indicator = False,
        bool_predicting_from_power_one_quarter_of_actual_tally_interaction_county_indicator = False,
        bool_predicting_from_square_root_of_actual_tally_interaction_county_indicator = False,
        bool_predicting_from_power_three_quarters_of_actual_tally_interaction_county_indicator = False,
        bool_predicting_from_actual_tally_interaction_county_indicator = False,
        bool_predicting_from_power_one_and_a_half_of_actual_tally_interaction_county_indicator = False,
        bool_predicting_from_squared_actual_tally_interaction_county_indicator = False,
        #
        bool_predicting_from_ln_of_incremented_benchmark_tally_interaction_county_indicator = False,
        bool_predicting_from_power_one_quarter_of_benchmark_tally_interaction_county_indicator = False,
        bool_predicting_from_square_root_of_benchmark_tally_interaction_county_indicator = False,
        bool_predicting_from_power_three_quarters_of_benchmark_tally_interaction_county_indicator = False,
        bool_predicting_from_benchmark_tally_interaction_county_indicator = False,
        bool_predicting_from_power_one_and_a_half_of_benchmark_tally_interaction_county_indicator = False,
        bool_predicting_from_squared_benchmark_tally_interaction_county_indicator = False,
        #
        bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator = False,
        bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator = False,
        bool_predicting_from_actual_tally_interaction_benchmark_tally_interaction_county_indicator = False,
        #######################################################################

        #######################################################################
        # Common inputs (output graphics-related)
        #######################################################################
        str_plot_title = None,
        str_x_axis_label =
            "Cumulative Headcount Percent for Localities Sorted in Descending"+
            " Order of Predictability.",
        str_y_axis_label =
            "Contest Choice Percent. Right Tails Show the Impact of Outliers.",
        #######################################################################
        lst_str_vote_count_column_names_for_cumulative_tally_for_curves = None,
        lst_str_vote_count_column_name_prefixes_to_use_for_curves = None,
        lst_str_legend_labels_for_curves = None,
        lst_str_color_names_for_curves = None,
        lst_str_linestyles_for_curves = ['-'],
        lst_flt_linewidths_for_curves = [1.],
        lst_bool_draw_right_tail_circle_for_curves = [False],
        lst_str_annotations_for_curves = [''],
        lst_int_annotation_font_sizes_for_curves = [6],
        lst_bool_display_final_cumulative_percent_of_tally_for_curves = [False],
        lst_int_final_cumulative_percent_of_tally_font_sizes_for_curves = [6],
        lst_int_final_cumulative_percent_of_tally_rounding_precision_for_curves = [2],
        #######################################################################
        int_plot_title_font_size = None,
        int_x_axis_label_font_size = None,
        int_y_axis_label_font_size = None,
        int_x_axis_tick_labels_font_size = None,
        int_y_axis_tick_labels_font_size = None,
        int_legend_font_size = None,
        str_legend_loc = 'lower left',
        lst_str_text_line_content = [],
        lst_str_text_line_color = [],
        int_text_line_font_size = 7,
        flt_text_line_x_loc_as_fraction = 0.1275,
        flt_text_line_lower_y_loc_as_fraction = 0.14,
        flt_text_line_upper_y_loc_as_fraction = 0.85,
        flt_text_line_abs_delta_y_loc_as_fraction = 0.03,
        str_text_line_start_y_loc = "upper",
        bool_display_chart_id = False,
        int_chart_id_font_size = 7,
        bool_include_mac_address_in_chart_id = False,
        #######################################################################
        lst_flt_min_x = [0.],
        lst_flt_max_x = [100.],
        lst_flt_xticks_incr = [5.],
        lst_flt_min_y = [0.],
        lst_flt_max_y = [100.],
        lst_flt_yticks_incr = [5.],
        lst_flt_plot_width_in_inches = [6.4],
        lst_flt_plot_height_in_inches = [4.8],
        lst_int_dots_per_inch_for_png_and_jpg_plots = [1200],
        lst_flt_right_tails_circles_radius_in_pct_pnts = [0.],
        lst_str_file_suffix_for_saving_plot = [None,],
        #######################################################################
        bool_show_plot = False,
        bool_save_plot_to_png_file = False,
        bool_save_plot_to_jpg_file = False,
        bool_save_plot_to_pdf_file = False,
        bool_save_plot_to_svg_file = False,
        #######################################################################
        ) :
    '''<function description>
    
    <calling functions>
    <called functions>

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
    if bool_gather_all_job_files_in_one_directory :
        str_job_subdir_name = copy_input_files_to_job_directory(
            str_input_batch_bat_file_name_ext =
                str_input_batch_bat_file_name_ext,
            str_input_batch_csv_file_name_ext =
                str_input_batch_csv_file_name_ext,
            str_input_job_json_file_name =
                str_input_job_json_file_name,
            str_actual_vote_counts_csv_file_name =
                str_actual_vote_counts_csv_file_name,
            str_benchmark_vote_counts_csv_file_name =
                str_benchmark_vote_counts_csv_file_name,
            bool_gather_all_job_files_in_one_directory =
                bool_gather_all_job_files_in_one_directory,)
    else :
        str_job_subdir_name = None

    if bool_apply_non_parametric_model :
        if len(lst_str_file_suffix_for_saving_plot) > 0 :
            lst_str_file_name_for_saving_plot = \
                [(str_input_job_json_file_name + "_" + str_file_suffix +
                  STR_OUTPUT_PLOTS_NPAR_FILE_SUFFIX)
                 for str_file_suffix in lst_str_file_suffix_for_saving_plot]
        else :
            lst_str_file_name_for_saving_plot = [
                str_input_job_json_file_name + "_" +
                STR_OUTPUT_PLOTS_NPAR_FILE_SUFFIX]
        apply_non_parametric_model(
            ###################################################################
            # generate_merged_counts
            ###################################################################
            str_actual_vote_counts_csv_file_name =
                str_actual_vote_counts_csv_file_name,
            lst_str_actual_all_vote_count_column_names =
                lst_str_actual_all_vote_count_column_names,
            str_actual_residual_vote_count_column_name =
                str_actual_residual_vote_count_column_name,
            lst_str_actual_residual_vote_count_column_names =
                lst_str_actual_residual_vote_count_column_names,
            #
            str_benchmark_vote_counts_csv_file_name =
                str_benchmark_vote_counts_csv_file_name,
            lst_str_benchmark_all_vote_count_column_names =
                lst_str_benchmark_all_vote_count_column_names,
            str_benchmark_residual_vote_count_column_name =
                str_benchmark_residual_vote_count_column_name,
            lst_str_benchmark_residual_vote_count_column_names =
                lst_str_benchmark_residual_vote_count_column_names,
            #
            bool_aggregate_vote_counts_by_county =
                bool_aggregate_vote_counts_by_county,
            bool_approximate_missing_benchmark_precincts_by_scaled_benchmark_county =
                bool_approximate_missing_benchmark_precincts_by_scaled_benchmark_county,
            bool_aggregate_missing_benchmark_precincts_into_new_residual_county =
                bool_aggregate_missing_benchmark_precincts_into_new_residual_county,
            #
            int_random_number_generator_seed_for_sorting =
                int_random_number_generator_seed_for_sorting,
            bool_retain_sorting_criteria_columns =
                bool_retain_sorting_criteria_columns,
            bool_compute_cumulative_actual_vote_count_fractions =
                bool_compute_cumulative_actual_vote_count_fractions,
            bool_save_sorted_precincts_to_csv_file =
                bool_save_sorted_precincts_to_csv_file,
            str_csv_file_name_for_saving_sorted_precincts =
                (str_input_job_json_file_name +
                 STR_OUTPUT_DATA_NPAR_FILE_SUFFIX),
            #
            bool_distribute_job_output_files_in_directories_by_type =
                bool_distribute_job_output_files_in_directories_by_type,
            bool_gather_all_job_files_in_one_directory =
                bool_gather_all_job_files_in_one_directory,
            str_job_subdir_name = str_job_subdir_name,

            ###################################################################
            # sort_precincts_by_decreasing_predictability_for_non_parametric_model
            ###################################################################
            int_max_num_two_ways_passes = int_max_num_two_ways_passes,
            flt_window_size_scaling_factor = flt_window_size_scaling_factor,
            lst_str_all_predicting_predicted_vote_count_column_names =
                lst_str_all_predicting_predicted_vote_count_column_names,

            ###################################################################
            # generate_cumulative_plot
            ###################################################################
            str_plot_title = str_plot_title,
            str_x_axis_label = str_x_axis_label,
            str_y_axis_label = str_y_axis_label,
            ###################################################################
            lst_str_vote_count_column_names_for_cumulative_tally_for_curves =
                lst_str_vote_count_column_names_for_cumulative_tally_for_curves,
            lst_str_vote_count_column_name_prefixes_to_use_for_curves =
                lst_str_vote_count_column_name_prefixes_to_use_for_curves,
            lst_str_legend_labels_for_curves = lst_str_legend_labels_for_curves,
            lst_str_color_names_for_curves = lst_str_color_names_for_curves,
            lst_str_linestyles_for_curves = lst_str_linestyles_for_curves,
            lst_flt_linewidths_for_curves = lst_flt_linewidths_for_curves,
            lst_bool_draw_right_tail_circle_for_curves =
                lst_bool_draw_right_tail_circle_for_curves,
            lst_str_annotations_for_curves = lst_str_annotations_for_curves,
            lst_int_annotation_font_sizes_for_curves =
                lst_int_annotation_font_sizes_for_curves,
            lst_bool_display_final_cumulative_percent_of_tally_for_curves =
                lst_bool_display_final_cumulative_percent_of_tally_for_curves,
            lst_int_final_cumulative_percent_of_tally_font_sizes_for_curves =
                lst_int_final_cumulative_percent_of_tally_font_sizes_for_curves,
            lst_int_final_cumulative_percent_of_tally_rounding_precision_for_curves =
                lst_int_final_cumulative_percent_of_tally_rounding_precision_for_curves,
            ###################################################################
            int_plot_title_font_size = int_plot_title_font_size,
            int_x_axis_label_font_size = int_x_axis_label_font_size,
            int_y_axis_label_font_size = int_y_axis_label_font_size,
            int_x_axis_tick_labels_font_size = int_x_axis_tick_labels_font_size,
            int_y_axis_tick_labels_font_size = int_y_axis_tick_labels_font_size,
            int_legend_font_size = int_legend_font_size,
            str_legend_loc = str_legend_loc,
            lst_str_text_line_content = lst_str_text_line_content,
            lst_str_text_line_color = lst_str_text_line_color,
            int_text_line_font_size = int_text_line_font_size,
            flt_text_line_x_loc_as_fraction = flt_text_line_x_loc_as_fraction,
            flt_text_line_lower_y_loc_as_fraction = flt_text_line_lower_y_loc_as_fraction,
            flt_text_line_upper_y_loc_as_fraction = flt_text_line_upper_y_loc_as_fraction,
            flt_text_line_abs_delta_y_loc_as_fraction = flt_text_line_abs_delta_y_loc_as_fraction,
            str_text_line_start_y_loc = str_text_line_start_y_loc,
            bool_display_chart_id = bool_display_chart_id,
            int_chart_id_font_size = int_chart_id_font_size,
            bool_include_mac_address_in_chart_id = bool_include_mac_address_in_chart_id,
            ###################################################################
            lst_flt_min_x = lst_flt_min_x,
            lst_flt_max_x = lst_flt_max_x,
            lst_flt_xticks_incr = lst_flt_xticks_incr,
            lst_flt_min_y = lst_flt_min_y,
            lst_flt_max_y = lst_flt_max_y,
            lst_flt_yticks_incr = lst_flt_yticks_incr,
            lst_flt_plot_width_in_inches = lst_flt_plot_width_in_inches,
            lst_flt_plot_height_in_inches = lst_flt_plot_height_in_inches,
            lst_int_dots_per_inch_for_png_and_jpg_plots =
                lst_int_dots_per_inch_for_png_and_jpg_plots,
            lst_flt_right_tails_circles_radius_in_pct_pnts =
                lst_flt_right_tails_circles_radius_in_pct_pnts,
            lst_str_file_name_for_saving_plot =
                lst_str_file_name_for_saving_plot,
            ######################################################
            bool_show_plot = bool_show_plot,
            bool_save_plot_to_png_file = bool_save_plot_to_png_file,
            bool_save_plot_to_jpg_file = bool_save_plot_to_jpg_file,
            bool_save_plot_to_pdf_file = bool_save_plot_to_pdf_file,
            bool_save_plot_to_svg_file = bool_save_plot_to_svg_file,
            )

    if bool_apply_parametric_model :
        if len(lst_str_file_suffix_for_saving_plot) > 0 :
            lst_str_file_name_for_saving_plot = \
                [(str_input_job_json_file_name + "_" + str_file_suffix +
                  STR_OUTPUT_PLOTS_PARM_FILE_SUFFIX)
                 for str_file_suffix in lst_str_file_suffix_for_saving_plot]
        else :
            lst_str_file_name_for_saving_plot = [
                str_input_job_json_file_name + "_" +
                STR_OUTPUT_PLOTS_PARM_FILE_SUFFIX]
        apply_parametric_model(
            ###################################################################
            # generate_predicted_counts
            ###################################################################
            str_actual_vote_counts_csv_file_name =
                str_actual_vote_counts_csv_file_name,
            lst_str_actual_all_vote_count_column_names =
                lst_str_actual_all_vote_count_column_names,
            str_actual_residual_vote_count_column_name =
                str_actual_residual_vote_count_column_name,
            lst_str_actual_residual_vote_count_column_names =
                lst_str_actual_residual_vote_count_column_names,
            #
            str_benchmark_vote_counts_csv_file_name =
                str_benchmark_vote_counts_csv_file_name,
            lst_str_benchmark_all_vote_count_column_names =
                lst_str_benchmark_all_vote_count_column_names,
            str_benchmark_residual_vote_count_column_name =
                str_benchmark_residual_vote_count_column_name,
            lst_str_benchmark_residual_vote_count_column_names =
                lst_str_benchmark_residual_vote_count_column_names,
            #
            bool_aggregate_vote_counts_by_county =
                bool_aggregate_vote_counts_by_county,
            bool_approximate_missing_benchmark_precincts_by_scaled_benchmark_county =
                bool_approximate_missing_benchmark_precincts_by_scaled_benchmark_county,
            bool_aggregate_missing_benchmark_precincts_into_new_residual_county =
                bool_aggregate_missing_benchmark_precincts_into_new_residual_county,

            ###################################################################
            # sort_precincts_by_decreasing_predictability_for_parametric_model
            ###################################################################
            int_random_number_generator_seed_for_sorting =
                int_random_number_generator_seed_for_sorting,
            bool_retain_sorting_criteria_columns =
                bool_retain_sorting_criteria_columns,
            bool_compute_cumulative_actual_vote_count_fractions =
                bool_compute_cumulative_actual_vote_count_fractions,
            bool_save_sorted_precincts_to_csv_file =
                bool_save_sorted_precincts_to_csv_file,
            str_csv_file_name_for_saving_sorted_precincts =
                (str_input_job_json_file_name +
                 STR_OUTPUT_DATA_PARM_FILE_SUFFIX),
            ###################################################################
            bool_distribute_job_output_files_in_directories_by_type =
                bool_distribute_job_output_files_in_directories_by_type,
            bool_gather_all_job_files_in_one_directory =
                bool_gather_all_job_files_in_one_directory,
            str_job_subdir_name = str_job_subdir_name,

            ###################################################################
            bool_use_decimal_type = bool_use_decimal_type,
            int_decimal_computational_precision =
                int_decimal_computational_precision,
            int_decimal_reporting_precision = int_decimal_reporting_precision,
            int_max_num_iters_for_exact_hypergeom =
                int_max_num_iters_for_exact_hypergeom,
            int_max_num_iters_for_lanczos_approx_hypergeom =
                int_max_num_iters_for_lanczos_approx_hypergeom,
            int_max_num_iters_for_spouge_approx_hypergeom =
                int_max_num_iters_for_spouge_approx_hypergeom,
            int_min_sample_size_for_approx_normal =
                int_min_sample_size_for_approx_normal,
            flt_max_sample_sz_frac_of_pop_sz_for_approx_normal =
                flt_max_sample_sz_frac_of_pop_sz_for_approx_normal,
            flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal =
                flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal,
            flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal =
                flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal,
            #
            flt_lasso_alpha_l1_regularization_strength_term =
                flt_lasso_alpha_l1_regularization_strength_term,
            int_lasso_maximum_number_of_iterations =
                int_lasso_maximum_number_of_iterations,
            int_lasso_optimization_tolerance =
                int_lasso_optimization_tolerance,
            flt_lasso_cv_length_of_alphas_regularization_path =
                flt_lasso_cv_length_of_alphas_regularization_path,
            int_lasso_cv_num_candidate_alphas_on_regularization_path =
                int_lasso_cv_num_candidate_alphas_on_regularization_path,
            int_lasso_cv_maximum_number_of_iterations =
                int_lasso_cv_maximum_number_of_iterations,
            int_lasso_cv_optimization_tolerance =
                int_lasso_cv_optimization_tolerance,
            int_lasso_cv_number_of_folds_in_cross_validation =
                int_lasso_cv_number_of_folds_in_cross_validation,
            #
            bool_estimate_precinct_model_on_aggregated_vote_counts_by_county =
                bool_estimate_precinct_model_on_aggregated_vote_counts_by_county,
            bool_approximate_predictions_for_missing_benchmark_precincts_by_actual_values =
                bool_approximate_predictions_for_missing_benchmark_precincts_by_actual_values,
            bool_true_each_choice_with_all_other_choices__false_each_choice_with_each_choice =
                bool_true_each_choice_with_all_other_choices__false_each_choice_with_each_choice,
            bool_compute_outlier_score_level_1 =
                bool_compute_outlier_score_level_1,
            bool_compute_outlier_score_level_2 =
                bool_compute_outlier_score_level_2,
            bool_compute_outlier_score_level_3 =
                bool_compute_outlier_score_level_3,
            #
            bool_estimate_model_parameters_diagnostics =
                bool_estimate_model_parameters_diagnostics,
            bool_save_estimated_models_parameters_to_csv_file =
                bool_save_estimated_models_parameters_to_csv_file,
            str_csv_file_name_for_saving_estimated_models_parameters =
                (str_input_job_json_file_name + STR_OUTPUT_PARAMS_PARM_FILE_SUFFIX),
            bool_retain_predicted_vote_counts_columns_in_sorted_precincts_file =
                bool_retain_predicted_vote_counts_columns_in_sorted_precincts_file,
            bool_save_outlier_score_stats_to_csv_file =
                bool_save_outlier_score_stats_to_csv_file,
            str_csv_file_name_for_saving_outlier_score_stats =
                (str_input_job_json_file_name + STR_OUTPUT_SCORES_PARM_FILE_SUFFIX),
            bool_save_model_diagnostics_to_csv_file =
                bool_save_model_diagnostics_to_csv_file,
            str_csv_file_name_for_saving_model_diagnostics =
                (str_input_job_json_file_name + STR_OUTPUT_DIAGN_PARM_FILE_SUFFIX),
            #
            lst_str_predicted_for_actual_vote_count_column_names =
                lst_str_predicted_for_actual_vote_count_column_names,
            #
            lst_str_predicting_from_ln_of_incremented_actual_vote_count_column_names =
                lst_str_predicting_from_ln_of_incremented_actual_vote_count_column_names,
            lst_str_predicting_from_power_one_quarter_of_actual_vote_count_column_names =
                lst_str_predicting_from_power_one_quarter_of_actual_vote_count_column_names,
            lst_str_predicting_from_square_root_of_actual_vote_count_column_names = \
                lst_str_predicting_from_square_root_of_actual_vote_count_column_names,
            lst_str_predicting_from_power_three_quarters_of_actual_vote_count_column_names = \
                lst_str_predicting_from_power_three_quarters_of_actual_vote_count_column_names,
            lst_str_predicting_from_actual_vote_count_column_names = \
                lst_str_predicting_from_actual_vote_count_column_names,
            lst_str_predicting_from_power_one_and_a_half_of_actual_vote_count_column_names = \
                lst_str_predicting_from_power_one_and_a_half_of_actual_vote_count_column_names,
            lst_str_predicting_from_squared_actual_vote_count_column_names = \
                lst_str_predicting_from_squared_actual_vote_count_column_names,
            #
            lst_str_predicting_from_ln_of_incremented_benchmark_vote_count_column_names = 
                lst_str_predicting_from_ln_of_incremented_benchmark_vote_count_column_names,
            lst_str_predicting_from_power_one_quarter_of_benchmark_vote_count_column_names = \
                lst_str_predicting_from_power_one_quarter_of_benchmark_vote_count_column_names,
            lst_str_predicting_from_square_root_of_benchmark_vote_count_column_names = \
                lst_str_predicting_from_square_root_of_benchmark_vote_count_column_names,
            lst_str_predicting_from_power_three_quarters_of_benchmark_vote_count_column_names = \
                lst_str_predicting_from_power_three_quarters_of_benchmark_vote_count_column_names,
            lst_str_predicting_from_benchmark_vote_count_column_names = \
                lst_str_predicting_from_benchmark_vote_count_column_names,
            lst_str_predicting_from_power_one_and_a_half_of_benchmark_vote_count_column_names = \
                lst_str_predicting_from_power_one_and_a_half_of_benchmark_vote_count_column_names,
            lst_str_predicting_from_squared_benchmark_vote_count_column_names = \
                lst_str_predicting_from_squared_benchmark_vote_count_column_names,
            #
            lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_actual_vote_count_column_names = \
                lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_actual_vote_count_column_names,
            lst_pairs_str_predicting_from_square_root_of_actual_interactions_actual_vote_count_column_names = \
                lst_pairs_str_predicting_from_square_root_of_actual_interactions_actual_vote_count_column_names,
            lst_pairs_str_predicting_from_actual_interactions_actual_vote_count_column_names = \
                lst_pairs_str_predicting_from_actual_interactions_actual_vote_count_column_names,
            #
            lst_pairs_str_predicting_from_fourth_root_of_benchmark_interactions_benchmark_vote_count_column_names = \
                lst_pairs_str_predicting_from_fourth_root_of_benchmark_interactions_benchmark_vote_count_column_names,
            lst_pairs_str_predicting_from_square_root_of_benchmark_interactions_benchmark_vote_count_column_names = \
                lst_pairs_str_predicting_from_square_root_of_benchmark_interactions_benchmark_vote_count_column_names,
            lst_pairs_str_predicting_from_benchmark_interactions_benchmark_vote_count_column_names = \
                lst_pairs_str_predicting_from_benchmark_interactions_benchmark_vote_count_column_names,
            #
            lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_benchmark_vote_count_column_names = \
                lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_benchmark_vote_count_column_names,
            lst_pairs_str_predicting_from_square_root_of_actual_interactions_benchmark_vote_count_column_names = \
                lst_pairs_str_predicting_from_square_root_of_actual_interactions_benchmark_vote_count_column_names,
            lst_pairs_str_predicting_from_actual_interactions_benchmark_vote_count_column_names = \
                lst_pairs_str_predicting_from_actual_interactions_benchmark_vote_count_column_names,
            ###################################################################
            bool_drop_predicted_variable_during_computation_of_predicting_tally = \
                bool_drop_predicted_variable_during_computation_of_predicting_tally,
            ###################################################################
            bool_predicting_from_ln_of_incremented_actual_tally = \
                bool_predicting_from_ln_of_incremented_actual_tally,
            bool_predicting_from_power_one_quarter_of_actual_tally = \
                bool_predicting_from_power_one_quarter_of_actual_tally,
            bool_predicting_from_square_root_of_actual_tally = \
                bool_predicting_from_square_root_of_actual_tally,
            bool_predicting_from_power_three_quarters_of_actual_tally = \
                bool_predicting_from_power_three_quarters_of_actual_tally,
            bool_predicting_from_actual_tally = \
                bool_predicting_from_actual_tally,
            bool_predicting_from_power_one_and_a_half_of_actual_tally = \
                bool_predicting_from_power_one_and_a_half_of_actual_tally,
            bool_predicting_from_squared_actual_tally = \
                bool_predicting_from_squared_actual_tally,
            #
            bool_predicting_from_ln_of_incremented_benchmark_tally = \
                bool_predicting_from_ln_of_incremented_benchmark_tally,
            bool_predicting_from_power_one_quarter_of_benchmark_tally = \
                bool_predicting_from_power_one_quarter_of_benchmark_tally,
            bool_predicting_from_square_root_of_benchmark_tally = \
                bool_predicting_from_square_root_of_benchmark_tally,
            bool_predicting_from_power_three_quarters_of_benchmark_tally = \
                bool_predicting_from_power_three_quarters_of_benchmark_tally,
            bool_predicting_from_benchmark_tally = \
                bool_predicting_from_benchmark_tally,
            bool_predicting_from_power_one_and_a_half_of_benchmark_tally = \
                bool_predicting_from_power_one_and_a_half_of_benchmark_tally,
            bool_predicting_from_squared_benchmark_tally = \
                bool_predicting_from_squared_benchmark_tally,
            #
            bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally = \
                bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally,
            bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally = \
                bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally,
            bool_predicting_from_actual_tally_interaction_benchmark_tally = \
                bool_predicting_from_actual_tally_interaction_benchmark_tally,
            ###################################################################
            bool_predicting_from_ln_of_incremented_actual_tally_interaction_county_indicator = \
                bool_predicting_from_ln_of_incremented_actual_tally_interaction_county_indicator,
            bool_predicting_from_power_one_quarter_of_actual_tally_interaction_county_indicator = \
                bool_predicting_from_power_one_quarter_of_actual_tally_interaction_county_indicator,
            bool_predicting_from_square_root_of_actual_tally_interaction_county_indicator = \
                bool_predicting_from_square_root_of_actual_tally_interaction_county_indicator,
            bool_predicting_from_power_three_quarters_of_actual_tally_interaction_county_indicator = \
                bool_predicting_from_power_three_quarters_of_actual_tally_interaction_county_indicator,
            bool_predicting_from_actual_tally_interaction_county_indicator = \
                bool_predicting_from_actual_tally_interaction_county_indicator,
            bool_predicting_from_power_one_and_a_half_of_actual_tally_interaction_county_indicator = \
                bool_predicting_from_power_one_and_a_half_of_actual_tally_interaction_county_indicator,
            bool_predicting_from_squared_actual_tally_interaction_county_indicator = \
                bool_predicting_from_squared_actual_tally_interaction_county_indicator,
            #
            bool_predicting_from_ln_of_incremented_benchmark_tally_interaction_county_indicator = \
                bool_predicting_from_ln_of_incremented_benchmark_tally_interaction_county_indicator,
            bool_predicting_from_power_one_quarter_of_benchmark_tally_interaction_county_indicator = \
                bool_predicting_from_power_one_quarter_of_benchmark_tally_interaction_county_indicator,
            bool_predicting_from_square_root_of_benchmark_tally_interaction_county_indicator = \
                bool_predicting_from_square_root_of_benchmark_tally_interaction_county_indicator,
            bool_predicting_from_power_three_quarters_of_benchmark_tally_interaction_county_indicator = \
                bool_predicting_from_power_three_quarters_of_benchmark_tally_interaction_county_indicator,
            bool_predicting_from_benchmark_tally_interaction_county_indicator = \
                bool_predicting_from_benchmark_tally_interaction_county_indicator,
            bool_predicting_from_power_one_and_a_half_of_benchmark_tally_interaction_county_indicator = \
                bool_predicting_from_power_one_and_a_half_of_benchmark_tally_interaction_county_indicator,
            bool_predicting_from_squared_benchmark_tally_interaction_county_indicator = \
                bool_predicting_from_squared_benchmark_tally_interaction_county_indicator,
            #
            bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator = \
                bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator,
            bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator = \
                bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator,
            bool_predicting_from_actual_tally_interaction_benchmark_tally_interaction_county_indicator = \
                bool_predicting_from_actual_tally_interaction_benchmark_tally_interaction_county_indicator,

            ###################################################################
            # generate_cumulative_plot
            ###################################################################
            str_plot_title = str_plot_title,
            str_x_axis_label = str_x_axis_label,
            str_y_axis_label = str_y_axis_label,
            ###################################################################
            lst_str_vote_count_column_names_for_cumulative_tally_for_curves = \
                lst_str_vote_count_column_names_for_cumulative_tally_for_curves,
            lst_str_vote_count_column_name_prefixes_to_use_for_curves = \
                lst_str_vote_count_column_name_prefixes_to_use_for_curves,
            lst_str_legend_labels_for_curves = \
                lst_str_legend_labels_for_curves,
            lst_str_color_names_for_curves = \
                lst_str_color_names_for_curves,
            lst_str_linestyles_for_curves = \
                lst_str_linestyles_for_curves,
            lst_flt_linewidths_for_curves = \
                lst_flt_linewidths_for_curves,
            lst_bool_draw_right_tail_circle_for_curves = \
                lst_bool_draw_right_tail_circle_for_curves,
            lst_str_annotations_for_curves = \
                lst_str_annotations_for_curves,
            lst_int_annotation_font_sizes_for_curves = \
                lst_int_annotation_font_sizes_for_curves,
            lst_bool_display_final_cumulative_percent_of_tally_for_curves = \
                lst_bool_display_final_cumulative_percent_of_tally_for_curves,
            lst_int_final_cumulative_percent_of_tally_font_sizes_for_curves = \
                lst_int_final_cumulative_percent_of_tally_font_sizes_for_curves,
            lst_int_final_cumulative_percent_of_tally_rounding_precision_for_curves = \
                lst_int_final_cumulative_percent_of_tally_rounding_precision_for_curves,
            ###################################################################
            int_plot_title_font_size = int_plot_title_font_size,
            int_x_axis_label_font_size = int_x_axis_label_font_size,
            int_y_axis_label_font_size = int_y_axis_label_font_size,
            int_x_axis_tick_labels_font_size = int_x_axis_tick_labels_font_size,
            int_y_axis_tick_labels_font_size = int_y_axis_tick_labels_font_size,
            int_legend_font_size = int_legend_font_size,
            str_legend_loc = str_legend_loc,
            lst_str_text_line_content = lst_str_text_line_content,
            lst_str_text_line_color = lst_str_text_line_color,
            int_text_line_font_size = int_text_line_font_size,
            flt_text_line_x_loc_as_fraction = flt_text_line_x_loc_as_fraction,
            flt_text_line_lower_y_loc_as_fraction = flt_text_line_lower_y_loc_as_fraction,
            flt_text_line_upper_y_loc_as_fraction = flt_text_line_upper_y_loc_as_fraction,
            flt_text_line_abs_delta_y_loc_as_fraction = flt_text_line_abs_delta_y_loc_as_fraction,
            str_text_line_start_y_loc = str_text_line_start_y_loc,
            bool_display_chart_id = bool_display_chart_id,
            int_chart_id_font_size = int_chart_id_font_size,
            bool_include_mac_address_in_chart_id = bool_include_mac_address_in_chart_id,
            ###################################################################
            lst_flt_min_x = lst_flt_min_x,
            lst_flt_max_x = lst_flt_max_x,
            lst_flt_xticks_incr = lst_flt_xticks_incr,
            lst_flt_min_y = lst_flt_min_y,
            lst_flt_max_y = lst_flt_max_y,
            lst_flt_yticks_incr = lst_flt_yticks_incr,
            lst_flt_plot_width_in_inches = lst_flt_plot_width_in_inches,
            lst_flt_plot_height_in_inches = lst_flt_plot_height_in_inches,
            lst_int_dots_per_inch_for_png_and_jpg_plots =
                lst_int_dots_per_inch_for_png_and_jpg_plots,
            lst_flt_right_tails_circles_radius_in_pct_pnts = \
                lst_flt_right_tails_circles_radius_in_pct_pnts,
            lst_str_file_name_for_saving_plot =
                lst_str_file_name_for_saving_plot,
            ###################################################################
            bool_show_plot = bool_show_plot,
            bool_save_plot_to_png_file = bool_save_plot_to_png_file,
            bool_save_plot_to_jpg_file = bool_save_plot_to_jpg_file,
            bool_save_plot_to_pdf_file = bool_save_plot_to_pdf_file,
            bool_save_plot_to_svg_file = bool_save_plot_to_svg_file,
            )

###############################################################################

def process_input_job_json_file(
        str_input_job_json_file_name,
        str_input_batch_bat_file_name_ext = None,
        str_input_batch_csv_file_name_ext = None,
        bool_skip_non_parametric_model = False,
        bool_skip_parametric_model = False,
        bool_skip_showing_plot = False,
    ) :
    '''<function description>
    
    <calling functions>
    <called functions>

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

    dict_input_config = None
    str_input_job_json_file_full_path = os.path.join(
        STR_INPUT_JOBS_JSON_PATH, str_input_job_json_file_name + ".json")
    try :
        with open(str_input_job_json_file_full_path) as f:
            dict_input_config = json.load(f)
    except Exception as exception :
        print_exception_message(exception)

    # TBD: validate "dict_input_job_from_json" and all its elements.

    ###########################################################################
    try :
        dict_core = dict_input_config["core"]
        dict_non_parametric = dict_input_config.get("non_parametric", None)
        dict_parametric = dict_input_config.get("parametric", None)
        dict_graphics = dict_input_config["graphics"]
        #######################################################################

        #######################################################################
        # Core inputs
        #######################################################################
        if bool_skip_non_parametric_model :
            bool_apply_non_parametric_model = False
        else :
            bool_apply_non_parametric_model = dict_core[
                "bool_apply_non_parametric_model"]
        if bool_skip_parametric_model :
            bool_apply_parametric_model = False
        else :
            bool_apply_parametric_model = dict_core[
                "bool_apply_parametric_model"]
        #
        str_actual_vote_counts_csv_file_name = dict_core[
            "str_actual_vote_counts_csv_file_name"]
        lst_str_actual_all_vote_count_column_names = dict_core[
            "lst_str_actual_all_vote_count_column_names"]
        str_actual_residual_vote_count_column_name = dict_core[
            "str_actual_residual_vote_count_column_name"]
        lst_str_actual_residual_vote_count_column_names = dict_core[
            "lst_str_actual_residual_vote_count_column_names"]
        #if str_actual_residual_vote_count_column_name is not None and \
        #   len(lst_str_actual_residual_vote_count_column_names) > 0 and \
        #   str_actual_residual_vote_count_column_name not in \
        #   lst_str_actual_all_vote_count_column_names :
        #    lst_str_actual_all_vote_count_column_names.append(
        #        str_actual_residual_vote_count_column_name)
        #
        str_benchmark_vote_counts_csv_file_name = dict_core[
            "str_benchmark_vote_counts_csv_file_name"]
        lst_str_benchmark_all_vote_count_column_names = dict_core[
            "lst_str_benchmark_all_vote_count_column_names"]
        str_benchmark_residual_vote_count_column_name = dict_core[
            "str_benchmark_residual_vote_count_column_name"]
        lst_str_benchmark_residual_vote_count_column_names = dict_core[
            "lst_str_benchmark_residual_vote_count_column_names"]
        #if str_benchmark_residual_vote_count_column_name is not None and \
        #   len(lst_str_benchmark_residual_vote_count_column_names) > 0 and \
        #   str_benchmark_residual_vote_count_column_name not in \
        #   lst_str_benchmark_all_vote_count_column_names :
        #    lst_str_benchmark_all_vote_count_column_names.append(
        #        str_benchmark_residual_vote_count_column_name)
        #
        bool_aggregate_vote_counts_by_county = dict_core[
            "bool_aggregate_vote_counts_by_county"]
        bool_approximate_missing_benchmark_precincts_by_scaled_benchmark_county = dict_core[
            "bool_approximate_missing_benchmark_precincts_by_scaled_benchmark_county"]
        bool_aggregate_missing_benchmark_precincts_into_new_residual_county = dict_core[
            "bool_aggregate_missing_benchmark_precincts_into_new_residual_county"]
        #
        int_random_number_generator_seed_for_sorting = dict_core[
            "int_random_number_generator_seed_for_sorting"]
        bool_retain_sorting_criteria_columns = dict_core[
            "bool_retain_sorting_criteria_columns"]
        bool_compute_cumulative_actual_vote_count_fractions = dict_core[
            "bool_compute_cumulative_actual_vote_count_fractions"]
        bool_save_sorted_precincts_to_csv_file = dict_core[
            "bool_save_sorted_precincts_to_csv_file"]
        #
        bool_distribute_job_output_files_in_directories_by_type = dict_core[
            "bool_distribute_job_output_files_in_directories_by_type"]
        bool_gather_all_job_files_in_one_directory = dict_core[
            "bool_gather_all_job_files_in_one_directory"]
        #######################################################################

        #######################################################################
        # Non-parametric-specific inputs
        #######################################################################
        int_max_num_two_ways_passes = (1 if dict_non_parametric is None else
            dict_non_parametric.get("int_max_num_two_ways_passes", 1)) # integer in [1; +Inf]
        flt_window_size_scaling_factor = (0.5 if dict_non_parametric is None else
            dict_non_parametric.get("flt_window_size_scaling_factor", 0.5)) # 0 < f < 1
        lst_str_all_predicting_predicted_vote_count_column_names = (
            [] if dict_non_parametric is None else dict_non_parametric.get(
                "lst_str_all_predicting_predicted_vote_count_column_names", []))
        #######################################################################

        #######################################################################
        # Parametric-specific inputs
        #######################################################################
        bool_use_decimal_type = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_use_decimal_type", False))
        int_decimal_computational_precision = (
            1024 if dict_parametric is None else dict_parametric.get(
                "int_decimal_computational_precision", 1024))
                # [0; +Inf). # 1024 by default
        int_decimal_reporting_precision = (
            16 if dict_parametric is None else dict_parametric.get(
                "int_decimal_reporting_precision", 16))
                # [0; +Inf). # 16 by default
        int_max_num_iters_for_exact_hypergeom = (
            1_000_000 if dict_parametric is None else dict_parametric.get(
                "int_max_num_iters_for_exact_hypergeom", 1_000_000))
                # [1; +Inf). # 1_000_000 by default
        int_max_num_iters_for_lanczos_approx_hypergeom = (
            500_000 if dict_parametric is None else dict_parametric.get(
                "int_max_num_iters_for_lanczos_approx_hypergeom", 500_000))
                # [1; +Inf). # 500_000 by default
        int_max_num_iters_for_spouge_approx_hypergeom = (
            500_000 if dict_parametric is None else dict_parametric.get(
                "int_max_num_iters_for_spouge_approx_hypergeom", 500_000))
                # [1; +Inf). # 500_000 by default
        int_min_sample_size_for_approx_normal = (
            1_000 if dict_parametric is None else dict_parametric.get(
                "int_min_sample_size_for_approx_normal", 1_000))
                # [0; +Inf). # 1_000 by default
        flt_max_sample_sz_frac_of_pop_sz_for_approx_normal = (
            0.1 if dict_parametric is None else dict_parametric.get(
                "flt_max_sample_sz_frac_of_pop_sz_for_approx_normal", 0.1))
                # [0.; 1.]. # 0.1 by default
        flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal = (
            0.1 if dict_parametric is None else dict_parametric.get(
                "flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal", 0.1))
                # [0.; 1.]. # 0.1 by default
        flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal = (
            0.0 if dict_parametric is None else dict_parametric.get(
                "flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal", 0.0))
                # [0.; +Inf). # 0.0 by default.
                # z = 2.575829303549 is for 99% in range mu +/- z*sigma
        #
        flt_lasso_alpha_l1_regularization_strength_term = (
            None if dict_parametric is None else dict_parametric.get(
                "flt_lasso_alpha_l1_regularization_strength_term", None))
                # [0.; 1.]. # None by default (1.)
        int_lasso_maximum_number_of_iterations = (
            1_000_000 if dict_parametric is None else dict_parametric.get(
                "int_lasso_maximum_number_of_iterations", 1_000_000))
        int_lasso_optimization_tolerance = (
            0.00001 if dict_parametric is None else dict_parametric.get(
                "int_lasso_optimization_tolerance", 0.00001))
        flt_lasso_cv_length_of_alphas_regularization_path = (
            1. if dict_parametric is None else dict_parametric.get(
                "flt_lasso_cv_length_of_alphas_regularization_path", 1.))
                # (0.; +1.] # 1 by default (0.001); alpha_min / alpha_max
        int_lasso_cv_num_candidate_alphas_on_regularization_path = (
            0. if dict_parametric is None else dict_parametric.get(
                "int_lasso_cv_num_candidate_alphas_on_regularization_path", 0.))
                # [0; +Inf] # 0 by default (100)
        int_lasso_cv_maximum_number_of_iterations = (
            1_000_000 if dict_parametric is None else dict_parametric.get(
                "int_lasso_cv_maximum_number_of_iterations", 1_000_000))
        int_lasso_cv_optimization_tolerance = (
            0.00001 if dict_parametric is None else dict_parametric.get(
                "int_lasso_cv_optimization_tolerance", 0.00001))
        int_lasso_cv_number_of_folds_in_cross_validation = (
            10 if dict_parametric is None else dict_parametric.get(
                "int_lasso_cv_number_of_folds_in_cross_validation", 10))
        #
        bool_estimate_precinct_model_on_aggregated_vote_counts_by_county = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_estimate_precinct_model_on_aggregated_vote_counts_by_county", False))
        bool_approximate_predictions_for_missing_benchmark_precincts_by_actual_values = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_approximate_predictions_for_missing_benchmark_precincts_by_actual_values", False))
        bool_true_each_choice_with_all_other_choices__false_each_choice_with_each_choice = (
            True if dict_parametric is None else dict_parametric.get(
                "bool_true_each_choice_with_all_other_choices__false_each_choice_with_each_choice", True))
        bool_compute_outlier_score_level_1 = (
            True if dict_parametric is None else dict_parametric.get(
                "bool_compute_outlier_score_level_1", True))
        bool_compute_outlier_score_level_2 = (
            True if dict_parametric is None else dict_parametric.get(
                "bool_compute_outlier_score_level_2", True))
        bool_compute_outlier_score_level_3 = (
            True if dict_parametric is None else dict_parametric.get(
                "bool_compute_outlier_score_level_3", True))
        #
        bool_estimate_model_parameters_diagnostics = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_estimate_model_parameters_diagnostics", False))
        bool_save_estimated_models_parameters_to_csv_file = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_save_estimated_models_parameters_to_csv_file", False))
        bool_retain_predicted_vote_counts_columns_in_sorted_precincts_file = (
            True if dict_parametric is None else dict_parametric.get(
                "bool_retain_predicted_vote_counts_columns_in_sorted_precincts_file", True))
        bool_save_outlier_score_stats_to_csv_file = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_save_outlier_score_stats_to_csv_file", False))
        bool_save_model_diagnostics_to_csv_file = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_save_model_diagnostics_to_csv_file", False))
        #
        lst_str_predicted_for_actual_vote_count_column_names = (
            [] if dict_parametric is None else dict_parametric.get(
                "lst_str_predicted_for_actual_vote_count_column_names", []))
        #
        lst_str_predicting_from_ln_of_incremented_actual_vote_count_column_names = (
            [] if dict_parametric is None else dict_parametric.get(
                "lst_str_predicting_from_ln_of_incremented_actual_vote_count_column_names", []))
        lst_str_predicting_from_power_one_quarter_of_actual_vote_count_column_names = (
            [] if dict_parametric is None else dict_parametric.get(
                "lst_str_predicting_from_power_one_quarter_of_actual_vote_count_column_names", []))
        lst_str_predicting_from_square_root_of_actual_vote_count_column_names = (
            [] if dict_parametric is None else dict_parametric.get(
                "lst_str_predicting_from_square_root_of_actual_vote_count_column_names", []))
        lst_str_predicting_from_power_three_quarters_of_actual_vote_count_column_names = (
            [] if dict_parametric is None else dict_parametric.get(
                "lst_str_predicting_from_power_three_quarters_of_actual_vote_count_column_names", []))
        lst_str_predicting_from_actual_vote_count_column_names = (
            [] if dict_parametric is None else dict_parametric.get(
                "lst_str_predicting_from_actual_vote_count_column_names", []))
        lst_str_predicting_from_power_one_and_a_half_of_actual_vote_count_column_names = (
            [] if dict_parametric is None else dict_parametric.get(
                "lst_str_predicting_from_power_one_and_a_half_of_actual_vote_count_column_names", []))
        lst_str_predicting_from_squared_actual_vote_count_column_names = (
            [] if dict_parametric is None else dict_parametric.get(
                "lst_str_predicting_from_squared_actual_vote_count_column_names", []))
        #
        lst_str_predicting_from_ln_of_incremented_benchmark_vote_count_column_names = (
            [] if dict_parametric is None else dict_parametric.get(
                "lst_str_predicting_from_ln_of_incremented_benchmark_vote_count_column_names", []))
        lst_str_predicting_from_power_one_quarter_of_benchmark_vote_count_column_names = (
            [] if dict_parametric is None else dict_parametric.get(
                "lst_str_predicting_from_power_one_quarter_of_benchmark_vote_count_column_names", []))
        lst_str_predicting_from_square_root_of_benchmark_vote_count_column_names = (
            [] if dict_parametric is None else dict_parametric.get(
                "lst_str_predicting_from_square_root_of_benchmark_vote_count_column_names", []))
        lst_str_predicting_from_power_three_quarters_of_benchmark_vote_count_column_names = (
            [] if dict_parametric is None else dict_parametric.get(
                "lst_str_predicting_from_power_three_quarters_of_benchmark_vote_count_column_names", []))
        lst_str_predicting_from_benchmark_vote_count_column_names = (
            [] if dict_parametric is None else dict_parametric.get(
                "lst_str_predicting_from_benchmark_vote_count_column_names", []))
        lst_str_predicting_from_power_one_and_a_half_of_benchmark_vote_count_column_names = (
            [] if dict_parametric is None else dict_parametric.get(
                "lst_str_predicting_from_power_one_and_a_half_of_benchmark_vote_count_column_names", []))
        lst_str_predicting_from_squared_benchmark_vote_count_column_names = (
            [] if dict_parametric is None else dict_parametric.get(
                "lst_str_predicting_from_squared_benchmark_vote_count_column_names", []))
        #
        lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_actual_vote_count_column_names = (
            [] if dict_parametric is None else dict_parametric.get(
                "lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_actual_vote_count_column_names", []))
        lst_pairs_str_predicting_from_square_root_of_actual_interactions_actual_vote_count_column_names = (
            [] if dict_parametric is None else dict_parametric.get(
                "lst_pairs_str_predicting_from_square_root_of_actual_interactions_actual_vote_count_column_names", []))
        lst_pairs_str_predicting_from_actual_interactions_actual_vote_count_column_names = (
            [] if dict_parametric is None else dict_parametric.get(
                "lst_pairs_str_predicting_from_actual_interactions_actual_vote_count_column_names", []))
        #
        lst_pairs_str_predicting_from_fourth_root_of_benchmark_interactions_benchmark_vote_count_column_names = (
            [] if dict_parametric is None else dict_parametric.get(
                "lst_pairs_str_predicting_from_fourth_root_of_benchmark_interactions_benchmark_vote_count_column_names", []))
        lst_pairs_str_predicting_from_square_root_of_benchmark_interactions_benchmark_vote_count_column_names = (
            [] if dict_parametric is None else dict_parametric.get(
                "lst_pairs_str_predicting_from_square_root_of_benchmark_interactions_benchmark_vote_count_column_names", []))
        lst_pairs_str_predicting_from_benchmark_interactions_benchmark_vote_count_column_names = (
            [] if dict_parametric is None else dict_parametric.get(
                "lst_pairs_str_predicting_from_benchmark_interactions_benchmark_vote_count_column_names", []))
        #
        lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_benchmark_vote_count_column_names = (
            [] if dict_parametric is None else dict_parametric.get(
                "lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_benchmark_vote_count_column_names", []))
        lst_pairs_str_predicting_from_square_root_of_actual_interactions_benchmark_vote_count_column_names = (
            [] if dict_parametric is None else dict_parametric.get(
                "lst_pairs_str_predicting_from_square_root_of_actual_interactions_benchmark_vote_count_column_names", []))
        lst_pairs_str_predicting_from_actual_interactions_benchmark_vote_count_column_names = (
            [] if dict_parametric is None else dict_parametric.get(
                "lst_pairs_str_predicting_from_actual_interactions_benchmark_vote_count_column_names", []))
        #######################################################################
        bool_drop_predicted_variable_during_computation_of_predicting_tally = (
            True if dict_parametric is None else dict_parametric.get(
                "bool_drop_predicted_variable_during_computation_of_predicting_tally", True))
        #######################################################################
        bool_predicting_from_ln_of_incremented_actual_tally = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_ln_of_incremented_actual_tally", False))
        bool_predicting_from_power_one_quarter_of_actual_tally = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_power_one_quarter_of_actual_tally", False))
        bool_predicting_from_square_root_of_actual_tally = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_square_root_of_actual_tally", False))
        bool_predicting_from_power_three_quarters_of_actual_tally = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_power_three_quarters_of_actual_tally", False))
        bool_predicting_from_actual_tally = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_actual_tally", False))
        bool_predicting_from_power_one_and_a_half_of_actual_tally = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_power_one_and_a_half_of_actual_tally", False))
        bool_predicting_from_squared_actual_tally = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_squared_actual_tally", False))
        #
        bool_predicting_from_ln_of_incremented_benchmark_tally = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_ln_of_incremented_benchmark_tally", False))
        bool_predicting_from_power_one_quarter_of_benchmark_tally = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_power_one_quarter_of_benchmark_tally", False))
        bool_predicting_from_square_root_of_benchmark_tally = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_square_root_of_benchmark_tally", False))
        bool_predicting_from_power_three_quarters_of_benchmark_tally = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_power_three_quarters_of_benchmark_tally", False))
        bool_predicting_from_benchmark_tally = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_benchmark_tally", False))
        bool_predicting_from_power_one_and_a_half_of_benchmark_tally = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_power_one_and_a_half_of_benchmark_tally", False))
        bool_predicting_from_squared_benchmark_tally = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_squared_benchmark_tally", False))
        #
        bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally", False))
        bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally", False))
        bool_predicting_from_actual_tally_interaction_benchmark_tally = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_actual_tally_interaction_benchmark_tally", False))
        #######################################################################
        bool_predicting_from_ln_of_incremented_actual_tally_interaction_county_indicator = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_ln_of_incremented_actual_tally_interaction_county_indicator", False))
        bool_predicting_from_power_one_quarter_of_actual_tally_interaction_county_indicator = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_power_one_quarter_of_actual_tally_interaction_county_indicator", False))
        bool_predicting_from_square_root_of_actual_tally_interaction_county_indicator = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_square_root_of_actual_tally_interaction_county_indicator", False))
        bool_predicting_from_power_three_quarters_of_actual_tally_interaction_county_indicator = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_power_three_quarters_of_actual_tally_interaction_county_indicator", False))
        bool_predicting_from_actual_tally_interaction_county_indicator = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_actual_tally_interaction_county_indicator", False))
        bool_predicting_from_power_one_and_a_half_of_actual_tally_interaction_county_indicator = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_power_one_and_a_half_of_actual_tally_interaction_county_indicator", False))
        bool_predicting_from_squared_actual_tally_interaction_county_indicator = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_squared_actual_tally_interaction_county_indicator", False))
        #
        bool_predicting_from_ln_of_incremented_benchmark_tally_interaction_county_indicator = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_ln_of_incremented_benchmark_tally_interaction_county_indicator", False))
        bool_predicting_from_power_one_quarter_of_benchmark_tally_interaction_county_indicator = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_power_one_quarter_of_benchmark_tally_interaction_county_indicator", False))
        bool_predicting_from_square_root_of_benchmark_tally_interaction_county_indicator = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_square_root_of_benchmark_tally_interaction_county_indicator", False))
        bool_predicting_from_power_three_quarters_of_benchmark_tally_interaction_county_indicator = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_power_three_quarters_of_benchmark_tally_interaction_county_indicator", False))
        bool_predicting_from_benchmark_tally_interaction_county_indicator = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_benchmark_tally_interaction_county_indicator", False))
        bool_predicting_from_power_one_and_a_half_of_benchmark_tally_interaction_county_indicator = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_power_one_and_a_half_of_benchmark_tally_interaction_county_indicator", False))
        bool_predicting_from_squared_benchmark_tally_interaction_county_indicator = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_squared_benchmark_tally_interaction_county_indicator", False))
        #
        bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator", False))
        bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator", False))
        bool_predicting_from_actual_tally_interaction_benchmark_tally_interaction_county_indicator = (
            False if dict_parametric is None else dict_parametric.get(
                "bool_predicting_from_actual_tally_interaction_benchmark_tally_interaction_county_indicator", False))
        #######################################################################

        #######################################################################
        # Common inputs (output graphics-related)
        #######################################################################
        str_plot_title = dict_graphics[
            "str_plot_title"]
        str_x_axis_label = dict_graphics[
            "str_x_axis_label"]
        str_y_axis_label = dict_graphics[
            "str_y_axis_label"]
        #######################################################################
        lst_str_vote_count_column_names_for_cumulative_tally_for_curves = dict_graphics[
            "lst_str_vote_count_column_names_for_cumulative_tally_for_curves"]
        lst_str_vote_count_column_name_prefixes_to_use_for_curves = dict_graphics[
            "lst_str_vote_count_column_name_prefixes_to_use_for_curves"]
        lst_str_legend_labels_for_curves = dict_graphics[
            "lst_str_legend_labels_for_curves"]
        lst_str_color_names_for_curves = dict_graphics[
            "lst_str_color_names_for_curves"]
        lst_str_linestyles_for_curves = dict_graphics[
            "lst_str_linestyles_for_curves"]
        lst_flt_linewidths_for_curves = dict_graphics[
            "lst_flt_linewidths_for_curves"]
        lst_bool_draw_right_tail_circle_for_curves = dict_graphics[
            "lst_bool_draw_right_tail_circle_for_curves"]
        lst_str_annotations_for_curves = dict_graphics[
            "lst_str_annotations_for_curves"]
        lst_int_annotation_font_sizes_for_curves = dict_graphics[
            "lst_int_annotation_font_sizes_for_curves"]
        lst_bool_display_final_cumulative_percent_of_tally_for_curves = dict_graphics[
            "lst_bool_display_final_cumulative_percent_of_tally_for_curves"]
        lst_int_final_cumulative_percent_of_tally_font_sizes_for_curves = dict_graphics[
            "lst_int_final_cumulative_percent_of_tally_font_sizes_for_curves"]
        lst_int_final_cumulative_percent_of_tally_rounding_precision_for_curves = dict_graphics[
            "lst_int_final_cumulative_percent_of_tally_rounding_precision_for_curves"]
        #######################################################################
        int_plot_title_font_size = dict_graphics[
            "int_plot_title_font_size"]
        int_x_axis_label_font_size = dict_graphics[
            "int_x_axis_label_font_size"]
        int_y_axis_label_font_size = dict_graphics[
            "int_y_axis_label_font_size"]
        int_x_axis_tick_labels_font_size = dict_graphics[
            "int_x_axis_tick_labels_font_size"]
        int_y_axis_tick_labels_font_size = dict_graphics[
            "int_y_axis_tick_labels_font_size"]
        int_legend_font_size = dict_graphics[
            "int_legend_font_size"]
        str_legend_loc = dict_graphics[
            "str_legend_loc"]
        lst_str_text_line_content = dict_graphics[
            "lst_str_text_line_content"]
        lst_str_text_line_color = dict_graphics[
            "lst_str_text_line_color"]
        int_text_line_font_size = dict_graphics[
            "int_text_line_font_size"]
        flt_text_line_x_loc_as_fraction = dict_graphics[
            "flt_text_line_x_loc_as_fraction"]
        flt_text_line_lower_y_loc_as_fraction = dict_graphics[
            "flt_text_line_lower_y_loc_as_fraction"]
        flt_text_line_upper_y_loc_as_fraction = dict_graphics[
            "flt_text_line_upper_y_loc_as_fraction"]
        flt_text_line_abs_delta_y_loc_as_fraction = dict_graphics[
            "flt_text_line_abs_delta_y_loc_as_fraction"]
        str_text_line_start_y_loc = dict_graphics[
            "str_text_line_start_y_loc"]
        bool_display_chart_id = dict_graphics[
            "bool_display_chart_id"]
        int_chart_id_font_size = dict_graphics[
            "int_chart_id_font_size"]
        bool_include_mac_address_in_chart_id = dict_graphics[
            "bool_include_mac_address_in_chart_id"]
        #######################################################################
        lst_flt_min_x = dict_graphics[
            "lst_flt_min_x"]
        lst_flt_max_x = dict_graphics[
            "lst_flt_max_x"]
        lst_flt_xticks_incr = dict_graphics[
            "lst_flt_xticks_incr"]
        lst_flt_min_y = dict_graphics[
            "lst_flt_min_y"]
        lst_flt_max_y = dict_graphics[
            "lst_flt_max_y"]
        lst_flt_yticks_incr = dict_graphics[
            "lst_flt_yticks_incr"]
        lst_flt_plot_width_in_inches = dict_graphics[
            "lst_flt_plot_width_in_inches"]
        lst_flt_plot_height_in_inches = dict_graphics[
            "lst_flt_plot_height_in_inches"]
        lst_int_dots_per_inch_for_png_and_jpg_plots = dict_graphics[
            "lst_int_dots_per_inch_for_png_and_jpg_plots"]
        lst_flt_right_tails_circles_radius_in_pct_pnts = dict_graphics[
            "lst_flt_right_tails_circles_radius_in_pct_pnts"]
        lst_str_file_suffix_for_saving_plot = dict_graphics[
            "lst_str_file_suffix_for_saving_plot"]
        #######################################################################
        if bool_skip_showing_plot :
            bool_show_plot = False
        else :
            bool_show_plot = dict_graphics[
                "bool_show_plot"]
        bool_save_plot_to_png_file = dict_graphics[
            "bool_save_plot_to_png_file"]
        bool_save_plot_to_jpg_file = dict_graphics[
            "bool_save_plot_to_jpg_file"]
        bool_save_plot_to_pdf_file = dict_graphics[
            "bool_save_plot_to_pdf_file"]
        bool_save_plot_to_svg_file = dict_graphics[
            "bool_save_plot_to_svg_file"]
        #######################################################################

        run_election_outliers_job(

            ###################################################################
            # Common inputs (input and data processing-related)
            ###################################################################
            str_input_batch_bat_file_name_ext = str_input_batch_bat_file_name_ext,
            str_input_batch_csv_file_name_ext = str_input_batch_csv_file_name_ext,
            str_input_job_json_file_name = str_input_job_json_file_name,
            bool_apply_non_parametric_model = bool_apply_non_parametric_model,
            bool_apply_parametric_model = bool_apply_parametric_model,
            #
            str_actual_vote_counts_csv_file_name =
                str_actual_vote_counts_csv_file_name,
            lst_str_actual_all_vote_count_column_names =
                lst_str_actual_all_vote_count_column_names,
            str_actual_residual_vote_count_column_name =
                str_actual_residual_vote_count_column_name,
            lst_str_actual_residual_vote_count_column_names =
                lst_str_actual_residual_vote_count_column_names,
            #
            str_benchmark_vote_counts_csv_file_name =
                str_benchmark_vote_counts_csv_file_name,
            lst_str_benchmark_all_vote_count_column_names =
                lst_str_benchmark_all_vote_count_column_names,
            str_benchmark_residual_vote_count_column_name =
                str_benchmark_residual_vote_count_column_name,
            lst_str_benchmark_residual_vote_count_column_names =
                lst_str_benchmark_residual_vote_count_column_names,
            #
            bool_aggregate_vote_counts_by_county =
                bool_aggregate_vote_counts_by_county,
            bool_approximate_missing_benchmark_precincts_by_scaled_benchmark_county =
                bool_approximate_missing_benchmark_precincts_by_scaled_benchmark_county,
            bool_aggregate_missing_benchmark_precincts_into_new_residual_county =
                bool_aggregate_missing_benchmark_precincts_into_new_residual_county,
            #
            int_random_number_generator_seed_for_sorting = 
                int_random_number_generator_seed_for_sorting,
            bool_retain_sorting_criteria_columns =
                bool_retain_sorting_criteria_columns,
            bool_compute_cumulative_actual_vote_count_fractions =
                bool_compute_cumulative_actual_vote_count_fractions,
            bool_save_sorted_precincts_to_csv_file =
                bool_save_sorted_precincts_to_csv_file,
            #
            bool_distribute_job_output_files_in_directories_by_type =
                bool_distribute_job_output_files_in_directories_by_type,
            bool_gather_all_job_files_in_one_directory =
                bool_gather_all_job_files_in_one_directory,
            ###################################################################

            ###################################################################
            # Non-parametric-specific inputs
            ###################################################################
            int_max_num_two_ways_passes =
                int_max_num_two_ways_passes,
            flt_window_size_scaling_factor =
                flt_window_size_scaling_factor,
            lst_str_all_predicting_predicted_vote_count_column_names =
                lst_str_all_predicting_predicted_vote_count_column_names,
            ###################################################################

            ###################################################################
            # Parametric-specific inputs
            ###################################################################
            bool_use_decimal_type =
                bool_use_decimal_type,
            int_decimal_computational_precision =
                int_decimal_computational_precision, # [0; +Inf). # 1024 by default
            int_decimal_reporting_precision =
                int_decimal_reporting_precision, # [0; +Inf). # 16 by default
            int_max_num_iters_for_exact_hypergeom =
                int_max_num_iters_for_exact_hypergeom, # [1; +Inf). # 1_000_000 by default
            int_max_num_iters_for_lanczos_approx_hypergeom =
                int_max_num_iters_for_lanczos_approx_hypergeom, # [1; +Inf). # 500_000 by default
            int_max_num_iters_for_spouge_approx_hypergeom =
                int_max_num_iters_for_spouge_approx_hypergeom, # [1; +Inf). # 500_000 by default
            int_min_sample_size_for_approx_normal =
                int_min_sample_size_for_approx_normal, # [0; +Inf). # 1_000 by default
            flt_max_sample_sz_frac_of_pop_sz_for_approx_normal =
                flt_max_sample_sz_frac_of_pop_sz_for_approx_normal, # [0.; 1.]. # 0.1 by default
            flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal =
                flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal, # [0.; 1.]. # 0.1 by default
            flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal =
                flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal, # [0.; +Inf). # 0.0 by default.
                # z = 2.575829303549 is for 99% in range mu +/- z*sigma
            #
            flt_lasso_alpha_l1_regularization_strength_term =
                flt_lasso_alpha_l1_regularization_strength_term, # [0.; 1.]. # None by default (1.)
            int_lasso_maximum_number_of_iterations =
                int_lasso_maximum_number_of_iterations,
            int_lasso_optimization_tolerance =
                int_lasso_optimization_tolerance,
            flt_lasso_cv_length_of_alphas_regularization_path =
                flt_lasso_cv_length_of_alphas_regularization_path, # (0.; +1.] # 1 by default (0.001); alpha_min / alpha_max
            int_lasso_cv_num_candidate_alphas_on_regularization_path =
                int_lasso_cv_num_candidate_alphas_on_regularization_path, # [0; +Inf] # 0 by default (100)
            int_lasso_cv_maximum_number_of_iterations =
                int_lasso_cv_maximum_number_of_iterations,
            int_lasso_cv_optimization_tolerance =
                int_lasso_cv_optimization_tolerance,
            int_lasso_cv_number_of_folds_in_cross_validation =
                int_lasso_cv_number_of_folds_in_cross_validation,
            #
            bool_estimate_precinct_model_on_aggregated_vote_counts_by_county =
                bool_estimate_precinct_model_on_aggregated_vote_counts_by_county,
            bool_approximate_predictions_for_missing_benchmark_precincts_by_actual_values =
                bool_approximate_predictions_for_missing_benchmark_precincts_by_actual_values,
            bool_true_each_choice_with_all_other_choices__false_each_choice_with_each_choice =
                bool_true_each_choice_with_all_other_choices__false_each_choice_with_each_choice,
            bool_compute_outlier_score_level_1 =
                bool_compute_outlier_score_level_1,
            bool_compute_outlier_score_level_2 =
                bool_compute_outlier_score_level_2,
            bool_compute_outlier_score_level_3 =
                bool_compute_outlier_score_level_3,
            #
            bool_estimate_model_parameters_diagnostics =
                bool_estimate_model_parameters_diagnostics,
            bool_save_estimated_models_parameters_to_csv_file =
                bool_save_estimated_models_parameters_to_csv_file,
            bool_retain_predicted_vote_counts_columns_in_sorted_precincts_file =
                bool_retain_predicted_vote_counts_columns_in_sorted_precincts_file,
            bool_save_outlier_score_stats_to_csv_file =
                bool_save_outlier_score_stats_to_csv_file,
            bool_save_model_diagnostics_to_csv_file =
                bool_save_model_diagnostics_to_csv_file,
            #
            lst_str_predicted_for_actual_vote_count_column_names =
                lst_str_predicted_for_actual_vote_count_column_names,
            #
            lst_str_predicting_from_ln_of_incremented_actual_vote_count_column_names =
                lst_str_predicting_from_ln_of_incremented_actual_vote_count_column_names,
            lst_str_predicting_from_power_one_quarter_of_actual_vote_count_column_names =
                lst_str_predicting_from_power_one_quarter_of_actual_vote_count_column_names,
            lst_str_predicting_from_square_root_of_actual_vote_count_column_names =
                lst_str_predicting_from_square_root_of_actual_vote_count_column_names,
            lst_str_predicting_from_power_three_quarters_of_actual_vote_count_column_names =
                lst_str_predicting_from_power_three_quarters_of_actual_vote_count_column_names,
            lst_str_predicting_from_actual_vote_count_column_names =
                lst_str_predicting_from_actual_vote_count_column_names,
            lst_str_predicting_from_power_one_and_a_half_of_actual_vote_count_column_names =
                lst_str_predicting_from_power_one_and_a_half_of_actual_vote_count_column_names,
            lst_str_predicting_from_squared_actual_vote_count_column_names =
                lst_str_predicting_from_squared_actual_vote_count_column_names,
            #
            lst_str_predicting_from_ln_of_incremented_benchmark_vote_count_column_names =
                lst_str_predicting_from_ln_of_incremented_benchmark_vote_count_column_names,
            lst_str_predicting_from_power_one_quarter_of_benchmark_vote_count_column_names =
                lst_str_predicting_from_power_one_quarter_of_benchmark_vote_count_column_names,
            lst_str_predicting_from_square_root_of_benchmark_vote_count_column_names =
                lst_str_predicting_from_square_root_of_benchmark_vote_count_column_names,
            lst_str_predicting_from_power_three_quarters_of_benchmark_vote_count_column_names =
                lst_str_predicting_from_power_three_quarters_of_benchmark_vote_count_column_names,
            lst_str_predicting_from_benchmark_vote_count_column_names =
                lst_str_predicting_from_benchmark_vote_count_column_names,
            lst_str_predicting_from_power_one_and_a_half_of_benchmark_vote_count_column_names =
                lst_str_predicting_from_power_one_and_a_half_of_benchmark_vote_count_column_names,
            lst_str_predicting_from_squared_benchmark_vote_count_column_names =
                lst_str_predicting_from_squared_benchmark_vote_count_column_names,
            #
            # [(a,b) for a in lst_str_predicting_from_actual_vote_count_column_names
            #        for b in lst_str_predicting_from_actual_vote_count_column_names]
            lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_actual_vote_count_column_names =
                lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_actual_vote_count_column_names,
            lst_pairs_str_predicting_from_square_root_of_actual_interactions_actual_vote_count_column_names =
                lst_pairs_str_predicting_from_square_root_of_actual_interactions_actual_vote_count_column_names,
            lst_pairs_str_predicting_from_actual_interactions_actual_vote_count_column_names =
                lst_pairs_str_predicting_from_actual_interactions_actual_vote_count_column_names,
            #
            # [(a,b) for a in lst_str_predicting_from_benchmark_vote_count_column_names
            #        for b in lst_str_predicting_from_benchmark_vote_count_column_names]
            lst_pairs_str_predicting_from_fourth_root_of_benchmark_interactions_benchmark_vote_count_column_names =
                lst_pairs_str_predicting_from_fourth_root_of_benchmark_interactions_benchmark_vote_count_column_names,
            lst_pairs_str_predicting_from_square_root_of_benchmark_interactions_benchmark_vote_count_column_names =
                lst_pairs_str_predicting_from_square_root_of_benchmark_interactions_benchmark_vote_count_column_names,
            lst_pairs_str_predicting_from_benchmark_interactions_benchmark_vote_count_column_names =
                lst_pairs_str_predicting_from_benchmark_interactions_benchmark_vote_count_column_names,
            #
            # [(a,b) for a in lst_str_predicting_from_actual_vote_count_column_names
            #        for b in lst_str_predicting_from_benchmark_vote_count_column_names]
            lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_benchmark_vote_count_column_names =
                lst_pairs_str_predicting_from_fourth_root_of_actual_interactions_benchmark_vote_count_column_names,
            lst_pairs_str_predicting_from_square_root_of_actual_interactions_benchmark_vote_count_column_names =
                lst_pairs_str_predicting_from_square_root_of_actual_interactions_benchmark_vote_count_column_names,
            lst_pairs_str_predicting_from_actual_interactions_benchmark_vote_count_column_names =
                lst_pairs_str_predicting_from_actual_interactions_benchmark_vote_count_column_names,
            ###################################################################
            bool_drop_predicted_variable_during_computation_of_predicting_tally =
                bool_drop_predicted_variable_during_computation_of_predicting_tally,
            ###################################################################
            bool_predicting_from_ln_of_incremented_actual_tally =
                bool_predicting_from_ln_of_incremented_actual_tally,
            bool_predicting_from_power_one_quarter_of_actual_tally =
                bool_predicting_from_power_one_quarter_of_actual_tally,
            bool_predicting_from_square_root_of_actual_tally =
                bool_predicting_from_square_root_of_actual_tally,
            bool_predicting_from_power_three_quarters_of_actual_tally =
                bool_predicting_from_power_three_quarters_of_actual_tally,
            bool_predicting_from_actual_tally =
                bool_predicting_from_actual_tally,
            bool_predicting_from_power_one_and_a_half_of_actual_tally =
                bool_predicting_from_power_one_and_a_half_of_actual_tally,
            bool_predicting_from_squared_actual_tally =
                bool_predicting_from_squared_actual_tally,
            #
            bool_predicting_from_ln_of_incremented_benchmark_tally =
                bool_predicting_from_ln_of_incremented_benchmark_tally,
            bool_predicting_from_power_one_quarter_of_benchmark_tally =
                bool_predicting_from_power_one_quarter_of_benchmark_tally,
            bool_predicting_from_square_root_of_benchmark_tally =
                bool_predicting_from_square_root_of_benchmark_tally,
            bool_predicting_from_power_three_quarters_of_benchmark_tally =
                bool_predicting_from_power_three_quarters_of_benchmark_tally,
            bool_predicting_from_benchmark_tally =
                bool_predicting_from_benchmark_tally,
            bool_predicting_from_power_one_and_a_half_of_benchmark_tally =
                bool_predicting_from_power_one_and_a_half_of_benchmark_tally,
            bool_predicting_from_squared_benchmark_tally =
                bool_predicting_from_squared_benchmark_tally,
            #
            bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally =
                bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally,
            bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally =
                bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally,
            bool_predicting_from_actual_tally_interaction_benchmark_tally =
                bool_predicting_from_actual_tally_interaction_benchmark_tally,
            ###################################################################
            bool_predicting_from_ln_of_incremented_actual_tally_interaction_county_indicator =
                bool_predicting_from_ln_of_incremented_actual_tally_interaction_county_indicator,
            bool_predicting_from_power_one_quarter_of_actual_tally_interaction_county_indicator =
                bool_predicting_from_power_one_quarter_of_actual_tally_interaction_county_indicator,
            bool_predicting_from_square_root_of_actual_tally_interaction_county_indicator =
                bool_predicting_from_square_root_of_actual_tally_interaction_county_indicator,
            bool_predicting_from_power_three_quarters_of_actual_tally_interaction_county_indicator =
                bool_predicting_from_power_three_quarters_of_actual_tally_interaction_county_indicator,
            bool_predicting_from_actual_tally_interaction_county_indicator =
                bool_predicting_from_actual_tally_interaction_county_indicator,
            bool_predicting_from_power_one_and_a_half_of_actual_tally_interaction_county_indicator =
                bool_predicting_from_power_one_and_a_half_of_actual_tally_interaction_county_indicator,
            bool_predicting_from_squared_actual_tally_interaction_county_indicator =
                bool_predicting_from_squared_actual_tally_interaction_county_indicator,
            #
            bool_predicting_from_ln_of_incremented_benchmark_tally_interaction_county_indicator =
                bool_predicting_from_ln_of_incremented_benchmark_tally_interaction_county_indicator,
            bool_predicting_from_power_one_quarter_of_benchmark_tally_interaction_county_indicator =
                bool_predicting_from_power_one_quarter_of_benchmark_tally_interaction_county_indicator,
            bool_predicting_from_square_root_of_benchmark_tally_interaction_county_indicator =
                bool_predicting_from_square_root_of_benchmark_tally_interaction_county_indicator,
            bool_predicting_from_power_three_quarters_of_benchmark_tally_interaction_county_indicator =
                bool_predicting_from_power_three_quarters_of_benchmark_tally_interaction_county_indicator,
            bool_predicting_from_benchmark_tally_interaction_county_indicator =
                bool_predicting_from_benchmark_tally_interaction_county_indicator,
            bool_predicting_from_power_one_and_a_half_of_benchmark_tally_interaction_county_indicator =
                bool_predicting_from_power_one_and_a_half_of_benchmark_tally_interaction_county_indicator,
            bool_predicting_from_squared_benchmark_tally_interaction_county_indicator =
                bool_predicting_from_squared_benchmark_tally_interaction_county_indicator,
            #
            bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator =
                bool_predicting_from_fourth_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator,
            bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator =
                bool_predicting_from_square_root_of_actual_tally_interaction_benchmark_tally_interaction_county_indicator,
            bool_predicting_from_actual_tally_interaction_benchmark_tally_interaction_county_indicator =
                bool_predicting_from_actual_tally_interaction_benchmark_tally_interaction_county_indicator,
            ###################################################################
            
            ###################################################################
            # Common inputs (output graphics-related)
            ###################################################################
            str_plot_title =
                str_plot_title,
            str_x_axis_label =
                str_x_axis_label,
            str_y_axis_label =
                str_y_axis_label,
            ###################################################################
            lst_str_vote_count_column_names_for_cumulative_tally_for_curves =
                lst_str_vote_count_column_names_for_cumulative_tally_for_curves,
            lst_str_vote_count_column_name_prefixes_to_use_for_curves =
                lst_str_vote_count_column_name_prefixes_to_use_for_curves,
            lst_str_legend_labels_for_curves =
                lst_str_legend_labels_for_curves,
            lst_str_color_names_for_curves =
                lst_str_color_names_for_curves,
            lst_str_linestyles_for_curves =
                lst_str_linestyles_for_curves,
            lst_flt_linewidths_for_curves =
                lst_flt_linewidths_for_curves,
            lst_bool_draw_right_tail_circle_for_curves =
                lst_bool_draw_right_tail_circle_for_curves,
            lst_str_annotations_for_curves =
                lst_str_annotations_for_curves,
            lst_int_annotation_font_sizes_for_curves =
                lst_int_annotation_font_sizes_for_curves,
            lst_bool_display_final_cumulative_percent_of_tally_for_curves =
                lst_bool_display_final_cumulative_percent_of_tally_for_curves,
            lst_int_final_cumulative_percent_of_tally_font_sizes_for_curves =
                lst_int_final_cumulative_percent_of_tally_font_sizes_for_curves,
            lst_int_final_cumulative_percent_of_tally_rounding_precision_for_curves =
                lst_int_final_cumulative_percent_of_tally_rounding_precision_for_curves,
            ###################################################################
            int_plot_title_font_size = int_plot_title_font_size,
            int_x_axis_label_font_size = int_x_axis_label_font_size,
            int_y_axis_label_font_size = int_y_axis_label_font_size,
            int_x_axis_tick_labels_font_size = int_x_axis_tick_labels_font_size,
            int_y_axis_tick_labels_font_size = int_y_axis_tick_labels_font_size,
            int_legend_font_size = int_legend_font_size,
            str_legend_loc = str_legend_loc,
            lst_str_text_line_content = lst_str_text_line_content,
            lst_str_text_line_color = lst_str_text_line_color,
            int_text_line_font_size = int_text_line_font_size,
            flt_text_line_x_loc_as_fraction = flt_text_line_x_loc_as_fraction,
            flt_text_line_lower_y_loc_as_fraction = \
                flt_text_line_lower_y_loc_as_fraction,
            flt_text_line_upper_y_loc_as_fraction = \
                flt_text_line_upper_y_loc_as_fraction,
            flt_text_line_abs_delta_y_loc_as_fraction = \
                flt_text_line_abs_delta_y_loc_as_fraction,
            str_text_line_start_y_loc = str_text_line_start_y_loc,
            bool_display_chart_id = bool_display_chart_id,
            int_chart_id_font_size = int_chart_id_font_size,
            bool_include_mac_address_in_chart_id =
                bool_include_mac_address_in_chart_id,
            ###################################################################
            lst_flt_min_x = lst_flt_min_x,
            lst_flt_max_x = lst_flt_max_x,
            lst_flt_xticks_incr = lst_flt_xticks_incr,
            lst_flt_min_y = lst_flt_min_y,
            lst_flt_max_y = lst_flt_max_y,
            lst_flt_yticks_incr = lst_flt_yticks_incr,
            lst_flt_plot_width_in_inches = lst_flt_plot_width_in_inches,
            lst_flt_plot_height_in_inches = lst_flt_plot_height_in_inches,
            lst_int_dots_per_inch_for_png_and_jpg_plots =
                lst_int_dots_per_inch_for_png_and_jpg_plots,
            lst_flt_right_tails_circles_radius_in_pct_pnts =
                lst_flt_right_tails_circles_radius_in_pct_pnts,
            lst_str_file_suffix_for_saving_plot =
                lst_str_file_suffix_for_saving_plot,
            ###################################################################
            bool_show_plot = bool_show_plot,
            bool_save_plot_to_png_file = bool_save_plot_to_png_file,
            bool_save_plot_to_jpg_file = bool_save_plot_to_jpg_file,
            bool_save_plot_to_pdf_file = bool_save_plot_to_pdf_file,
            bool_save_plot_to_svg_file = bool_save_plot_to_svg_file,
            ###################################################################
            )
    except Exception as exception :
        print_exception_message(exception)


###############################################################################

def process_input_batch_csv_file(
        str_input_batch_bat_file_name_ext,
        str_input_batch_csv_file_name_ext,) :
    '''<function description>
    
    <calling functions>
    <called functions>

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
    if str_input_batch_csv_file_name_ext.startswith(".") or \
       "/" in str_input_batch_csv_file_name_ext or \
       "\\" in str_input_batch_csv_file_name_ext or \
       ":" in str_input_batch_csv_file_name_ext :
        # Already an absolute or relative path with name, not just file name.
        str_input_batch_csv_full_file_name = str_input_batch_csv_file_name_ext
    else :
        str_input_batch_csv_full_file_name = os.path.join(
            ".", STR_INPUT_BATCHES_CSV_PATH, str_input_batch_csv_file_name_ext)
    df_input_jobs_batch = pd.read_csv(str_input_batch_csv_full_file_name)
    int_total_num_of_jobs_to_process = df_input_jobs_batch[
        'bool_process_job_json_file'].values.sum()
    print("Starting batch has " + str(int_total_num_of_jobs_to_process) +
        " pending job(s) out of " + str(len(df_input_jobs_batch.index)) +
        " total job(s).")
    print()
    int_num_of_jobs_processed = 0
    int_batch_start_time_ns = time.monotonic_ns()
    for (int_job_index, row) in df_input_jobs_batch.iterrows() :
        str_input_job_json_file_name = str(row[
            "str_input_job_json_file_name"])
        bool_process_job_json_file = bool(row[
            "bool_process_job_json_file"])
        bool_skip_non_parametric_model = bool(row[
            "bool_skip_non_parametric_model"])
        bool_skip_parametric_model = bool(row[
            "bool_skip_parametric_model"])
        bool_skip_showing_plot = bool(row[
            "bool_skip_showing_plot"])
        if bool_process_job_json_file :
            print('Starting job #' + str(int_job_index + 1) + ' (' +
                  str(int_num_of_jobs_processed + 1) + ' of ' +
                  str(int_total_num_of_jobs_to_process) + '): "' +
                  str_input_job_json_file_name + '".')
            int_job_start_time_ns = time.monotonic_ns()
            process_input_job_json_file(
                str_input_job_json_file_name =
                    str_input_job_json_file_name,
                str_input_batch_bat_file_name_ext =
                    str_input_batch_bat_file_name_ext,
                str_input_batch_csv_file_name_ext =
                    str_input_batch_csv_file_name_ext,
                bool_skip_non_parametric_model =
                    bool_skip_non_parametric_model,
                bool_skip_parametric_model = bool_skip_parametric_model,
                bool_skip_showing_plot = bool_skip_showing_plot,)
            int_job_end_time_ns = time.monotonic_ns()
            flt_job_runtime_sec = round(float(
                int_job_end_time_ns - int_job_start_time_ns) /
                FLT_NANOSECONDS_PER_SECOND, 3)
            print('Job runtime was ' + str(flt_job_runtime_sec) + ' seconds.')
            print('Completed job #' + str(int_job_index + 1) + ' (' +
                  str(int_num_of_jobs_processed + 1) + ' of ' +
                  str(int_total_num_of_jobs_to_process) + '): "' +
                  str_input_job_json_file_name + '".')
            print()
            int_num_of_jobs_processed += 1
    int_batch_end_time_ns = time.monotonic_ns()
    flt_batch_runtime_sec = round(float(
        int_batch_end_time_ns - int_batch_start_time_ns) /
        FLT_NANOSECONDS_PER_SECOND, 3)
    print("Finished batch with " + str(int_total_num_of_jobs_to_process) +
          " completed job(s) out of " + str(len(df_input_jobs_batch.index)) +
          " total job(s).")
    print('Batch had total runtime ' + str(flt_batch_runtime_sec) +
          ' seconds.')

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
    if len(sys.argv) == 1 :
        lst_str_input_batch_csv_file_names = \
            [STR_INPUT_JOBS_BATCH_CSV_FILE_NAME]
    elif len(sys.argv) > 1 :
        lst_str_input_batch_csv_file_names = sys.argv[1:]
    else :
        lst_str_input_batch_csv_file_names = []

    int_total_num_of_batches_to_process = \
        len(lst_str_input_batch_csv_file_names)
    int_num_of_batches_processed = 0

    print('Starting batch sequence with ' +
          str(int_total_num_of_batches_to_process) + ' batch(es).')
    print()

    int_batch_sequence_start_time_ns = time.monotonic_ns()
    for (int_batch_index, str_input_batch_csv_file_name) in \
            enumerate(lst_str_input_batch_csv_file_names) :

        if not str_input_batch_csv_file_name.endswith(".csv") :
            str_input_batch_bat_file_name_ext = \
                str_input_batch_csv_file_name + ".bat"
        else :
            str_input_batch_bat_file_name_ext = str_input_batch_csv_file_name[
                :(len(str_input_batch_csv_file_name) - len(".csv"))] + ".bat"

        if not str_input_batch_csv_file_name.endswith(".csv") :
            str_input_batch_csv_file_name_ext = \
                str_input_batch_csv_file_name + ".csv"
        else :
            str_input_batch_csv_file_name_ext = str_input_batch_csv_file_name

        print('Starting batch #' + str(int_batch_index + 1) + ' (' +
              str(int_num_of_batches_processed + 1) + ' of ' +
              str(int_total_num_of_batches_to_process) + '): "' +
              str_input_batch_csv_file_name_ext + '".')
        process_input_batch_csv_file(
            str_input_batch_bat_file_name_ext = \
                str_input_batch_bat_file_name_ext,
            str_input_batch_csv_file_name_ext = \
                str_input_batch_csv_file_name_ext)
        print('Completed batch #' + str(int_batch_index + 1) + ' (' +
              str(int_num_of_batches_processed + 1) + ' of ' +
              str(int_total_num_of_batches_to_process) + '): "' +
              str_input_batch_csv_file_name_ext + '".')
        print()
        int_num_of_batches_processed += 1
    int_batch_sequence_end_time_ns = time.monotonic_ns()

    flt_batch_sequence_runtime_sec = round(float(
        int_batch_sequence_end_time_ns - int_batch_sequence_start_time_ns) /
        FLT_NANOSECONDS_PER_SECOND, 3)
    print('Finished batch sequence with ' +
          str(int_total_num_of_batches_to_process) +
          ' batch(es) and total runtime of ' +
          str(flt_batch_sequence_runtime_sec) +
          ' seconds.')

###############################################################################

if __name__ ==  '__main__' :
    main()

###############################################################################
