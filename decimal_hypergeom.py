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

import sys
import math
import numpy as np
from scipy.stats import hypergeom, norm
from decimal import Decimal, getcontext, localcontext

###############################################################################
# !!!
# utilities

INT_DECIMAL_COMPUTATIONAL_PRECISION = 1024
INT_DECIMAL_REPORTING_PRECISION = 16

def split_float(number, str_int_precision =
                str(INT_DECIMAL_REPORTING_PRECISION)) :
    # float 9.88888888888889 has 14 digits.
    # float 0.1111111111111111 has 16 digits. Use 16 digits:
    if math.isnan(number) :
        (flt_coefficient, int_exponent) = (math.nan, -(sys.maxsize * 2 + 1))
    else :
        (str_coefficient, str_exponent) = \
            ("{:." + str_int_precision + "E}").format(number).split("E")
        (flt_coefficient,int_exponent) = (float(str_coefficient),
                                          int(str_exponent))
        if flt_coefficient == 0. :
            int_exponent = -(sys.maxsize * 2 + 1)
    return (flt_coefficient, int_exponent)


def _simplify_fraction_for_hypergeom_pmf(lst_lst_num, lst_lst_den) :

    int_num_stages = min(len(lst_lst_num), len(lst_lst_den))
    for int_fraction_cancellation_stage in range(int_num_stages) :

        # find range index NI in numerator [1, NU] with maximum NU
        int_widest_num_range = -1
        int_widest_num_range_ind = -1
        for i in range(len(lst_lst_num)) :
            if lst_lst_num[i][0] == 1 :
                int_current_num_range = lst_lst_num[i][1]
                if int_current_num_range > int_widest_num_range :
                    int_widest_num_range = int_current_num_range
                    int_widest_num_range_ind = i

        # find range index DI in denominator [1, DU] with maximum DU
        int_widest_den_range = -1
        int_widest_den_range_ind = -1
        for j in range(len(lst_lst_den)) :
            if lst_lst_den[j][0] == 1 :
                int_current_den_range = lst_lst_den[j][1]
                if int_current_den_range > int_widest_den_range :
                    int_widest_den_range = int_current_den_range
                    int_widest_den_range_ind = j

        # If NU > DU :
        if int_widest_num_range > int_widest_den_range :
            # [1, NU] -> [DU+1, NU]
            # [1, DU] -> [1, 1]
            lst_lst_num[int_widest_num_range_ind][0] = int_widest_den_range + 1
            lst_lst_den[int_widest_den_range_ind][1] = 1
        # elif NU < DU :
        elif int_widest_num_range < int_widest_den_range :
            # [1, NU] -> [1, 1]
            # [1, DU] -> [NU+1, DU]
            lst_lst_num[int_widest_num_range_ind][1] = 1
            lst_lst_den[int_widest_den_range_ind][0] = int_widest_num_range + 1
        # else : # if NU == DU
        else : # int_widest_num_range == int_widest_den_range
            # [1, NU] -> [1, 1]
            # [1, DU] -> [1, 1]
            lst_lst_num[int_widest_num_range_ind][1] = 1
            lst_lst_den[int_widest_den_range_ind][1] = 1

    return (lst_lst_num, lst_lst_den)


def _compute_fraction_for_hypergeom_pmf(
        lst_lst_num,
        lst_lst_den,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = True,
        int_decimal_computational_precision = INT_DECIMAL_COMPUTATIONAL_PRECISION,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,) :

    with localcontext() as ctx :
        # Perform a high precision calculation
        ctx.prec = int_decimal_computational_precision
        int_current_num_range_ind = 0
        int_current_num_range_elem = lst_lst_num[int_current_num_range_ind][0]
        int_current_den_range_ind = 0
        int_current_den_range_elem = lst_lst_den[int_current_den_range_ind][0]
        if bool_use_decimal_type :
            pmf = Decimal(1.)
        else :
            pmf = 1.
        pmf *= int_current_num_range_elem
        pmf /= int_current_den_range_elem
        while int_current_num_range_ind < len(lst_lst_num) and \
              int_current_den_range_ind < len(lst_lst_den) :
            if pmf > 1. :
                int_current_den_range_elem += 1
                if int_current_den_range_elem <= lst_lst_den[
                        int_current_den_range_ind][1]:
                    pmf /= int_current_den_range_elem
                else :
                    int_current_den_range_ind += 1
                    if int_current_den_range_ind < len(lst_lst_den) :
                        int_current_den_range_elem = lst_lst_den[
                            int_current_den_range_ind][0]
                        pmf /= int_current_den_range_elem
                    else :
                        int_current_den_range_elem = 1
            else :
                int_current_num_range_elem += 1
                if int_current_num_range_elem <= lst_lst_num[
                        int_current_num_range_ind][1]:
                    pmf *= int_current_num_range_elem
                else :
                    int_current_num_range_ind += 1
                    if int_current_num_range_ind < len(lst_lst_num) :
                        int_current_num_range_elem = lst_lst_num[
                            int_current_num_range_ind][0]
                        pmf *= int_current_num_range_elem
                    else :
                        int_current_num_range_elem = 1

        if int_current_num_range_ind < len(lst_lst_num) and \
           int_current_den_range_ind == len(lst_lst_den) :
            while int_current_num_range_ind < len(lst_lst_num) :
                int_current_num_range_elem += 1
                if int_current_num_range_elem <= lst_lst_num[
                        int_current_num_range_ind][1]:
                    pmf *= int_current_num_range_elem
                else :
                    int_current_num_range_ind += 1
                    if int_current_num_range_ind < len(lst_lst_num) :
                        int_current_num_range_elem = lst_lst_num[
                            int_current_num_range_ind][0]
                        pmf *= int_current_num_range_elem
                    else :
                        int_current_num_range_elem = 1

        if int_current_num_range_ind == len(lst_lst_num) and \
           int_current_den_range_ind < len(lst_lst_den) :
            while int_current_den_range_ind < len(lst_lst_den) :
                int_current_den_range_elem += 1
                if int_current_den_range_elem <= lst_lst_den[
                        int_current_den_range_ind][1]:
                    pmf /= int_current_den_range_elem
                else :
                    int_current_den_range_ind += 1
                    if int_current_den_range_ind < len(lst_lst_den) :
                        int_current_den_range_elem = lst_lst_den[
                            int_current_den_range_ind][0]
                        pmf /= int_current_den_range_elem
                    else :
                        int_current_den_range_elem = 1

    getcontext().prec = int_decimal_reporting_precision
    pmf = +pmf # Round the final result back to the default precision
    if bool_split_into_coeff_and_base_ten_exponent :
        return split_float(number = pmf)
    else :
        return pmf


def _sum_hypergeom_pmfs(
        k, M, n, N,
        bool_use_half_of_boundary_pmf = False,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = True,
        int_decimal_computational_precision = INT_DECIMAL_COMPUTATIONAL_PRECISION,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,
        bool_true_cdf_false_sf_none_min = True,
        bool_numerical_approximation = False,
        bool_approx_true_lanczos_false_spouge = True,
        ) :

    if bool_numerical_approximation :
        if bool_approx_true_lanczos_false_spouge :
            # Lanczos Approximation to the Log Gamma Function.
            # The code is based on the idea from this source:
            # http://mrob.com/pub/ries/lanczos-gamma.html#fn_4
            if bool_use_decimal_type :
                with localcontext() as ctx :
                    ctx.prec = int_decimal_computational_precision
                    '''
                    b1 = Decimal("2.5066282751072974716040095600075")
                    b2 = Decimal("190.95517189307639668340239599573")
                    b3 = Decimal("216.83668184372796999227712657839")
                    b4 = Decimal("60.194417640233328256162764625455")
                    b5 = Decimal("3.0875132392854582647683040300775")
                    b6 = Decimal("0.00302963870525306071894639603305")
                    b7 = Decimal("1.35238595907259587106470354765e-5")
                    c1 = Decimal("1")
                    c1p5 = Decimal("1.5")
                    c2 = Decimal("2")
                    c3 = Decimal("3")
                    c4 = Decimal("4")
                    c5 = Decimal("5")
                    c6 = Decimal("6")
                    c6p5 = Decimal("6.5")
                    c7 = Decimal("7")
                    '''
                    _n = 13
                    _g = 6.02468
                    g = Decimal(format(_g, ".15g"))
                    lst_p = [
                        Decimal('2.50662827463100027016561693547825754072337540757200082367'),
                        Decimal('589.5105778528748081083440754114017970881744165562870628551'),
                        Decimal('-888.02534533501237172652316346971976076979770518225084992'),
                        Decimal('395.838757159176115722783354674181284804742524064707536'),
                        Decimal('-53.21395413703462595543160513282731562408231287715217'),
                        Decimal('1.2771826424117897170129599091132309034574601465637'),
                        Decimal('-0.0004046170655169348179547621938348030520821860594'),
                        Decimal('-0.000007347585209589689589422864286753037601670817'),
                        Decimal('0.000008208805239871217130461324758555114442513669'),
                        Decimal('-0.000005159542415359044989159951746415314096492689'),
                        Decimal('0.000002319630748531474375016814467281014423862250'),
                        Decimal('-6.67124339402896748175608182929918031465333E-7'),
                        Decimal('9.06038883356544784242812502552213897255901E-8')]
            else :
                '''
                b1 = np.float64("2.5066282751072974716040095600075")
                b2 = np.float64("190.95517189307639668340239599573")
                b3 = np.float64("216.83668184372796999227712657839")
                b4 = np.float64("60.194417640233328256162764625455")
                b5 = np.float64("3.0875132392854582647683040300775")
                b6 = np.float64("0.00302963870525306071894639603305")
                b7 = np.float64("1.35238595907259587106470354765e-5")
                c1 = np.float64("1")
                c1p5 = np.float64("1.5")
                c2 = np.float64("2")
                c3 = np.float64("3")
                c4 = np.float64("4")
                c5 = np.float64("5")
                c6 = np.float64("6")
                c6p5 = np.float64("6.5")
                c7 = np.float64("7")
                '''
                _n = 13
                _g = 6.02468
                g = np.float64(_g)
                lst_p = [
                    np.float64('2.50662827463100027016561693547825754072337540757200082367'),
                    np.float64('589.5105778528748081083440754114017970881744165562870628551'),
                    np.float64('-888.02534533501237172652316346971976076979770518225084992'),
                    np.float64('395.838757159176115722783354674181284804742524064707536'),
                    np.float64('-53.21395413703462595543160513282731562408231287715217'),
                    np.float64('1.2771826424117897170129599091132309034574601465637'),
                    np.float64('-0.0004046170655169348179547621938348030520821860594'),
                    np.float64('-0.000007347585209589689589422864286753037601670817'),
                    np.float64('0.000008208805239871217130461324758555114442513669'),
                    np.float64('-0.000005159542415359044989159951746415314096492689'),
                    np.float64('0.000002319630748531474375016814467281014423862250'),
                    np.float64('-6.67124339402896748175608182929918031465333E-7'),
                    np.float64('9.06038883356544784242812502552213897255901E-8')]
        else : # if not bool_approx_true_lanczos_false_spouge :
            # https://en.wikipedia.org/wiki/Spouge%27s_approximation
            if bool_use_decimal_type :
                with localcontext() as ctx :
                    ctx.prec = int_decimal_computational_precision
                    lst_c = [
                        Decimal('2.506628274631000502415765284811045253006986740609938316629923576'),
                        Decimal('777986313.1091454928811402005400108162295251348113128426334961852'),
                        Decimal('-5014289818.386629570797576541967490103652258514832569674279751820'),
                        Decimal('14391249419.00289373155156170253309810152458462806088822279545592'),
                        Decimal('-24265005794.6668308801208968281605470912586269268831951251701441'),
                        Decimal('26706480135.5683394082012008772502119322670635833810445328827937'),
                        Decimal('-20167204252.3651678703249776522516454209876558668604305771976457'),
                        Decimal('10693689215.66914860975029079126690660069469100630653881202448081'),
                        Decimal('-4008321901.881434750140706868171689145285101201137451110548606052'),
                        Decimal('1055739088.101602099423513581371922881638807456278410924429883147'),
                        Decimal('-191947202.1477629664251405783223503351353012277209941617708328718'),
                        Decimal('23357892.39307511798087638904470551337994322140297269819420234213'),
                        Decimal('-1814408.148359252731818439838993707677526340125003219834376779638'),
                        Decimal('83839.73741140680219453398758050281620165654670033425860998174394'),
                        Decimal('-2072.661857059051124169469319593967128917693485708949680126818223'),
                        Decimal('23.23427473123225253219821352451089625565552416728329648391674089'),
                        Decimal('-0.08966195046443543309219652814526417878744371424147365192636459614'),
                        Decimal('0.00007157552707474602995541065998770807262371282572284672472244648859'),
                        Decimal('-3.850750431615709437581025834192410706598703829407022527230216420E-9'),
                        Decimal('4.245740647764882878237117834434502631724174404640504624072297716E-16')]
            else :
                lst_c = [
                    np.float64('2.506628274631000502415765284811045253006986740609938316629923576'),
                    np.float64('777986313.1091454928811402005400108162295251348113128426334961852'),
                    np.float64('-5014289818.386629570797576541967490103652258514832569674279751820'),
                    np.float64('14391249419.00289373155156170253309810152458462806088822279545592'),
                    np.float64('-24265005794.6668308801208968281605470912586269268831951251701441'),
                    np.float64('26706480135.5683394082012008772502119322670635833810445328827937'),
                    np.float64('-20167204252.3651678703249776522516454209876558668604305771976457'),
                    np.float64('10693689215.66914860975029079126690660069469100630653881202448081'),
                    np.float64('-4008321901.881434750140706868171689145285101201137451110548606052'),
                    np.float64('1055739088.101602099423513581371922881638807456278410924429883147'),
                    np.float64('-191947202.1477629664251405783223503351353012277209941617708328718'),
                    np.float64('23357892.39307511798087638904470551337994322140297269819420234213'),
                    np.float64('-1814408.148359252731818439838993707677526340125003219834376779638'),
                    np.float64('83839.73741140680219453398758050281620165654670033425860998174394'),
                    np.float64('-2072.661857059051124169469319593967128917693485708949680126818223'),
                    np.float64('23.23427473123225253219821352451089625565552416728329648391674089'),
                    np.float64('-0.08966195046443543309219652814526417878744371424147365192636459614'),
                    np.float64('0.00007157552707474602995541065998770807262371282572284672472244648859'),
                    np.float64('-3.850750431615709437581025834192410706598703829407022527230216420E-9'),
                    np.float64('4.245740647764882878237117834434502631724174404640504624072297716E-16')]
    def approx_hypergeom_pmf_inner(
            k, M, n, N,
            bool_split_into_coeff_and_base_ten_exponent = False,
            bool_use_decimal_type = True,
            int_decimal_computational_precision = INT_DECIMAL_COMPUTATIONAL_PRECISION,
            int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,
            bool_approx_true_lanczos_false_spouge = True,) :
        if bool_approx_true_lanczos_false_spouge :
            if bool_use_decimal_type :
                with localcontext() as ctx :
                    ctx.prec = int_decimal_computational_precision
                    '''
                    log_gamma_lanczos_approx = lambda a : ctx.ln(
                        (b1+b2/(a+c2)-b3/(a+c3)+b4/(a+c4)-b5/(a+c5)+b6/(a+c6)-b7/(a+c7))/
                        (a+c1))-(a+c6p5)+(a+c1p5)*ctx.ln(a+c6p5)
                    approx_hypergeom_pmf_lambda = lambda k, M, n, N : ctx.exp(
                        -log_gamma_lanczos_approx(k)
                        +log_gamma_lanczos_approx(M-N)
                        -log_gamma_lanczos_approx(n-k)
                        +log_gamma_lanczos_approx(N)
                        -log_gamma_lanczos_approx(N-k)
                        +log_gamma_lanczos_approx(n)
                        -log_gamma_lanczos_approx(M)
                        +log_gamma_lanczos_approx(M-n)
                        -log_gamma_lanczos_approx(M-N-n+k))
                    pmf = approx_hypergeom_pmf_lambda(k=k, M=M, n=n, N=N)
                    '''
                    def log_gamma_lanczos_approx(z, _n, g, lst_p) :
                        r = lst_p[0]
                        for i in range(1,_n) :
                            r += lst_p[i] / (z+i)
                        half = Decimal(1) / Decimal(2)
                        t = Decimal(z) + g + half
                        log_g = ctx.ln(r) + (Decimal(z) + half) * ctx.ln(t) - t
                        return log_g
                    approx_hypergeom_pmf_lambda = lambda k, M, n, N, _n, g, lst_p : ctx.exp(
                        -log_gamma_lanczos_approx(z=k, _n=_n, g=g, lst_p=lst_p)
                        +log_gamma_lanczos_approx(z=M-N, _n=_n, g=g, lst_p=lst_p)
                        -log_gamma_lanczos_approx(z=n-k, _n=_n, g=g, lst_p=lst_p)
                        +log_gamma_lanczos_approx(z=N, _n=_n, g=g, lst_p=lst_p)
                        -log_gamma_lanczos_approx(z=N-k, _n=_n, g=g, lst_p=lst_p)
                        +log_gamma_lanczos_approx(z=n, _n=_n, g=g, lst_p=lst_p)
                        -log_gamma_lanczos_approx(z=M, _n=_n, g=g, lst_p=lst_p)
                        +log_gamma_lanczos_approx(z=M-n, _n=_n, g=g, lst_p=lst_p)
                        -log_gamma_lanczos_approx(z=M-N-n+k, _n=_n, g=g, lst_p=lst_p))
                    pmf = approx_hypergeom_pmf_lambda(
                        k=k, M=M, n=n, N=N, _n=_n, g=g, lst_p=lst_p)
                getcontext().prec = int_decimal_reporting_precision
                pmf = +pmf # Round the final result back to the default precision
            else :
                '''
                log_gamma_lanczos_approx = lambda a : np.log(
                    (b1+b2/(a+c2)-b3/(a+c3)+b4/(a+c4)-b5/(a+c5)+b6/(a+c6)-b7/(a+c7))/
                    (a+c1))-(a+c6p5)+(a+c1p5)*np.log(a+c6p5)
                approx_hypergeom_pmf_lambda = lambda k, M, n, N : np.exp(
                    -log_gamma_lanczos_approx(k)
                    +log_gamma_lanczos_approx(M-N)
                    -log_gamma_lanczos_approx(n-k)
                    +log_gamma_lanczos_approx(N)
                    -log_gamma_lanczos_approx(N-k)
                    +log_gamma_lanczos_approx(n)
                    -log_gamma_lanczos_approx(M)
                    +log_gamma_lanczos_approx(M-n)
                    -log_gamma_lanczos_approx(M-N-n+k))
                pmf = approx_hypergeom_pmf_lambda(k=k, M=M, n=n, N=N)
                '''
                def log_gamma_lanczos_approx(z, _n, g, lst_p) :
                    r = lst_p[0]
                    for i in range(1,_n) :
                        r += lst_p[i] / (z+i)
                    half = np.float64(.5)
                    t = np.float64(z) + g + half
                    log_g = np.log(r) + (np.float64(z) + half) * np.log(t) - t
                    return log_g
                approx_hypergeom_pmf_lambda = lambda k, M, n, N, _n, g, lst_p : np.exp(
                    -log_gamma_lanczos_approx(z=k, _n=_n, g=g, lst_p=lst_p)
                    +log_gamma_lanczos_approx(z=M-N, _n=_n, g=g, lst_p=lst_p)
                    -log_gamma_lanczos_approx(z=n-k, _n=_n, g=g, lst_p=lst_p)
                    +log_gamma_lanczos_approx(z=N, _n=_n, g=g, lst_p=lst_p)
                    -log_gamma_lanczos_approx(z=N-k, _n=_n, g=g, lst_p=lst_p)
                    +log_gamma_lanczos_approx(z=n, _n=_n, g=g, lst_p=lst_p)
                    -log_gamma_lanczos_approx(z=M, _n=_n, g=g, lst_p=lst_p)
                    +log_gamma_lanczos_approx(z=M-n, _n=_n, g=g, lst_p=lst_p)
                    -log_gamma_lanczos_approx(z=M-N-n+k, _n=_n, g=g, lst_p=lst_p))
                pmf = approx_hypergeom_pmf_lambda(
                    k=k, M=M, n=n, N=N, _n=_n, g=g, lst_p=lst_p)
        else : # if not bool_approx_true_lanczos_false_spouge :
            if bool_use_decimal_type :
                with localcontext() as ctx :
                    ctx.prec = int_decimal_computational_precision
                    log_gamma_spouge_approx = lambda z, lst_c : (
                        ctx.ln(z+len(lst_c)) * (z+Decimal(1)/Decimal(2)) +
                        (-z-len(lst_c)) +
                        ctx.ln(sum([lst_c[0]] + [
                            lst_c[k] / (z+k) for k in range(1,len(lst_c))])))
                    approx_hypergeom_pmf_lambda = lambda k, M, n, N, lst_c : ctx.exp(
                        -log_gamma_spouge_approx(k, lst_c)
                        +log_gamma_spouge_approx(M-N, lst_c)
                        -log_gamma_spouge_approx(n-k, lst_c)
                        +log_gamma_spouge_approx(N, lst_c)
                        -log_gamma_spouge_approx(N-k, lst_c)
                        +log_gamma_spouge_approx(n, lst_c)
                        -log_gamma_spouge_approx(M, lst_c)
                        +log_gamma_spouge_approx(M-n, lst_c)
                        -log_gamma_spouge_approx(M-N-n+k, lst_c))
                    pmf = approx_hypergeom_pmf_lambda(k=k, M=M, n=n, N=N, lst_c=lst_c)
                getcontext().prec = int_decimal_reporting_precision
                pmf = +pmf # Round the final result back to the default precision
            else :
                log_gamma_spouge_approx = lambda z, lst_c : (
                    np.log(z+len(lst_c)) * (z+Decimal(1)/Decimal(2)) +
                    (-z-len(lst_c)) +
                    np.log(sum([lst_c[0]] + [
                        lst_c[k] / (z+k) for k in range(1,len(lst_c))])))
                approx_hypergeom_pmf_lambda = lambda k, M, n, N, lst_c : np.exp(
                    -log_gamma_spouge_approx(k, lst_c)
                    +log_gamma_spouge_approx(M-N, lst_c)
                    -log_gamma_spouge_approx(n-k, lst_c)
                    +log_gamma_spouge_approx(N, lst_c)
                    -log_gamma_spouge_approx(N-k, lst_c)
                    +log_gamma_spouge_approx(n, lst_c)
                    -log_gamma_spouge_approx(M, lst_c)
                    +log_gamma_spouge_approx(M-n, lst_c)
                    -log_gamma_spouge_approx(M-N-n+k, lst_c))
                pmf = approx_hypergeom_pmf_lambda(k=k, M=M, n=n, N=N, lst_c=lst_c)
        if bool_split_into_coeff_and_base_ten_exponent :
            return split_float(number = pmf)
        else :
            return pmf

    if (M == n or # => k == N
        M == N or # => k == n
        N == k or
        ((M - N) == (n - k) and bool_use_half_of_boundary_pmf) or
        n == 0 or # => k == 0
        N == 0    # => k == 0
        ) : 
        # Cases when (CDF == 1 and SF == 0) or (CDF == 0 and SF == 1)
        if bool_numerical_approximation :
            half_pmf = approx_hypergeom_pmf_inner(
                k=k, M=M, n=n, N=N,
                bool_split_into_coeff_and_base_ten_exponent = \
                    bool_split_into_coeff_and_base_ten_exponent,
                bool_use_decimal_type = bool_use_decimal_type,
                int_decimal_computational_precision = \
                    int_decimal_computational_precision,
                int_decimal_reporting_precision = \
                    int_decimal_reporting_precision,
                bool_approx_true_lanczos_false_spouge = \
                    bool_approx_true_lanczos_false_spouge,
                ) / 2
        else :
            half_pmf = exact_hypergeom_pmf(
                k=k, M=M, n=n, N=N,
                bool_split_into_coeff_and_base_ten_exponent = False,
                bool_use_decimal_type = bool_use_decimal_type,
                int_decimal_computational_precision = \
                    int_decimal_computational_precision,
                int_decimal_reporting_precision = \
                    int_decimal_reporting_precision,) / 2
        if (M - N) == (n - k) and bool_use_half_of_boundary_pmf :
            if bool_true_cdf_false_sf_none_min :
                cdf = (Decimal(0) if bool_use_decimal_type else np.float64(0))
                if bool_use_half_of_boundary_pmf :
                    cdf += half_pmf
            else :
                sf = (Decimal(1) if bool_use_decimal_type else np.float64(1))
                if bool_use_half_of_boundary_pmf :
                    sf -= half_pmf
        else :
            if bool_true_cdf_false_sf_none_min :
                cdf = (Decimal(1) if bool_use_decimal_type else np.float64(1))
                if bool_use_half_of_boundary_pmf :
                    cdf -= half_pmf
            else :
                sf = (Decimal(0) if bool_use_decimal_type else np.float64(0))
                if bool_use_half_of_boundary_pmf :
                    sf += half_pmf

    # elif k <= (n - k) :
    elif bool_true_cdf_false_sf_none_min :
        # Compute PMF(i=0)
        i_first = max(0, n + N - M)
        if bool_numerical_approximation :
            pmf = approx_hypergeom_pmf_inner(
                k=i_first, M=M, n=n, N=N,
                bool_split_into_coeff_and_base_ten_exponent = False,
                bool_use_decimal_type = \
                    bool_use_decimal_type,
                int_decimal_computational_precision = \
                    int_decimal_computational_precision,
                int_decimal_reporting_precision = \
                    int_decimal_computational_precision,
                bool_approx_true_lanczos_false_spouge = \
                    bool_approx_true_lanczos_false_spouge,)
        else :
            lst_lst_num = [
                [1, max(1,n)],
                [1, max(1,N)],
                [1, max(1,M-N)],
                [1, max(1,M-n)],]
            lst_lst_den = [
                [1, max(1,n-i_first)],
                [1, max(1,N-i_first)],
                [1, max(1,M)],
                [1, max(1,M-N-n+i_first)],
                [1, max(1,i_first)],]
            (lst_lst_num, lst_lst_den) = _simplify_fraction_for_hypergeom_pmf(
                lst_lst_num, lst_lst_den)
            pmf = _compute_fraction_for_hypergeom_pmf(
                lst_lst_num = lst_lst_num,
                lst_lst_den = lst_lst_den,
                bool_split_into_coeff_and_base_ten_exponent = False,
                bool_use_decimal_type = \
                    bool_use_decimal_type,
                int_decimal_computational_precision = \
                    int_decimal_computational_precision,
                int_decimal_reporting_precision = \
                    int_decimal_computational_precision,)

        with localcontext() as ctx :
            # Perform a high precision calculation
            ctx.prec = int_decimal_computational_precision
            if bool_use_half_of_boundary_pmf and i_first+1 >= k+1 :
                pmf /= 2
            if i_first <= k :
                cdf = pmf
            else :
                cdf = (Decimal(0) if bool_use_decimal_type else np.float64(0))
            for i in range(i_first+1, k+1, +1) :
                #print("M={0:d}, N={1:d}, n={2:d}, k={3:d}, i={4:d}.".format(
                #    M,N,n,k,i))
                '''
                if bool_numerical_approximation :
                    pmf = approx_hypergeom_pmf_inner(
                        k=i, M=M, n=n, N=N,
                        bool_split_into_coeff_and_base_ten_exponent = False,
                        bool_use_decimal_type = \
                            bool_use_decimal_type,
                        int_decimal_computational_precision = \
                            int_decimal_computational_precision,
                        int_decimal_reporting_precision = \
                            int_decimal_computational_precision,
                        bool_approx_true_lanczos_false_spouge = \
                            bool_approx_true_lanczos_false_spouge,)
                else :
                '''
                # Numerical approximation is slower than incremental approach!
                # Use incremental approach even when bool_numerical_approximation is True
                if True :
                    pmf *= ((n - i + 1) * (N - i + 1))
                    pmf /= ((M - N - n + i) * (i))
                if bool_use_half_of_boundary_pmf and i == k :
                    pmf /= 2
                cdf += pmf
    # else : # k > (n - k)
    else : # not bool_true_cdf_false_sf_none_min
        if not bool_use_half_of_boundary_pmf and k == n :
            sf = (Decimal(0) if bool_use_decimal_type else np.float64(0))
        else :
            # Compute PMF(i=n)
            i_last = min(n, N)
            if bool_numerical_approximation :
                pmf = approx_hypergeom_pmf_inner(
                    k=i_last, M=M, n=n, N=N,
                    bool_split_into_coeff_and_base_ten_exponent = False,
                    bool_use_decimal_type = \
                        bool_use_decimal_type,
                    int_decimal_computational_precision = \
                        int_decimal_computational_precision,
                    int_decimal_reporting_precision = \
                        int_decimal_computational_precision,
                    bool_approx_true_lanczos_false_spouge = \
                        bool_approx_true_lanczos_false_spouge,)
            else :
                lst_lst_num = [
                    [1, max(1,n)],
                    [1, max(1,N)],
                    [1, max(1,M-N)],
                    [1, max(1,M-n)],]
                lst_lst_den = [
                    [1, max(1,n-i_last)],
                    [1, max(1,N-i_last)],
                    [1, max(1,M)],
                    [1, max(1,M-N-n+i_last)],
                    [1, max(1,i_last)],]
                (lst_lst_num, lst_lst_den) = _simplify_fraction_for_hypergeom_pmf(
                    lst_lst_num, lst_lst_den)
                pmf = _compute_fraction_for_hypergeom_pmf(
                    lst_lst_num = lst_lst_num,
                    lst_lst_den = lst_lst_den,
                    bool_split_into_coeff_and_base_ten_exponent = False,
                    bool_use_decimal_type = \
                        bool_use_decimal_type,
                    int_decimal_computational_precision = \
                        int_decimal_computational_precision,
                    int_decimal_reporting_precision = \
                        int_decimal_computational_precision,)

            with localcontext() as ctx :
                # Perform a high precision calculation
                ctx.prec = int_decimal_computational_precision
                if bool_use_half_of_boundary_pmf and i_last-1 <= k-1 :
                    pmf /= 2
                if i_last > k :
                    sf = pmf
                elif bool_use_half_of_boundary_pmf and i_last == k :
                    sf = pmf
                else :
                    sf = (Decimal(0) if bool_use_decimal_type else np.float64(0))
                for i in range(i_last-1, k-1, -1) :
                    if i > k or bool_use_half_of_boundary_pmf :
                        #print("M={0:d}, N={1:d}, n={2:d}, k={3:d}, i={4:d}.".format(
                        #    M,N,n,k,i))
                        '''
                        if bool_numerical_approximation :
                            pmf = approx_hypergeom_pmf_inner(
                                k=i, M=M, n=n, N=N,
                                bool_split_into_coeff_and_base_ten_exponent = False,
                                bool_use_decimal_type = \
                                    bool_use_decimal_type,
                                int_decimal_computational_precision = \
                                    int_decimal_computational_precision,
                                int_decimal_reporting_precision = \
                                    int_decimal_computational_precision,
                                bool_approx_true_lanczos_false_spouge = \
                                    bool_approx_true_lanczos_false_spouge,)
                        else :
                        '''
                        # Numerical approximation is slower than incremental approach!
                        # Use incremental approach even when bool_numerical_approximation is True
                        if True :
                            pmf *= ((M - N - n + i + 1) * (i + 1))
                            pmf /= ((n - i) * (N - i))
                        if bool_use_half_of_boundary_pmf and i == k :
                            pmf /= 2
                        sf += pmf

    getcontext().prec = int_decimal_reporting_precision
    # Round the final result back to the default precision
    if bool_true_cdf_false_sf_none_min :
        return_value = +cdf
    else :
        return_value = +sf
    if bool_use_decimal_type :
        return_value = return_value.min(Decimal(1)).max(Decimal(0))
    else :
        return_value = max(min(return_value, np.float64(1)), np.float64(0))
    if bool_split_into_coeff_and_base_ten_exponent :
        return split_float(number = return_value)
    else :
        return return_value


# https://www.i4cy.com/pi/
# Bailey Borwein Plouffe Algorithm
def approx_pi(digits) :
    getcontext().prec = digits + 2
    pi = Decimal(0)
    int_num_iterations = digits + 1
    one = Decimal(1)
    two = Decimal(2)
    four = Decimal(4)
    five = Decimal(5)
    six = Decimal(6)
    eight = Decimal(8)
    sixteen = Decimal(16)
    for k in range(int_num_iterations) :
        # Multiplier, M = 1 / 16^k
        M = one / (getcontext().power(sixteen, k))
        # Term 1, T1 = 4 / (8k + 1)
        T1 = four / (eight * k + one)
        # Term 2, T2 = 2 / (8k + 4)
        T2 = two / (eight * k + four)
        # Term 3, T3 = 1 / (8k + 5)
        T3 = one / (eight * k + five)
        # Term 4, T4 = 1 / (8k + 6)
        T4 = one / (eight * k + six)
        # Pi partial summation.
        pi = pi + M * (T1 - T2 - T3 - T4)
    getcontext().prec = digits
    pi = +pi
    return pi


# Inputs:
#
# k (k): number of successes in the draws without replacement
# n (n): number of trials/draws
# N (K): number of successes in the population
# M (N): population size
# bLeftSide: left side CDF or right side CDF.
#
# Return Value (approximated Hypergeometric CDF):
# If bLeftSide is true, then
#     Normal CDF(x <= k + 0.5; , mu = n * N / M, sigma =
#    (n * N * (M - N) / (M * M)) ^ 0.5):
#    probability of drawing at most "k" successes in n trials
# else
#     Normal CDF(x >= k - 0.5; , mu = n * N / M, sigma =
#    (n * N * (M - N) / (M * M)) ^ 0.5)
#    probability of drawing at least "k" successes in n trials
def _sum_taylor_normal_cdf_terms(
        k, M, n, N,
        bool_use_half_of_boundary_pmf = False,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = True,
        int_decimal_computational_precision = INT_DECIMAL_COMPUTATIONAL_PRECISION,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,
        bool_true_cdf_false_sf_none_min = True,
        bool_numerical_approximation = False,
        ) :
    if bool_use_decimal_type :
        with localcontext() as ctx :
            ctx.prec = int_decimal_computational_precision
            cdf = Decimal('nan')
            sf = Decimal('nan')
            #print(("M={0:d}, N={1:d}, n={2:d}, k={3:d}.").format(M,N,n,k))
            if 0 < M and 0 <= N <= M and 0 <= n <= M and 0 <= k <= N and \
               0 <= (n - k) <= (M - N) : # (0 <= k <= n) is implied from others
                zero = Decimal(0)
                one = Decimal(1)
                if (M == n or # => k == N
                    M == N or # => k == n
                    N == k or
                    n == 0 or # => k == 0
                    N == 0    # => k == 0
                    ) and not bool_use_half_of_boundary_pmf : 
                    cdf = one
                    sf = zero
                else :
                    two = Decimal(2)
                    three = Decimal(3)
                    four = Decimal(4)
                    pi = approx_pi(digits=int_decimal_computational_precision)
                    p = Decimal(N) / Decimal(M) # probability of success
                    q = one - p # probability of failure
                    mu = Decimal(n) * p
                    sigma = ctx.sqrt(Decimal(n) * p * q)
                    # Adj. to account for discrete nature of Hypergeom. distr.
                    half = one / two
                    k_adj = Decimal(k)
                    if bool_numerical_approximation :
                        if bool_use_half_of_boundary_pmf or k == 0 :
                            k_adj = Decimal(k)
                        else :
                            k_adj = Decimal(k) - half
                        if sigma != zero :
                            x = (k_adj - mu) / sigma
                        else :
                            x = zero
                        if bool_true_cdf_false_sf_none_min :
                            x = -x
                        ret_val = zero if (x > Decimal("36.5")) else ctx.exp(
                            -x * x / two) * (
                            ((((((Decimal("3.52624965998911e-02") * x + Decimal("0.700383064443688")) * x +
                                 Decimal("6.37396220353165")) * x + Decimal("33.912866078383")) * x +
                               Decimal("112.079291497871")) * x + Decimal("221.213596169931")) * x +
                             Decimal("220.206867912376")) /
                            (((((((Decimal("8.83883476483184e-02") * x + Decimal("1.75566716318264")) * x +
                                  Decimal("16.064177579207")) * x + Decimal("86.7807322029461")) * x +
                                Decimal("296.564248779674")) * x + Decimal("637.333633378831")) * x +
                              Decimal("793.826512519948")) * x + Decimal("440.413735824752"))
                            if (x < Decimal("7.07106781186546")) else one / (
                                ctx.sqrt(two * pi) * (
                                    x + one/(x + two/(x + three/(x + four/(
                                        x + Decimal(13) / Decimal(20))))))))
                        ret_val = min(one, max(zero, ret_val))
                        if bool_true_cdf_false_sf_none_min :
                            cdf = ret_val
                        else :
                            sf = ret_val
                    else :
                        if bool_use_half_of_boundary_pmf :
                            k_adj = Decimal(k)
                        else :
                            k_adj = Decimal(k) - half
                        z = (k_adj - mu) / sigma if sigma != zero else zero
                        err_toler = ctx.abs(ctx.power(
                            Decimal(10), -(min(
                                int_decimal_computational_precision,
                                int_decimal_reporting_precision)) / two))
                        if (z > Decimal(11)) : # 1 - eps < p < 1
                            if bool_true_cdf_false_sf_none_min :
                                cdf = one
                            else :
                                sf = zero
                        elif (z < Decimal(-11)) : # 0 < p < eps
                            if bool_true_cdf_false_sf_none_min :
                                cdf = zero
                            else :
                                sf = one
                        else :
                            # Initialize the integral of the Normal CDF Taylor Expansion.
                            integral = z if z >= zero else -z
                            curr_pow = two
                            cum_pow = -two
                            prev_coeff = -one
                            prev_adj = -one * ctx.power(ctx.abs(z), (zero + one))
                            # Add up integrated terms (polynomials) of the Tailor series
                            while ctx.abs(prev_adj) >= err_toler :
                                prev_coeff = one / (cum_pow * (curr_pow + one))
                                prev_adj = prev_coeff * ctx.power(
                                    ctx.abs(z), (curr_pow + one))
                                integral += prev_adj
                                curr_pow += two
                                cum_pow *= -curr_pow
                            adj_to_half = (one if z >= zero else -one) * (
                                one / ctx.sqrt(two * pi) * integral)
                            if bool_true_cdf_false_sf_none_min :
                                cdf = (half + adj_to_half).min(one).max(zero)
                            else :
                                sf = (half - adj_to_half).min(one).max(zero)
        getcontext().prec = int_decimal_reporting_precision
        # Round the final result back to the default precision
        if bool_true_cdf_false_sf_none_min :
            return_value = +cdf
        else :
            return_value = +sf
    else :
        cdf = np.float64('nan')
        sf = np.float64('nan')
        if 0 < M and 0 <= N <= M and 0 <= n <= M and 0 <= k <= N and \
           0 <= (n - k) <= (M - N) : # (0 <= k <= n) is implied from others
            zero = np.float64(0)
            one = np.float64(1)
            if (M == n or # => k == N
                M == N or # => k == n
                N == k or
                n == 0 or # => k == 0
                N == 0    # => k == 0
                ) and not bool_use_half_of_boundary_pmf : 
                cdf = one
                sf = zero
            else :
                two = np.float64(2)
                three = np.float64(3)
                four = np.float64(4)
                p = np.float64(N) / np.float64(M) # probability of success
                q = one - p # probability of failure
                mu = np.float64(n) * p
                sigma = np.sqrt(np.float64(n) * p * q)
                # Adj. to account for discrete nature of Hypergeom. distr.
                half = one / two
                if bool_use_half_of_boundary_pmf :
                    k_adj = np.float64(k)
                else :
                    k_adj = np.float64(k) - half
                if bool_numerical_approximation :
                    #x = np.fabs((k_adj - mu) / sigma) if sigma != zero else zero
                    if sigma != zero :
                        x = (k_adj - mu) / sigma
                    else :
                        x = zero
                    if bool_true_cdf_false_sf_none_min :
                        x = -x
                    ret_val = zero if (x > np.float64(36.5)) else np.exp(
                        -x * x / two) * (
                        ((((((np.float64(3.52624965998911e-02) * x + np.float64(0.700383064443688)) * x +
                             np.float64(6.37396220353165)) * x + np.float64(33.912866078383)) * x +
                           np.float64(112.079291497871)) * x + np.float64(221.213596169931)) * x +
                         np.float64(220.206867912376)) /
                        (((((((np.float64(8.83883476483184e-02) * x + np.float64(1.75566716318264)) * x +
                              np.float64(16.064177579207)) * x + np.float64(86.7807322029461)) * x +
                            np.float64(296.564248779674)) * x + np.float64(637.333633378831)) * x +
                          np.float64(793.826512519948)) * x + np.float64(440.413735824752))
                        if (x < np.float64(7.07106781186546)) else one / (
                            np.sqrt(two * np.pi) * (
                                x + one/(x + two/(x + three/(x + four/(
                                    x + np.float64(13) / np.float64(20))))))))
                    if bool_true_cdf_false_sf_none_min :
                        cdf = ret_val
                    else :
                        sf = ret_val
                else :
                    z = (k_adj - mu) / sigma if sigma != zero else zero
                    err_toler = np.fabs(np.pow(
                        np.float64(10), -(min(
                            int_decimal_computational_precision,
                            int_decimal_reporting_precision)) / two))
                    if (z > np.float64(6)) : # p > 0.999999998
                        cdf = one
                        sf = zero
                    elif (z < np.float64(-6)) : # p < 0.000000001
                        cdf = zero
                        sf = one
                    else :
                        # Initialize the integral of the Normal CDF Taylor Expansion.
                        integral = z if z >= zero else -z
                        curr_pow = two
                        cum_pow = -two
                        prev_coeff = -one
                        prev_adj = -one * np.pow(np.fabs(z), (zero + one))
                        # Add up integrated terms (polynomials) of the Tailor series
                        while np.fabs(prev_adj) >= err_toler :
                            prev_coeff = one / (cum_pow * (curr_pow + one))
                            prev_adj = prev_coeff * np.pow(
                                np.fabs(z), (curr_pow + one))
                            integral += prev_adj
                            curr_pow += two
                            cum_pow *= -curr_pow
                        adj_to_half = (one if z >= zero else -one) * (
                                one / np.sqrt(two * np.float64(
                                    3.141592653589793238462643383)) * \
                                    integral)
                        if bool_true_cdf_false_sf_none_min :
                            cdf = max(min(half + adj_to_half, one), zero)
                        else :
                            sf = max(min(half - adj_to_half, one), zero)
        if bool_true_cdf_false_sf_none_min :
            return_value = +cdf
        else :
            return_value = +sf

    if bool_split_into_coeff_and_base_ten_exponent :
        return_value = split_float(number = return_value)
    return return_value

###############################################################################
# !!!
# function type selector

INT_MAX_NUM_ITERS_FOR_EXACT_HYPERGEOM = 1_000_000 # 0 < _
INT_MAX_NUM_ITERS_FOR_LANCZOS_APPROX_HYPERGEOM = 500_000 # 0 < _
INT_MAX_NUM_ITERS_FOR_SPOUGE_APPROX_HYPERGEOM = 500_000 # 0 < _
INT_MIN_SAMPLE_SZ_FOR_APPROX_NORMAL = 1_000 # 0 <= _
FLT_MAX_SAMPLE_SZ_FRAC_OF_POP_SZ_FOR_APPROX_NORMAL = 0.1 # 0 <= _ <= 1
FLT_MAX_ABS_DIFF_POP_CATEG_FRACT_OF_POP_SIZE_TO_HALF_FOR_APPROX_NORMAL = 0.1 # 0 <= _ <= 0.5
FLT_MIN_NUM_STD_DEVS_FROM_MEAN_FOR_SAMPLE_CATEG_FOR_APPROX_NORMAL = 0. # 2.575829303549 # 99% in range mu +/- z*sigma

def choose_hypergeom_algorithm(
        k, M, n, N,
        int_max_num_iters_for_exact_hypergeom = \
            INT_MAX_NUM_ITERS_FOR_EXACT_HYPERGEOM,
        int_max_num_iters_for_lanczos_approx_hypergeom = \
            INT_MAX_NUM_ITERS_FOR_LANCZOS_APPROX_HYPERGEOM,
        int_max_num_iters_for_spouge_approx_hypergeom = \
            INT_MAX_NUM_ITERS_FOR_SPOUGE_APPROX_HYPERGEOM,
        int_min_sample_size_for_approx_normal = \
            INT_MIN_SAMPLE_SZ_FOR_APPROX_NORMAL,
        flt_max_sample_sz_frac_of_pop_sz_for_approx_normal = \
            FLT_MAX_SAMPLE_SZ_FRAC_OF_POP_SZ_FOR_APPROX_NORMAL,
        flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal = \
            FLT_MAX_ABS_DIFF_POP_CATEG_FRACT_OF_POP_SIZE_TO_HALF_FOR_APPROX_NORMAL,
        flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal = \
            FLT_MIN_NUM_STD_DEVS_FROM_MEAN_FOR_SAMPLE_CATEG_FOR_APPROX_NORMAL,
        bool_choose_for_pmf = False,
        bool_true_cdf_false_sf_none_min = None,
        ) :
    if bool_choose_for_pmf :
        # k + k + N + (N-k) + k = k + k + N + N;
        # (n-k) + (k) + (N) + (N-k) + (n-k) = n - k + n - k + N + N
        if (((k+k) + (N+N)) if k <= (n-k) else ((n-k+n-k) + (N+N))) <= \
           int_max_num_iters_for_exact_hypergeom :
            # Exact Hypergeometric
            int_enum_algorithm = 0
        elif (
                # _n = 13; 9 calls of log_gamma_lanczos_approx;
                # calling approx_hypergeom_pmf_lambda cost: (_n-1) * 9 = 12 * 9
                (13 - 1) * 9
             ) <= int_max_num_iters_for_lanczos_approx_hypergeom :
            # Lanczos Hypergeometric Approximation
            int_enum_algorithm = 1
        elif (
                # len(lst_c) = 20; 9 calls of log_gamma_spouge_approx;
                # calling approx_hypergeom_pmf_lambda cost: (len(lst_c)-1) * 9 = 19 * 9
                (20 - 1) * 9
             ) <= int_max_num_iters_for_spouge_approx_hypergeom :
            # Spouge Hypergeometric Approximation
            int_enum_algorithm = 2
        else :
            # Scipy Hypergeometric
            int_enum_algorithm = 4
    else :
        if (
                (min(N+N,n+n)+max(1,(k-1)*4)+min(M-N+M-N,n+n)+max(1,(n-k-1)*4)) # CDF and SF
                    if bool_true_cdf_false_sf_none_min is None else
                (
                    (min(N+N,n+n)+max(1,(k-1)*4)) # CDF = PMF(0) + sum(PMF(1), PMF(2), ...)
                        if bool_true_cdf_false_sf_none_min else
                    (min(M-N+M-N,n+n)+max(1,(n-k-1)*4))  # SF = PMF(k) + sum(PMF(k-1), PMF(k-2), ...)
                )
           ) <= int_max_num_iters_for_exact_hypergeom :
            # Exact Hypergeometric
            int_enum_algorithm = 0
        elif (
                # _n = 13; 9 log_gamma_lanczos_approx; pmf = approx_hypergeom_pmf_lambda: (_n-1) * 9 = 12 * 9
                # Compute PDF(0) or PDF(k-1) with simplified ratio of factorials
                (13 - 1) * 9 +
                # Other PDFs
                #max(1, (min(k, n-k) - 1) * (13 - 1) * 9) # use Lanczos approach for PMF
                max(1, (min(k, n-k) - 1) * 4) # use incremental approach for PMF
             ) <= int_max_num_iters_for_lanczos_approx_hypergeom :
            # Lanczos Hypergeometric Approximation
            int_enum_algorithm = 1
        elif (
                # len(lst_c) = 20; 9 log_gamma_spouge_approx; pmf = log_gamma_spouge_approx: (len(lst_c)-1) * 9 = 19 * 9
                # Compute PDF(0) or PDF(k-1) with simplified ratio of factorials
                (20 - 1) * 9 +
                # Other PDFs
                #max(1, (min(k, n-k) - 1) * (20 - 1) * 9) # use Spouge approach for PMF
                max(1, (min(k, n-k) - 1) * 4) # use incremental approach for PMF
             ) <= int_max_num_iters_for_spouge_approx_hypergeom :
            # Spouge Hypergeometric Approximation
            int_enum_algorithm = 2
        else :
            N_div_by_M = float(N) / float(M)
            mu = n * N_div_by_M
            sigma = math.sqrt(mu * (1. - N_div_by_M))
            if (
                 # n is large but small compared to M (and N, see skewness test next)
                 int_min_sample_size_for_approx_normal <= n <= \
                     flt_max_sample_sz_frac_of_pop_sz_for_approx_normal * M and
                 # Skewness close to zero.
                 (0.5 - flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal)
                     <= N / M <=
                 (0.5 + flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal) and
                 # k is in one of the two tails.
                 (k <= mu - flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal * sigma or \
                  k >= mu + flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal * sigma)
               ) :
                # Normal Taylor Expansion Approximation of Hypergeometric
                int_enum_algorithm = 3
            else :
                # Scipy Hypergeometric
                int_enum_algorithm = 4
    return int_enum_algorithm

###############################################################################
# !!!
# pmf

def custom_hypergeom_pmf(
        k, M, n, N,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = True,
        int_decimal_computational_precision = INT_DECIMAL_COMPUTATIONAL_PRECISION,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,
        int_max_num_iters_for_exact_hypergeom = \
            INT_MAX_NUM_ITERS_FOR_EXACT_HYPERGEOM,
        int_max_num_iters_for_lanczos_approx_hypergeom = \
            INT_MAX_NUM_ITERS_FOR_LANCZOS_APPROX_HYPERGEOM,
        int_max_num_iters_for_spouge_approx_hypergeom = \
            INT_MAX_NUM_ITERS_FOR_SPOUGE_APPROX_HYPERGEOM,
        int_min_sample_size_for_approx_normal = \
            INT_MIN_SAMPLE_SZ_FOR_APPROX_NORMAL,
        flt_max_sample_sz_frac_of_pop_sz_for_approx_normal = \
            FLT_MAX_SAMPLE_SZ_FRAC_OF_POP_SZ_FOR_APPROX_NORMAL,
        flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal = \
            FLT_MAX_ABS_DIFF_POP_CATEG_FRACT_OF_POP_SIZE_TO_HALF_FOR_APPROX_NORMAL,
        flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal = \
            FLT_MIN_NUM_STD_DEVS_FROM_MEAN_FOR_SAMPLE_CATEG_FOR_APPROX_NORMAL,
        ) :
    int_enum_algorithm = choose_hypergeom_algorithm(
        k=k, M=M, n=n, N=N,
        int_max_num_iters_for_exact_hypergeom = \
            int_max_num_iters_for_exact_hypergeom,
        int_max_num_iters_for_lanczos_approx_hypergeom = \
            int_max_num_iters_for_lanczos_approx_hypergeom,
        int_max_num_iters_for_spouge_approx_hypergeom = \
            int_max_num_iters_for_spouge_approx_hypergeom,
        int_min_sample_size_for_approx_normal = \
            int_min_sample_size_for_approx_normal,
        flt_max_sample_sz_frac_of_pop_sz_for_approx_normal = \
            flt_max_sample_sz_frac_of_pop_sz_for_approx_normal,
        flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal = \
            flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal,
        flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal = \
            flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal,
        bool_choose_for_pmf = True,
        bool_true_cdf_false_sf_none_min = None,)
    if int_enum_algorithm == 0 :
        return exact_hypergeom_pmf(
            k=k, M=M, n=n, N=N,
            bool_split_into_coeff_and_base_ten_exponent = \
                bool_split_into_coeff_and_base_ten_exponent,
            bool_use_decimal_type = bool_use_decimal_type,
            int_decimal_computational_precision = \
                int_decimal_computational_precision,
            int_decimal_reporting_precision = \
                int_decimal_reporting_precision,)
    elif int_enum_algorithm == 1 :
        return approx_lanczos_hypergeom_pmf(
            k=k, M=M, n=n, N=N,
            bool_split_into_coeff_and_base_ten_exponent = \
                bool_split_into_coeff_and_base_ten_exponent,
            bool_use_decimal_type = bool_use_decimal_type,
            int_decimal_computational_precision = \
                int_decimal_computational_precision,
            int_decimal_reporting_precision = \
                int_decimal_reporting_precision,)
    elif int_enum_algorithm == 2 :
        return approx_spouge_hypergeom_pmf(
            k=k, M=M, n=n, N=N,
            bool_split_into_coeff_and_base_ten_exponent = \
                bool_split_into_coeff_and_base_ten_exponent,
            bool_use_decimal_type = bool_use_decimal_type,
            int_decimal_computational_precision = \
                int_decimal_computational_precision,
            int_decimal_reporting_precision = \
                int_decimal_reporting_precision,)
    elif int_enum_algorithm == 4 :
        return scipy_hypergeom_pmf(
            k=k, M=M, n=n, N=N,
            bool_split_into_coeff_and_base_ten_exponent = \
                bool_split_into_coeff_and_base_ten_exponent,
            bool_use_decimal_type = bool_use_decimal_type,
            int_decimal_reporting_precision = \
                int_decimal_reporting_precision,)
    else :
        return None


def scipy_hypergeom_pmf(
        k, M, n, N,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = False,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,) :
    pmf = hypergeom.pmf(k=k, M=M, n=n, N=N)
    if bool_use_decimal_type :
        with localcontext() as ctx :
            ctx.prec = int_decimal_reporting_precision
            # Round the final result back to the reporting precision
            pmf = +Decimal(pmf)
    if bool_split_into_coeff_and_base_ten_exponent :
        pmf = split_float(number = pmf)
    return pmf


def exact_hypergeom_pmf(
        k, M, n, N,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = True,
        int_decimal_computational_precision = INT_DECIMAL_COMPUTATIONAL_PRECISION,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,) :
    lst_lst_num = [
        [1, max(1,n)],
        [1, max(1,N)],
        [1, max(1,M-N)],
        [1, max(1,M-n)],]
    lst_lst_den = [
        [1, max(1,n-k)],
        [1, max(1,N-k)],
        [1, max(1,M)],
        [1, max(1,M-N-n+k)],
        [1, max(1,k)],]
    (lst_lst_num, lst_lst_den) = _simplify_fraction_for_hypergeom_pmf(
        lst_lst_num, lst_lst_den)
    return _compute_fraction_for_hypergeom_pmf(
        lst_lst_num = lst_lst_num,
        lst_lst_den = lst_lst_den,
        bool_split_into_coeff_and_base_ten_exponent = \
            bool_split_into_coeff_and_base_ten_exponent,
        bool_use_decimal_type = \
            bool_use_decimal_type,
        int_decimal_computational_precision = \
            int_decimal_computational_precision,
        int_decimal_reporting_precision = \
            int_decimal_reporting_precision,)


def approx_lanczos_hypergeom_pmf(
        k, M, n, N,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = True,
        int_decimal_computational_precision = INT_DECIMAL_COMPUTATIONAL_PRECISION,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION) :
    if bool_use_decimal_type :
        with localcontext() as ctx :
            ctx.prec = int_decimal_computational_precision
            # http://www.vttoth.com/CMS/projects/41
            # https://beta.boost.org/doc/libs/1_48_0/libs/math/doc/sf_and_dist/html/math_toolkit/backgrounders/lanczos.html
            # http://mtweb.cs.ucl.ac.uk/mus/arabidopsis/xiang/software/boost_1_47_0/libs/math/doc/sf_and_dist/html/math_toolkit/backgrounders/lanczos.html
            _n = 13
            _g = 6.02468
            g = Decimal(format(_g, ".15g"))
            lst_p = [
                Decimal('2.50662827463100027016561693547825754072337540757200082367'),
                Decimal('589.5105778528748081083440754114017970881744165562870628551'),
                Decimal('-888.02534533501237172652316346971976076979770518225084992'),
                Decimal('395.838757159176115722783354674181284804742524064707536'),
                Decimal('-53.21395413703462595543160513282731562408231287715217'),
                Decimal('1.2771826424117897170129599091132309034574601465637'),
                Decimal('-0.0004046170655169348179547621938348030520821860594'),
                Decimal('-0.000007347585209589689589422864286753037601670817'),
                Decimal('0.000008208805239871217130461324758555114442513669'),
                Decimal('-0.000005159542415359044989159951746415314096492689'),
                Decimal('0.000002319630748531474375016814467281014423862250'),
                Decimal('-6.67124339402896748175608182929918031465333E-7'),
                Decimal('9.06038883356544784242812502552213897255901E-8')]
            def log_gamma_lanczos_approx(z, _n, g, lst_p) :
                r = lst_p[0]
                for i in range(1,_n) :
                    r += lst_p[i] / (z+i)
                half = Decimal(1) / Decimal(2)
                t = Decimal(z) + g + half
                log_g = ctx.ln(r) + (Decimal(z) + half) * ctx.ln(t) - t
                return log_g
            approx_hypergeom_pmf_lambda = lambda k, M, n, N, _n, g, lst_p : ctx.exp(
                -log_gamma_lanczos_approx(z=k, _n=_n, g=g, lst_p=lst_p)
                +log_gamma_lanczos_approx(z=M-N, _n=_n, g=g, lst_p=lst_p)
                -log_gamma_lanczos_approx(z=n-k, _n=_n, g=g, lst_p=lst_p)
                +log_gamma_lanczos_approx(z=N, _n=_n, g=g, lst_p=lst_p)
                -log_gamma_lanczos_approx(z=N-k, _n=_n, g=g, lst_p=lst_p)
                +log_gamma_lanczos_approx(z=n, _n=_n, g=g, lst_p=lst_p)
                -log_gamma_lanczos_approx(z=M, _n=_n, g=g, lst_p=lst_p)
                +log_gamma_lanczos_approx(z=M-n, _n=_n, g=g, lst_p=lst_p)
                -log_gamma_lanczos_approx(z=M-N-n+k, _n=_n, g=g, lst_p=lst_p))
            pmf = approx_hypergeom_pmf_lambda(
                k=k, M=M, n=n, N=N, _n=_n, g=g, lst_p=lst_p)
        getcontext().prec = int_decimal_reporting_precision
        pmf = +pmf # Round the final result back to the default precision
    else :
        _n = 13
        _g = 6.02468
        g = np.float64(_g)
        lst_p = [
            np.float64('2.50662827463100027016561693547825754072337540757200082367'),
            np.float64('589.5105778528748081083440754114017970881744165562870628551'),
            np.float64('-888.02534533501237172652316346971976076979770518225084992'),
            np.float64('395.838757159176115722783354674181284804742524064707536'),
            np.float64('-53.21395413703462595543160513282731562408231287715217'),
            np.float64('1.2771826424117897170129599091132309034574601465637'),
            np.float64('-0.0004046170655169348179547621938348030520821860594'),
            np.float64('-0.000007347585209589689589422864286753037601670817'),
            np.float64('0.000008208805239871217130461324758555114442513669'),
            np.float64('-0.000005159542415359044989159951746415314096492689'),
            np.float64('0.000002319630748531474375016814467281014423862250'),
            np.float64('-6.67124339402896748175608182929918031465333E-7'),
            np.float64('9.06038883356544784242812502552213897255901E-8')]
        def log_gamma_lanczos_approx(z, _n, g, lst_p) :
            r = lst_p[0]
            for i in range(1,_n) :
                r += lst_p[i] / (z+i)
            half = np.float64(.5)
            t = np.float64(z) + g + half
            log_g = np.log(r) + (np.float64(z) + half) * np.log(t) - t
            return log_g
        approx_hypergeom_pmf_lambda = lambda k, M, n, N, _n, g, lst_p : np.exp(
            -log_gamma_lanczos_approx(z=k, _n=_n, g=g, lst_p=lst_p)
            +log_gamma_lanczos_approx(z=M-N, _n=_n, g=g, lst_p=lst_p)
            -log_gamma_lanczos_approx(z=n-k, _n=_n, g=g, lst_p=lst_p)
            +log_gamma_lanczos_approx(z=N, _n=_n, g=g, lst_p=lst_p)
            -log_gamma_lanczos_approx(z=N-k, _n=_n, g=g, lst_p=lst_p)
            +log_gamma_lanczos_approx(z=n, _n=_n, g=g, lst_p=lst_p)
            -log_gamma_lanczos_approx(z=M, _n=_n, g=g, lst_p=lst_p)
            +log_gamma_lanczos_approx(z=M-n, _n=_n, g=g, lst_p=lst_p)
            -log_gamma_lanczos_approx(z=M-N-n+k, _n=_n, g=g, lst_p=lst_p))
        pmf = approx_hypergeom_pmf_lambda(
            k=k, M=M, n=n, N=N, _n=_n, g=g, lst_p=lst_p)

    if bool_split_into_coeff_and_base_ten_exponent :
        return split_float(number = pmf)
    else :
        return pmf


def approx_spouge_hypergeom_pmf(
        k, M, n, N,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = True,
        int_decimal_computational_precision = INT_DECIMAL_COMPUTATIONAL_PRECISION,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION) :
    # https://en.wikipedia.org/wiki/Spouge%27s_approximation
    if bool_use_decimal_type :
        with localcontext() as ctx :
            ctx.prec = int_decimal_computational_precision
            lst_c = [
                Decimal('2.506628274631000502415765284811045253006986740609938316629923576'),
                Decimal('777986313.1091454928811402005400108162295251348113128426334961852'),
                Decimal('-5014289818.386629570797576541967490103652258514832569674279751820'),
                Decimal('14391249419.00289373155156170253309810152458462806088822279545592'),
                Decimal('-24265005794.6668308801208968281605470912586269268831951251701441'),
                Decimal('26706480135.5683394082012008772502119322670635833810445328827937'),
                Decimal('-20167204252.3651678703249776522516454209876558668604305771976457'),
                Decimal('10693689215.66914860975029079126690660069469100630653881202448081'),
                Decimal('-4008321901.881434750140706868171689145285101201137451110548606052'),
                Decimal('1055739088.101602099423513581371922881638807456278410924429883147'),
                Decimal('-191947202.1477629664251405783223503351353012277209941617708328718'),
                Decimal('23357892.39307511798087638904470551337994322140297269819420234213'),
                Decimal('-1814408.148359252731818439838993707677526340125003219834376779638'),
                Decimal('83839.73741140680219453398758050281620165654670033425860998174394'),
                Decimal('-2072.661857059051124169469319593967128917693485708949680126818223'),
                Decimal('23.23427473123225253219821352451089625565552416728329648391674089'),
                Decimal('-0.08966195046443543309219652814526417878744371424147365192636459614'),
                Decimal('0.00007157552707474602995541065998770807262371282572284672472244648859'),
                Decimal('-3.850750431615709437581025834192410706598703829407022527230216420E-9'),
                Decimal('4.245740647764882878237117834434502631724174404640504624072297716E-16')]
            log_gamma_spouge_approx = lambda z, lst_c : (
                ctx.ln(z+len(lst_c)) * (z+Decimal(1)/Decimal(2)) +
                (-z-len(lst_c)) +
                ctx.ln(sum([lst_c[0]] + [
                    lst_c[k] / (z+k) for k in range(1,len(lst_c))])))
            approx_hypergeom_pmf_lambda = lambda k, M, n, N, lst_c : ctx.exp(
                -log_gamma_spouge_approx(k, lst_c)
                +log_gamma_spouge_approx(M-N, lst_c)
                -log_gamma_spouge_approx(n-k, lst_c)
                +log_gamma_spouge_approx(N, lst_c)
                -log_gamma_spouge_approx(N-k, lst_c)
                +log_gamma_spouge_approx(n, lst_c)
                -log_gamma_spouge_approx(M, lst_c)
                +log_gamma_spouge_approx(M-n, lst_c)
                -log_gamma_spouge_approx(M-N-n+k, lst_c))
            pmf = approx_hypergeom_pmf_lambda(k=k, M=M, n=n, N=N, lst_c=lst_c)
        getcontext().prec = int_decimal_reporting_precision
        pmf = +pmf # Round the final result back to the default precision
    else :
        lst_c = [
            np.float64('2.506628274631000502415765284811045253006986740609938316629923576'),
            np.float64('777986313.1091454928811402005400108162295251348113128426334961852'),
            np.float64('-5014289818.386629570797576541967490103652258514832569674279751820'),
            np.float64('14391249419.00289373155156170253309810152458462806088822279545592'),
            np.float64('-24265005794.6668308801208968281605470912586269268831951251701441'),
            np.float64('26706480135.5683394082012008772502119322670635833810445328827937'),
            np.float64('-20167204252.3651678703249776522516454209876558668604305771976457'),
            np.float64('10693689215.66914860975029079126690660069469100630653881202448081'),
            np.float64('-4008321901.881434750140706868171689145285101201137451110548606052'),
            np.float64('1055739088.101602099423513581371922881638807456278410924429883147'),
            np.float64('-191947202.1477629664251405783223503351353012277209941617708328718'),
            np.float64('23357892.39307511798087638904470551337994322140297269819420234213'),
            np.float64('-1814408.148359252731818439838993707677526340125003219834376779638'),
            np.float64('83839.73741140680219453398758050281620165654670033425860998174394'),
            np.float64('-2072.661857059051124169469319593967128917693485708949680126818223'),
            np.float64('23.23427473123225253219821352451089625565552416728329648391674089'),
            np.float64('-0.08966195046443543309219652814526417878744371424147365192636459614'),
            np.float64('0.00007157552707474602995541065998770807262371282572284672472244648859'),
            np.float64('-3.850750431615709437581025834192410706598703829407022527230216420E-9'),
            np.float64('4.245740647764882878237117834434502631724174404640504624072297716E-16')]
        log_gamma_spouge_approx = lambda z, lst_c : (
            np.log(z+len(lst_c)) * (z+Decimal(1)/Decimal(2)) +
            (-z-len(lst_c)) +
            np.log(sum([lst_c[0]] + [
                lst_c[k] / (z+k) for k in range(1,len(lst_c))])))
        approx_hypergeom_pmf_lambda = lambda k, M, n, N, lst_c : np.exp(
            -log_gamma_spouge_approx(k, lst_c)
            +log_gamma_spouge_approx(M-N, lst_c)
            -log_gamma_spouge_approx(n-k, lst_c)
            +log_gamma_spouge_approx(N, lst_c)
            -log_gamma_spouge_approx(N-k, lst_c)
            +log_gamma_spouge_approx(n, lst_c)
            -log_gamma_spouge_approx(M, lst_c)
            +log_gamma_spouge_approx(M-n, lst_c)
            -log_gamma_spouge_approx(M-N-n+k, lst_c))
        pmf = approx_hypergeom_pmf_lambda(k=k, M=M, n=n, N=N, lst_c=lst_c)

    if bool_split_into_coeff_and_base_ten_exponent :
        return split_float(number = pmf)
    else :
        return pmf


def normal_scipy_hypergeom_pmf(
        k, M, n, N,
        bool_split_into_coeff_and_base_ten_exponent = False,) :
    pdf = np.float64('nan')
    if 0 < M and 0 <= N <= M and 0 <= n <= M and 0 <= k <= N and \
       0 <= (n - k) <= (M - N) : # (0 <= k <= n) is implied from others
        p = np.float64(N) / np.float64(M) # probability of success
        q = np.float64(1.) - p # probability of failure
        mu = np.float64(n) * p
        sigma = np.sqrt(np.float64(n) * p * q)
        # Adj. to account for discrete nature of Hypergeom. distr.
        k_adj = np.float64(k)
        z = (k_adj - mu) / sigma if sigma != np.float64(0.) else np.float64(0.)
        pdf = norm.pdf(x=z, loc=np.float64(0.), scale=np.float64(1.))
    if bool_split_into_coeff_and_base_ten_exponent :
        pdf = split_float(number = pdf)
    return pdf


def normal_approx_hypergeom_pmf(
        k, M, n, N,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = True,
        int_decimal_computational_precision = INT_DECIMAL_COMPUTATIONAL_PRECISION,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION) :
    if bool_use_decimal_type :
        with localcontext() as ctx :
            ctx.prec = int_decimal_computational_precision
            zero = Decimal(0)
            one = Decimal(1)
            two = Decimal(2)
            pi = approx_pi(digits=int_decimal_computational_precision)
            p = Decimal(N) / Decimal(M) # probability of success
            q = one - p # probability of failure
            mu = Decimal(n) * p
            sigma = ctx.sqrt(Decimal(n) * p * q)
            # Adj. to account for discrete nature of Hypergeom. distr.
            k_adj = Decimal(k)
            z = (k_adj - mu) / sigma if sigma != zero else zero
            ret_val = ctx.exp(-z * z / two) / ctx.sqrt(two * pi)
        getcontext().prec = int_decimal_reporting_precision
        ret_val = +ret_val # Round the final result back to the default precision
    else :
        zero = np.float64(0)
        one = np.float64(1)
        two = np.float64(2)
        pi = np.float64(3.141592653589793238462643383)
        p = np.float64(N) / np.float64(M) # probability of success
        q = one - p # probability of failure
        mu = np.float64(n) * p
        sigma = np.sqrt(np.float64(n) * p * q)
        # Adj. to account for discrete nature of Hypergeom. distr.
        k_adj = np.float64(k)
        z = (k_adj - mu) / sigma if sigma != zero else zero
        ret_val = np.exp(-z * z / two) / np.sqrt(two * pi)
    if bool_split_into_coeff_and_base_ten_exponent :
        ret_val = split_float(number = ret_val)
    return ret_val

###############################################################################
# !!!
# cdf

def custom_hypergeom_cdf(
        k, M, n, N,
        bool_use_half_of_boundary_pmf = False,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = True,
        int_decimal_computational_precision = INT_DECIMAL_COMPUTATIONAL_PRECISION,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,
        int_max_num_iters_for_exact_hypergeom = \
            INT_MAX_NUM_ITERS_FOR_EXACT_HYPERGEOM,
        int_max_num_iters_for_lanczos_approx_hypergeom = \
            INT_MAX_NUM_ITERS_FOR_LANCZOS_APPROX_HYPERGEOM,
        int_max_num_iters_for_spouge_approx_hypergeom = \
            INT_MAX_NUM_ITERS_FOR_SPOUGE_APPROX_HYPERGEOM,
        int_min_sample_size_for_approx_normal = \
            INT_MIN_SAMPLE_SZ_FOR_APPROX_NORMAL,
        flt_max_sample_sz_frac_of_pop_sz_for_approx_normal = \
            FLT_MAX_SAMPLE_SZ_FRAC_OF_POP_SZ_FOR_APPROX_NORMAL,
        flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal = \
            FLT_MAX_ABS_DIFF_POP_CATEG_FRACT_OF_POP_SIZE_TO_HALF_FOR_APPROX_NORMAL,
        flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal = \
            FLT_MIN_NUM_STD_DEVS_FROM_MEAN_FOR_SAMPLE_CATEG_FOR_APPROX_NORMAL,
        ) :
    int_enum_algorithm = choose_hypergeom_algorithm(
        k=k, M=M, n=n, N=N,
        int_max_num_iters_for_exact_hypergeom = \
            int_max_num_iters_for_exact_hypergeom,
        int_max_num_iters_for_lanczos_approx_hypergeom = \
            int_max_num_iters_for_lanczos_approx_hypergeom,
        int_max_num_iters_for_spouge_approx_hypergeom = \
            int_max_num_iters_for_spouge_approx_hypergeom,
        int_min_sample_size_for_approx_normal = \
            int_min_sample_size_for_approx_normal,
        flt_max_sample_sz_frac_of_pop_sz_for_approx_normal = \
            flt_max_sample_sz_frac_of_pop_sz_for_approx_normal,
        flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal = \
            flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal,
        flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal = \
            flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal,
        bool_choose_for_pmf = False,
        bool_true_cdf_false_sf_none_min = True,)
    if int_enum_algorithm == 0 :
        return exact_hypergeom_cdf(
            k=k, M=M, n=n, N=N,
            bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
            bool_split_into_coeff_and_base_ten_exponent = \
                bool_split_into_coeff_and_base_ten_exponent,
            bool_use_decimal_type = bool_use_decimal_type,
            int_decimal_computational_precision = \
                int_decimal_computational_precision,
            int_decimal_reporting_precision = \
                int_decimal_reporting_precision,)
    elif int_enum_algorithm == 1 :
        return approx_lanczos_hypergeom_cdf(
            k=k, M=M, n=n, N=N,
            bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
            bool_split_into_coeff_and_base_ten_exponent = \
                bool_split_into_coeff_and_base_ten_exponent,
            bool_use_decimal_type = bool_use_decimal_type,
            int_decimal_computational_precision = \
                int_decimal_computational_precision,
            int_decimal_reporting_precision = \
                int_decimal_reporting_precision,)
    elif approx_spouge_hypergeom_cdf == 2 :
        return approx_lanczos_hypergeom_cdf(
            k=k, M=M, n=n, N=N,
            bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
            bool_split_into_coeff_and_base_ten_exponent = \
                bool_split_into_coeff_and_base_ten_exponent,
            bool_use_decimal_type = bool_use_decimal_type,
            int_decimal_computational_precision = \
                int_decimal_computational_precision,
            int_decimal_reporting_precision = \
                int_decimal_reporting_precision,)
    elif int_enum_algorithm == 3 :
        return normal_taylor_hypergeom_cdf(
            k=k, M=M, n=n, N=N,
            bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
            bool_split_into_coeff_and_base_ten_exponent = \
                bool_split_into_coeff_and_base_ten_exponent,
            bool_use_decimal_type = bool_use_decimal_type,
            int_decimal_computational_precision = \
                int_decimal_computational_precision,
            int_decimal_reporting_precision = \
                int_decimal_reporting_precision,)
    elif int_enum_algorithm == 4 :
        return scipy_hypergeom_cdf(
            k=k, M=M, n=n, N=N,
            bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
            bool_split_into_coeff_and_base_ten_exponent = \
                bool_split_into_coeff_and_base_ten_exponent,
            bool_use_decimal_type = bool_use_decimal_type,
            int_decimal_reporting_precision = \
                int_decimal_reporting_precision,)
    else :
        return None


def scipy_hypergeom_cdf(
        k, M, n, N,
        bool_use_half_of_boundary_pmf = False,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = False,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,) :
    cdf = hypergeom.cdf(k=k, M=M, n=n, N=N)
    if bool_use_half_of_boundary_pmf :
        cdf -= hypergeom.pmf(k=k, M=M, n=n, N=N) / 2.
    if bool_use_decimal_type :
        with localcontext() as ctx :
            ctx.prec = int_decimal_reporting_precision
            # Round the final result back to the reporting precision
            cdf = +Decimal(cdf)
    if bool_split_into_coeff_and_base_ten_exponent :
        cdf = split_float(number = cdf)
    return cdf


def exact_hypergeom_cdf(
        k, M, n, N,
        bool_use_half_of_boundary_pmf = False,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = True,
        int_decimal_computational_precision = INT_DECIMAL_COMPUTATIONAL_PRECISION,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,) :
    with localcontext() as ctx :
        # Perform a high precision calculation
        ctx.prec = int_decimal_computational_precision
        one = Decimal(1) if bool_use_decimal_type else np.float64(1.)
        half = Decimal(.5) if bool_use_decimal_type else np.float64(.5)
        # if k <= (n - k) : # Rough estimate of what is smaller: "cdf" or "sf"
        if min(N+N,n+n)+max(1,(k-1)*4) <= min(M-N+M-N,n+n)+max(1,(n-k-1)*4) :
            # Positive skewness means that the median is less than the mean.
            # min(N+N,n+n) <= min(M-N+M-N,n+n). See formula for skewness.
            cdf = _sum_hypergeom_pmfs(
                k=k, M=M, n=n, N=N,
                bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
                bool_split_into_coeff_and_base_ten_exponent = False,
                bool_use_decimal_type = bool_use_decimal_type,
                int_decimal_computational_precision = \
                    int_decimal_computational_precision,
                int_decimal_reporting_precision = \
                    int_decimal_reporting_precision,
                bool_true_cdf_false_sf_none_min = True,
                bool_numerical_approximation = False,
                bool_approx_true_lanczos_false_spouge = None,)
        #else : # if k > (n - k)
        else : # if min(N+N,n+n)+max(1,(k-1)*4) > min(M-N+M-N,n+n)+max(1,(n-k-1)*4) :
            # Negative skewness means that the median is greater than the mean.
            # min(N+N,n+n) > min(M-N+M-N,n+n). See formula for skewness.
            sf = _sum_hypergeom_pmfs(
                k=k, M=M, n=n, N=N,
                bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
                bool_split_into_coeff_and_base_ten_exponent = False,
                bool_use_decimal_type = bool_use_decimal_type,
                int_decimal_computational_precision = \
                    int_decimal_computational_precision,
                int_decimal_reporting_precision = \
                    int_decimal_reporting_precision,
                bool_true_cdf_false_sf_none_min = False,
                bool_numerical_approximation = False,
                bool_approx_true_lanczos_false_spouge = None,)
            if sf < half :
                cdf = one - sf # there is no loss in precision
            else : # Do NOT use "cdf = 1 - sf" due to precision loss!
                cdf = _sum_hypergeom_pmfs(
                    k=k, M=M, n=n, N=N,
                    bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
                    bool_split_into_coeff_and_base_ten_exponent = False,
                    bool_use_decimal_type = bool_use_decimal_type,
                    int_decimal_computational_precision = \
                        int_decimal_computational_precision,
                    int_decimal_reporting_precision = \
                        int_decimal_reporting_precision,
                    bool_true_cdf_false_sf_none_min = True,
                    bool_numerical_approximation = False,
                    bool_approx_true_lanczos_false_spouge = None,)
    if bool_use_decimal_type :
        with localcontext() as ctx :
            ctx.prec = int_decimal_reporting_precision
            # Round the final result back to the reporting precision
            cdf = +Decimal(cdf)
    if bool_split_into_coeff_and_base_ten_exponent :
        cdf = split_float(number = cdf)
    return cdf


def approx_lanczos_hypergeom_cdf(
        k, M, n, N,
        bool_use_half_of_boundary_pmf = False,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = True,
        int_decimal_computational_precision = INT_DECIMAL_COMPUTATIONAL_PRECISION,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,) :
    with localcontext() as ctx :
        # Perform a high precision calculation
        ctx.prec = int_decimal_computational_precision
        one = Decimal(1) if bool_use_decimal_type else np.float64(1.)
        half = Decimal(.5) if bool_use_decimal_type else np.float64(.5)
        if k <= (n - k) : # Rough estimate of what is smaller: "cdf" or "sf"
            cdf = _sum_hypergeom_pmfs(
                k=k, M=M, n=n, N=N,
                bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
                bool_split_into_coeff_and_base_ten_exponent = False,
                bool_use_decimal_type = bool_use_decimal_type,
                int_decimal_computational_precision = \
                    int_decimal_computational_precision,
                int_decimal_reporting_precision = \
                    int_decimal_reporting_precision,
                bool_true_cdf_false_sf_none_min = True,
                bool_numerical_approximation = True,
                bool_approx_true_lanczos_false_spouge = True,)
        else : # if k > (n - k)
            sf = _sum_hypergeom_pmfs(
                k=k, M=M, n=n, N=N,
                bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
                bool_split_into_coeff_and_base_ten_exponent = False,
                bool_use_decimal_type = bool_use_decimal_type,
                int_decimal_computational_precision = \
                    int_decimal_computational_precision,
                int_decimal_reporting_precision = \
                    int_decimal_reporting_precision,
                bool_true_cdf_false_sf_none_min = False,
                bool_numerical_approximation = True,
                bool_approx_true_lanczos_false_spouge = True,)
            if sf < half :
                cdf = one - sf # there is no loss in precision
            else : # Do NOT use "cdf = 1 - sf" due to precision loss!
                cdf = _sum_hypergeom_pmfs(
                    k=k, M=M, n=n, N=N,
                    bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
                    bool_split_into_coeff_and_base_ten_exponent = False,
                    bool_use_decimal_type = bool_use_decimal_type,
                    int_decimal_computational_precision = \
                        int_decimal_computational_precision,
                    int_decimal_reporting_precision = \
                        int_decimal_reporting_precision,
                    bool_true_cdf_false_sf_none_min = True,
                    bool_numerical_approximation = True,
                    bool_approx_true_lanczos_false_spouge = True,)
    if bool_use_decimal_type :
        with localcontext() as ctx :
            ctx.prec = int_decimal_reporting_precision
            # Round the final result back to the reporting precision
            cdf = +Decimal(cdf)
    if bool_split_into_coeff_and_base_ten_exponent :
        cdf = split_float(number = cdf)
    return cdf


def approx_spouge_hypergeom_cdf(
        k, M, n, N,
        bool_use_half_of_boundary_pmf = False,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = True,
        int_decimal_computational_precision = INT_DECIMAL_COMPUTATIONAL_PRECISION,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,) :
    with localcontext() as ctx :
        # Perform a high precision calculation
        ctx.prec = int_decimal_computational_precision
        one = Decimal(1) if bool_use_decimal_type else np.float64(1.)
        half = Decimal(.5) if bool_use_decimal_type else np.float64(.5)
        if k <= (n - k) : # Rough estimate of what is smaller: "cdf" or "sf"
            cdf = _sum_hypergeom_pmfs(
                k=k, M=M, n=n, N=N,
                bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
                bool_split_into_coeff_and_base_ten_exponent = False,
                bool_use_decimal_type = bool_use_decimal_type,
                int_decimal_computational_precision = \
                    int_decimal_computational_precision,
                int_decimal_reporting_precision = \
                    int_decimal_reporting_precision,
                bool_true_cdf_false_sf_none_min = True,
                bool_numerical_approximation = True,
                bool_approx_true_lanczos_false_spouge = False,)
        else : # if k > (n - k)
            sf = _sum_hypergeom_pmfs(
                k=k, M=M, n=n, N=N,
                bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
                bool_split_into_coeff_and_base_ten_exponent = False,
                bool_use_decimal_type = bool_use_decimal_type,
                int_decimal_computational_precision = \
                    int_decimal_computational_precision,
                int_decimal_reporting_precision = \
                    int_decimal_reporting_precision,
                bool_true_cdf_false_sf_none_min = False,
                bool_numerical_approximation = True,
                bool_approx_true_lanczos_false_spouge = False,)
            if sf < half :
                cdf = one - sf # there is no loss in precision
            else : # Do NOT use "cdf = 1 - sf" due to precision loss!
                cdf = _sum_hypergeom_pmfs(
                    k=k, M=M, n=n, N=N,
                    bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
                    bool_split_into_coeff_and_base_ten_exponent = False,
                    bool_use_decimal_type = bool_use_decimal_type,
                    int_decimal_computational_precision = \
                        int_decimal_computational_precision,
                    int_decimal_reporting_precision = \
                        int_decimal_reporting_precision,
                    bool_true_cdf_false_sf_none_min = True,
                    bool_numerical_approximation = True,
                    bool_approx_true_lanczos_false_spouge = False,)
    if bool_use_decimal_type :
        with localcontext() as ctx :
            ctx.prec = int_decimal_reporting_precision
            # Round the final result back to the reporting precision:
            cdf = +Decimal(cdf)
    if bool_split_into_coeff_and_base_ten_exponent :
        cdf = split_float(number = cdf)
    return cdf


def normal_scipy_hypergeom_cdf(
        k, M, n, N,
        bool_use_half_of_boundary_pmf = False,
        bool_split_into_coeff_and_base_ten_exponent = False,) :
    cdf = np.float64('nan')
    if 0 < M and 0 <= N <= M and 0 <= n <= M and 0 <= k <= N and \
       0 <= (n - k) <= (M - N) : # (0 <= k <= n) is implied from others
        if (M == n or # => k == N
            M == N or # => k == n
            N == k or
            n == 0 or # => k == 0
            N == 0    # => k == 0
            ) and not bool_use_half_of_boundary_pmf : 
            cdf = np.float64(1.)
        else :
            p = np.float64(N) / np.float64(M) # probability of success
            q = np.float64(1.) - p # probability of failure
            mu = np.float64(n) * p
            sigma = np.sqrt(np.float64(n) * p * q)
            # Adj. to account for discrete nature of Hypergeom. distr.
            if bool_use_half_of_boundary_pmf :
                k_adj = np.float64(k)
            else :
                k_adj = np.float64(k) - np.float64(.5)
            z = (k_adj - mu) / sigma if sigma != np.float64(0.) else np.float64(0.)
            cdf = norm.cdf(x=z, loc=np.float64(0.), scale=np.float64(1.))
    if bool_split_into_coeff_and_base_ten_exponent :
        cdf = split_float(number = cdf)
    return cdf


def normal_taylor_hypergeom_cdf(
        k, M, n, N,
        bool_use_half_of_boundary_pmf = False,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = True,
        int_decimal_computational_precision = INT_DECIMAL_COMPUTATIONAL_PRECISION,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,) :
    return _sum_taylor_normal_cdf_terms(
        k=k, M=M, n=n, N=N,
        bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
        bool_split_into_coeff_and_base_ten_exponent = \
            bool_split_into_coeff_and_base_ten_exponent,
        bool_use_decimal_type = bool_use_decimal_type,
        int_decimal_computational_precision = \
            int_decimal_computational_precision,
        int_decimal_reporting_precision = \
            int_decimal_reporting_precision,
        bool_true_cdf_false_sf_none_min = True,
        bool_numerical_approximation = False,)


def normal_approx_hypergeom_cdf(
        k, M, n, N,
        bool_use_half_of_boundary_pmf = False,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = True,
        int_decimal_computational_precision = INT_DECIMAL_COMPUTATIONAL_PRECISION,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,) :
    return _sum_taylor_normal_cdf_terms(
        k=k, M=M, n=n, N=N,
        bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
        bool_split_into_coeff_and_base_ten_exponent = \
            bool_split_into_coeff_and_base_ten_exponent,
        bool_use_decimal_type = bool_use_decimal_type,
        int_decimal_computational_precision = \
            int_decimal_computational_precision,
        int_decimal_reporting_precision = \
            int_decimal_reporting_precision,
        bool_true_cdf_false_sf_none_min = True,
        bool_numerical_approximation = True,)

###############################################################################
# !!!
# sf

def custom_hypergeom_sf(
        k, M, n, N,
        bool_use_half_of_boundary_pmf = False,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = True,
        int_decimal_computational_precision = INT_DECIMAL_COMPUTATIONAL_PRECISION,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,
        int_max_num_iters_for_exact_hypergeom = \
            INT_MAX_NUM_ITERS_FOR_EXACT_HYPERGEOM,
        int_max_num_iters_for_lanczos_approx_hypergeom = \
            INT_MAX_NUM_ITERS_FOR_LANCZOS_APPROX_HYPERGEOM,
        int_max_num_iters_for_spouge_approx_hypergeom = \
            INT_MAX_NUM_ITERS_FOR_SPOUGE_APPROX_HYPERGEOM,
        int_min_sample_size_for_approx_normal = \
            INT_MIN_SAMPLE_SZ_FOR_APPROX_NORMAL,
        flt_max_sample_sz_frac_of_pop_sz_for_approx_normal = \
            FLT_MAX_SAMPLE_SZ_FRAC_OF_POP_SZ_FOR_APPROX_NORMAL,
        flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal = \
            FLT_MAX_ABS_DIFF_POP_CATEG_FRACT_OF_POP_SIZE_TO_HALF_FOR_APPROX_NORMAL,
        flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal = \
            FLT_MIN_NUM_STD_DEVS_FROM_MEAN_FOR_SAMPLE_CATEG_FOR_APPROX_NORMAL,
        ) :
    int_enum_algorithm = choose_hypergeom_algorithm(
        k=k, M=M, n=n, N=N,
        int_max_num_iters_for_exact_hypergeom = \
            int_max_num_iters_for_exact_hypergeom,
        int_max_num_iters_for_lanczos_approx_hypergeom = \
            int_max_num_iters_for_lanczos_approx_hypergeom,
        int_max_num_iters_for_spouge_approx_hypergeom = \
            int_max_num_iters_for_spouge_approx_hypergeom,
        int_min_sample_size_for_approx_normal = \
            int_min_sample_size_for_approx_normal,
        flt_max_sample_sz_frac_of_pop_sz_for_approx_normal = \
            flt_max_sample_sz_frac_of_pop_sz_for_approx_normal,
        flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal = \
            flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal,
        flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal = \
            flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal,
        bool_choose_for_pmf = False,
        bool_true_cdf_false_sf_none_min = False,)
    if int_enum_algorithm == 0 :
        return exact_hypergeom_sf(
            k=k, M=M, n=n, N=N,
            bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
            bool_split_into_coeff_and_base_ten_exponent = \
                bool_split_into_coeff_and_base_ten_exponent,
            bool_use_decimal_type = bool_use_decimal_type,
            int_decimal_computational_precision = \
                int_decimal_computational_precision,
            int_decimal_reporting_precision = \
                int_decimal_reporting_precision,)
    elif int_enum_algorithm == 1 :
        return approx_lanczos_hypergeom_sf(
            k=k, M=M, n=n, N=N,
            bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
            bool_split_into_coeff_and_base_ten_exponent = \
                bool_split_into_coeff_and_base_ten_exponent,
            bool_use_decimal_type = bool_use_decimal_type,
            int_decimal_computational_precision = \
                int_decimal_computational_precision,
            int_decimal_reporting_precision = \
                int_decimal_reporting_precision,)
    elif int_enum_algorithm == 2 :
        return approx_spouge_hypergeom_sf(
            k=k, M=M, n=n, N=N,
            bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
            bool_split_into_coeff_and_base_ten_exponent = \
                bool_split_into_coeff_and_base_ten_exponent,
            bool_use_decimal_type = bool_use_decimal_type,
            int_decimal_computational_precision = \
                int_decimal_computational_precision,
            int_decimal_reporting_precision = \
                int_decimal_reporting_precision,)
    elif int_enum_algorithm == 3 :
        return normal_taylor_hypergeom_sf(
            k=k, M=M, n=n, N=N,
            bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
            bool_split_into_coeff_and_base_ten_exponent = \
                bool_split_into_coeff_and_base_ten_exponent,
            bool_use_decimal_type = bool_use_decimal_type,
            int_decimal_computational_precision = \
                int_decimal_computational_precision,
            int_decimal_reporting_precision = \
                int_decimal_reporting_precision,)
    elif int_enum_algorithm == 4 :
        return scipy_hypergeom_sf(
            k=k, M=M, n=n, N=N,
            bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
            bool_split_into_coeff_and_base_ten_exponent = \
                bool_split_into_coeff_and_base_ten_exponent,
            bool_use_decimal_type = bool_use_decimal_type,
            int_decimal_reporting_precision = \
                int_decimal_reporting_precision,)
    else :
        return None


def scipy_hypergeom_sf(
        k, M, n, N,
        bool_use_half_of_boundary_pmf = False,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = False,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,) :
    sf = hypergeom.sf(k=k, M=M, n=n, N=N)
    if bool_use_half_of_boundary_pmf :
        sf += hypergeom.pmf(k=k, M=M, n=n, N=N) / 2.
    if bool_use_decimal_type :
        with localcontext() as ctx :
            ctx.prec = int_decimal_reporting_precision
            # Round the final result back to the reporting precision:
            sf = +Decimal(sf)
    if bool_split_into_coeff_and_base_ten_exponent :
        sf = split_float(number = sf)
    return sf


def exact_hypergeom_sf(
        k, M, n, N,
        bool_use_half_of_boundary_pmf = False,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = True,
        int_decimal_computational_precision = INT_DECIMAL_COMPUTATIONAL_PRECISION,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,) :
    with localcontext() as ctx :
        # Perform a high precision calculation
        ctx.prec = int_decimal_computational_precision
        one = Decimal(1) if bool_use_decimal_type else np.float64(1.)
        half = Decimal(.5) if bool_use_decimal_type else np.float64(.5)
        # if k <= (n - k) : # Rough estimate of what is smaller: "cdf" or "sf"
        if min(N+N,n+n)+max(1,(k-1)*4) <= min(M-N+M-N,n+n)+max(1,(n-k-1)*4) :
            # Positive skewness means that the median is less than the mean.
            # min(N+N,n+n) <= min(M-N+M-N,n+n). See formula for skewness.
            cdf = _sum_hypergeom_pmfs(
                k=k, M=M, n=n, N=N,
                bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
                bool_split_into_coeff_and_base_ten_exponent = False,
                bool_use_decimal_type = bool_use_decimal_type,
                int_decimal_computational_precision = \
                    int_decimal_computational_precision,
                int_decimal_reporting_precision = \
                    int_decimal_reporting_precision,
                bool_true_cdf_false_sf_none_min = True,
                bool_numerical_approximation = False,
                bool_approx_true_lanczos_false_spouge = None,)
            if cdf < half :
                sf = one - cdf # there is no loss in precision
            else : # Do NOT use "sf = 1 - cdf" due to precision loss!
                sf = _sum_hypergeom_pmfs(
                    k=k, M=M, n=n, N=N,
                    bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
                    bool_split_into_coeff_and_base_ten_exponent = False,
                    bool_use_decimal_type = bool_use_decimal_type,
                    int_decimal_computational_precision = \
                        int_decimal_computational_precision,
                    int_decimal_reporting_precision = \
                        int_decimal_reporting_precision,
                    bool_true_cdf_false_sf_none_min = False,
                    bool_numerical_approximation = False,
                    bool_approx_true_lanczos_false_spouge = None,)
        #else : # if k > (n - k)
        else : # if min(N+N,n+n)+max(1,(k-1)*4) > min(M-N+M-N,n+n)+max(1,(n-k-1)*4) :
            # Negative skewness means that the median is greater than the mean.
            # min(N+N,n+n) > min(M-N+M-N,n+n). See formula for skewness.
            sf = _sum_hypergeom_pmfs(
                k=k, M=M, n=n, N=N,
                bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
                bool_split_into_coeff_and_base_ten_exponent = False,
                bool_use_decimal_type = bool_use_decimal_type,
                int_decimal_computational_precision = \
                    int_decimal_computational_precision,
                int_decimal_reporting_precision = \
                    int_decimal_reporting_precision,
                bool_true_cdf_false_sf_none_min = False,
                bool_numerical_approximation = False,
                bool_approx_true_lanczos_false_spouge = None,)
    if bool_use_decimal_type :
        with localcontext() as ctx :
            ctx.prec = int_decimal_reporting_precision
            # Round the final result back to the reporting precision:
            sf = +Decimal(sf)
    if bool_split_into_coeff_and_base_ten_exponent :
        sf = split_float(number = sf)
    return sf


def approx_lanczos_hypergeom_sf(
        k, M, n, N,
        bool_use_half_of_boundary_pmf = False,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = True,
        int_decimal_computational_precision = INT_DECIMAL_COMPUTATIONAL_PRECISION,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,) :
    with localcontext() as ctx :
        # Perform a high precision calculation
        ctx.prec = int_decimal_computational_precision
        one = Decimal(1) if bool_use_decimal_type else np.float64(1.)
        half = Decimal(.5) if bool_use_decimal_type else np.float64(.5)
        if k <= (n - k) : # Rough estimate of what is smaller: "cdf" or "sf"
            cdf = _sum_hypergeom_pmfs(
                k=k, M=M, n=n, N=N,
                bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
                bool_split_into_coeff_and_base_ten_exponent = False,
                bool_use_decimal_type = bool_use_decimal_type,
                int_decimal_computational_precision = \
                    int_decimal_computational_precision,
                int_decimal_reporting_precision = \
                    int_decimal_reporting_precision,
                bool_true_cdf_false_sf_none_min = True,
                bool_numerical_approximation = True,
                bool_approx_true_lanczos_false_spouge = True,)
            if cdf < half :
                sf = one - cdf # there is no loss in precision
            else : # Do NOT use "sf = 1 - cdf" due to precision loss!
                sf = _sum_hypergeom_pmfs(
                    k=k, M=M, n=n, N=N,
                    bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
                    bool_split_into_coeff_and_base_ten_exponent = False,
                    bool_use_decimal_type = bool_use_decimal_type,
                    int_decimal_computational_precision = \
                        int_decimal_computational_precision,
                    int_decimal_reporting_precision = \
                        int_decimal_reporting_precision,
                    bool_true_cdf_false_sf_none_min = False,
                    bool_numerical_approximation = True,
                    bool_approx_true_lanczos_false_spouge = True,)
        else : # if k > (n - k)
            sf = _sum_hypergeom_pmfs(
                k=k, M=M, n=n, N=N,
                bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
                bool_split_into_coeff_and_base_ten_exponent = False,
                bool_use_decimal_type = bool_use_decimal_type,
                int_decimal_computational_precision = \
                    int_decimal_computational_precision,
                int_decimal_reporting_precision = \
                    int_decimal_reporting_precision,
                bool_true_cdf_false_sf_none_min = False,
                bool_numerical_approximation = True,
                bool_approx_true_lanczos_false_spouge = True,)
    if bool_use_decimal_type :
        with localcontext() as ctx :
            ctx.prec = int_decimal_reporting_precision
            # Round the final result back to the reporting precision:
            sf = +Decimal(sf)
    if bool_split_into_coeff_and_base_ten_exponent :
        sf = split_float(number = sf)
    return sf


def approx_spouge_hypergeom_sf(
        k, M, n, N,
        bool_use_half_of_boundary_pmf = False,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = True,
        int_decimal_computational_precision = INT_DECIMAL_COMPUTATIONAL_PRECISION,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,) :
    with localcontext() as ctx :
        # Perform a high precision calculation
        ctx.prec = int_decimal_computational_precision
        one = Decimal(1) if bool_use_decimal_type else np.float64(1.)
        half = Decimal(.5) if bool_use_decimal_type else np.float64(.5)
        if k <= (n - k) : # Rough estimate of what is smaller: "cdf" or "sf"
            cdf = _sum_hypergeom_pmfs(
                k=k, M=M, n=n, N=N,
                bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
                bool_split_into_coeff_and_base_ten_exponent = False,
                bool_use_decimal_type = bool_use_decimal_type,
                int_decimal_computational_precision = \
                    int_decimal_computational_precision,
                int_decimal_reporting_precision = \
                    int_decimal_reporting_precision,
                bool_true_cdf_false_sf_none_min = True,
                bool_numerical_approximation = True,
                bool_approx_true_lanczos_false_spouge = False,)
            if cdf < half :
                sf = one - cdf # there is no loss in precision
            else : # Do NOT use "sf = 1 - cdf" due to precision loss!
                sf = _sum_hypergeom_pmfs(
                    k=k, M=M, n=n, N=N,
                    bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
                    bool_split_into_coeff_and_base_ten_exponent = False,
                    bool_use_decimal_type = bool_use_decimal_type,
                    int_decimal_computational_precision = \
                        int_decimal_computational_precision,
                    int_decimal_reporting_precision = \
                        int_decimal_reporting_precision,
                    bool_true_cdf_false_sf_none_min = False,
                    bool_numerical_approximation = True,
                    bool_approx_true_lanczos_false_spouge = False,)
        else : # if k > (n - k)
            sf = _sum_hypergeom_pmfs(
                k=k, M=M, n=n, N=N,
                bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
                bool_split_into_coeff_and_base_ten_exponent = False,
                bool_use_decimal_type = bool_use_decimal_type,
                int_decimal_computational_precision = \
                    int_decimal_computational_precision,
                int_decimal_reporting_precision = \
                    int_decimal_reporting_precision,
                bool_true_cdf_false_sf_none_min = False,
                bool_numerical_approximation = True,
                bool_approx_true_lanczos_false_spouge = False,)
    if bool_use_decimal_type :
        with localcontext() as ctx :
            ctx.prec = int_decimal_reporting_precision
            # Round the final result back to the reporting precision
            sf = +Decimal(sf)
    if bool_split_into_coeff_and_base_ten_exponent :
        sf = split_float(number = sf)
    return sf


def normal_scipy_hypergeom_sf(
        k, M, n, N,
        bool_use_half_of_boundary_pmf = False,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = True,
        int_decimal_computational_precision = INT_DECIMAL_COMPUTATIONAL_PRECISION,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,) :
    sf = np.float64('nan')
    if 0 < M and 0 <= N <= M and 0 <= n <= M and 0 <= k <= N and \
       0 <= (n - k) <= (M - N) : # (0 <= k <= n) is implied from others
        if (M == n or # => k == N
            M == N or # => k == n
            N == k or
            n == 0 or # => k == 0
            N == 0    # => k == 0
            ) and not bool_use_half_of_boundary_pmf : 
            sf = np.float64(0.)
        else :
            p = np.float64(N) / np.float64(M) # probability of success
            q = np.float64(1.) - p # probability of failure
            mu = np.float64(n) * p
            sigma = np.sqrt(np.float64(n) * p * q)
            # Adj. to account for discrete nature of Hypergeom. distr.
            if bool_use_half_of_boundary_pmf :
                k_adj = np.float64(k)
            else :
                k_adj = np.float64(k) - np.float64(.5)
            z = (k_adj - mu) / sigma if sigma != np.float64(0.) else np.float64(0.)
            sf = norm.sf(x=z, loc=np.float64(0.), scale=np.float64(1.))
    if bool_split_into_coeff_and_base_ten_exponent :
        sf = split_float(number = sf)
    return sf


def normal_taylor_hypergeom_sf(
        k, M, n, N,
        bool_use_half_of_boundary_pmf = False,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = True,
        int_decimal_computational_precision = INT_DECIMAL_COMPUTATIONAL_PRECISION,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,) :
    return _sum_taylor_normal_cdf_terms(
        k=k, M=M, n=n, N=N,
        bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
        bool_split_into_coeff_and_base_ten_exponent = \
            bool_split_into_coeff_and_base_ten_exponent,
        bool_use_decimal_type = bool_use_decimal_type,
        int_decimal_computational_precision = \
            int_decimal_computational_precision,
        int_decimal_reporting_precision = \
            int_decimal_reporting_precision,
        bool_true_cdf_false_sf_none_min = False,
        bool_numerical_approximation = False,)


def normal_approx_hypergeom_sf(
        k, M, n, N,
        bool_use_half_of_boundary_pmf = False,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = True,
        int_decimal_computational_precision = INT_DECIMAL_COMPUTATIONAL_PRECISION,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,) :
    return _sum_taylor_normal_cdf_terms(
        k=k, M=M, n=n, N=N,
        bool_use_half_of_boundary_pmf = bool_use_half_of_boundary_pmf,
        bool_split_into_coeff_and_base_ten_exponent = \
            bool_split_into_coeff_and_base_ten_exponent,
        bool_use_decimal_type = bool_use_decimal_type,
        int_decimal_computational_precision = \
            int_decimal_computational_precision,
        int_decimal_reporting_precision = \
            int_decimal_reporting_precision,
        bool_true_cdf_false_sf_none_min = False,
        bool_numerical_approximation = True,)

###############################################################################
# !!!
# min_cdf_sf

def custom_hypergeom_min_cdf_sf(
        k, M, n, N,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = True,
        int_decimal_computational_precision = INT_DECIMAL_COMPUTATIONAL_PRECISION,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,
        int_max_num_iters_for_exact_hypergeom = \
            INT_MAX_NUM_ITERS_FOR_EXACT_HYPERGEOM,
        int_max_num_iters_for_lanczos_approx_hypergeom = \
            INT_MAX_NUM_ITERS_FOR_LANCZOS_APPROX_HYPERGEOM,
        int_max_num_iters_for_spouge_approx_hypergeom = \
            INT_MAX_NUM_ITERS_FOR_SPOUGE_APPROX_HYPERGEOM,
        int_min_sample_size_for_approx_normal = \
            INT_MIN_SAMPLE_SZ_FOR_APPROX_NORMAL,
        flt_max_sample_sz_frac_of_pop_sz_for_approx_normal = \
            FLT_MAX_SAMPLE_SZ_FRAC_OF_POP_SZ_FOR_APPROX_NORMAL,
        flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal = \
            FLT_MAX_ABS_DIFF_POP_CATEG_FRACT_OF_POP_SIZE_TO_HALF_FOR_APPROX_NORMAL,
        flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal = \
            FLT_MIN_NUM_STD_DEVS_FROM_MEAN_FOR_SAMPLE_CATEG_FOR_APPROX_NORMAL,
        ) :
    int_enum_algorithm = choose_hypergeom_algorithm(
        k=k, M=M, n=n, N=N,
        int_max_num_iters_for_exact_hypergeom = \
            int_max_num_iters_for_exact_hypergeom,
        int_max_num_iters_for_lanczos_approx_hypergeom = \
            int_max_num_iters_for_lanczos_approx_hypergeom,
        int_max_num_iters_for_spouge_approx_hypergeom = \
            int_max_num_iters_for_spouge_approx_hypergeom,
        int_min_sample_size_for_approx_normal = \
            int_min_sample_size_for_approx_normal,
        flt_max_sample_sz_frac_of_pop_sz_for_approx_normal = \
            flt_max_sample_sz_frac_of_pop_sz_for_approx_normal,
        flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal = \
            flt_max_abs_diff_pop_categ_frac_of_pop_size_to_half_for_approx_normal,
        flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal = \
            flt_min_num_std_devs_from_mean_for_sample_categ_for_approx_normal,
        bool_choose_for_pmf = False,
        bool_true_cdf_false_sf_none_min = None,)
    if int_enum_algorithm == 0 :
        return exact_hypergeom_min_cdf_sf(
            k=k, M=M, n=n, N=N,
            bool_split_into_coeff_and_base_ten_exponent = \
                bool_split_into_coeff_and_base_ten_exponent,
            bool_use_decimal_type = bool_use_decimal_type,
            int_decimal_computational_precision = \
                int_decimal_computational_precision,
            int_decimal_reporting_precision = \
                int_decimal_reporting_precision,)
    elif int_enum_algorithm == 1 :
        return approx_lanczos_hypergeom_min_cdf_sf(
            k=k, M=M, n=n, N=N,
            bool_split_into_coeff_and_base_ten_exponent = \
                bool_split_into_coeff_and_base_ten_exponent,
            bool_use_decimal_type = bool_use_decimal_type,
            int_decimal_computational_precision = \
                int_decimal_computational_precision,
            int_decimal_reporting_precision = \
                int_decimal_reporting_precision,)
    elif int_enum_algorithm == 2 :
        return approx_spouge_hypergeom_min_cdf_sf(
            k=k, M=M, n=n, N=N,
            bool_split_into_coeff_and_base_ten_exponent = \
                bool_split_into_coeff_and_base_ten_exponent,
            bool_use_decimal_type = bool_use_decimal_type,
            int_decimal_computational_precision = \
                int_decimal_computational_precision,
            int_decimal_reporting_precision = \
                int_decimal_reporting_precision,)
    elif int_enum_algorithm == 3 :
        return normal_taylor_hypergeom_min_cdf_sf(
            k=k, M=M, n=n, N=N,
            bool_split_into_coeff_and_base_ten_exponent = \
                bool_split_into_coeff_and_base_ten_exponent,
            bool_use_decimal_type = bool_use_decimal_type,
            int_decimal_computational_precision = \
                int_decimal_computational_precision,
            int_decimal_reporting_precision = \
                int_decimal_reporting_precision,)
    elif int_enum_algorithm == 4 :
        return scipy_hypergeom_min_cdf_sf(
            k=k, M=M, n=n, N=N,
            bool_split_into_coeff_and_base_ten_exponent = \
                bool_split_into_coeff_and_base_ten_exponent,
            bool_use_decimal_type = bool_use_decimal_type,
            int_decimal_reporting_precision = \
                int_decimal_reporting_precision,)
    else :
        return None


def scipy_hypergeom_min_cdf_sf(
        k, M, n, N,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = False,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,) :
    half = Decimal(.5) if bool_use_decimal_type else np.float64(.5)
    cdf = scipy_hypergeom_cdf(
        k=k, M=M, n=n, N=N,
        bool_use_half_of_boundary_pmf = True,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = bool_use_decimal_type,
        int_decimal_reporting_precision = int_decimal_reporting_precision,
        )
    if cdf < half :
        min_cdf_sf = cdf
    else : # Do NOT use "min_cdf_sf = sf = 1 - cdf" due to precision loss!
        sf = scipy_hypergeom_sf(
            k=k, M=M, n=n, N=N,
            bool_use_half_of_boundary_pmf = True,
            bool_split_into_coeff_and_base_ten_exponent = False,
            bool_use_decimal_type = bool_use_decimal_type,
            int_decimal_reporting_precision = int_decimal_reporting_precision,
            )
        min_cdf_sf = min(cdf, sf)
    if bool_use_decimal_type :
        with localcontext() as ctx :
            ctx.prec = int_decimal_reporting_precision
            # Round the final result back to the reporting precision:
            min_cdf_sf = +Decimal(min_cdf_sf)
    if bool_split_into_coeff_and_base_ten_exponent :
        min_cdf_sf = split_float(number = min_cdf_sf)
    return min_cdf_sf


def exact_hypergeom_min_cdf_sf(
        k, M, n, N,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = True,
        int_decimal_computational_precision = INT_DECIMAL_COMPUTATIONAL_PRECISION,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,) :
    with localcontext() as ctx :
        # Perform a high precision calculation
        ctx.prec = int_decimal_computational_precision
        half = Decimal(.5) if bool_use_decimal_type else np.float64(.5)
        # if k <= (n - k) : # Rough estimate of what is smaller: "cdf" or "sf"
        if min(N+N,n+n)+max(1,(k-1)*4) <= min(M-N+M-N,n+n)+max(1,(n-k-1)*4) :
            # Positive skewness means that the median is less than the mean.
            # min(N+N,n+n) <= min(M-N+M-N,n+n). See formula for skewness.
            cdf = _sum_hypergeom_pmfs(
                k=k, M=M, n=n, N=N,
                bool_use_half_of_boundary_pmf = True,
                bool_split_into_coeff_and_base_ten_exponent = False,
                bool_use_decimal_type = bool_use_decimal_type,
                int_decimal_computational_precision = \
                    int_decimal_computational_precision,
                int_decimal_reporting_precision = \
                    int_decimal_reporting_precision,
                bool_true_cdf_false_sf_none_min = True,
                bool_numerical_approximation = False,
                bool_approx_true_lanczos_false_spouge = None,)
            if cdf < half :
                min_cdf_sf = cdf
            else : # Do NOT use "min_cdf_sf = sf = 1 - cdf" due to precision loss!
                sf = _sum_hypergeom_pmfs(
                    k=k, M=M, n=n, N=N,
                    bool_use_half_of_boundary_pmf = True,
                    bool_split_into_coeff_and_base_ten_exponent = False,
                    bool_use_decimal_type = bool_use_decimal_type,
                    int_decimal_computational_precision = \
                        int_decimal_computational_precision,
                    int_decimal_reporting_precision = \
                        int_decimal_reporting_precision,
                    bool_true_cdf_false_sf_none_min = False,
                    bool_numerical_approximation = False,
                    bool_approx_true_lanczos_false_spouge = None,)
                min_cdf_sf = min(cdf, sf)
        #else : # if k > (n - k)
        else : # if min(N+N,n+n)+max(1,(k-1)*4) > min(M-N+M-N,n+n)+max(1,(n-k-1)*4) :
            # Negative skewness means that the median is greater than the mean.
            # min(N+N,n+n) > min(M-N+M-N,n+n). See formula for skewness.
            sf = _sum_hypergeom_pmfs(
                k=k, M=M, n=n, N=N,
                bool_use_half_of_boundary_pmf = True,
                bool_split_into_coeff_and_base_ten_exponent = False,
                bool_use_decimal_type = bool_use_decimal_type,
                int_decimal_computational_precision = \
                    int_decimal_computational_precision,
                int_decimal_reporting_precision = \
                    int_decimal_reporting_precision,
                bool_true_cdf_false_sf_none_min = False,
                bool_numerical_approximation = False,
                bool_approx_true_lanczos_false_spouge = None,)
            if sf < half :
                min_cdf_sf = sf
            else : # Do NOT use "min_cdf_sf = cdf = 1 - sf" due to precision loss!
                cdf = _sum_hypergeom_pmfs(
                    k=k, M=M, n=n, N=N,
                    bool_use_half_of_boundary_pmf = True,
                    bool_split_into_coeff_and_base_ten_exponent = False,
                    bool_use_decimal_type = bool_use_decimal_type,
                    int_decimal_computational_precision = \
                        int_decimal_computational_precision,
                    int_decimal_reporting_precision = \
                        int_decimal_reporting_precision,
                    bool_true_cdf_false_sf_none_min = True,
                    bool_numerical_approximation = False,
                    bool_approx_true_lanczos_false_spouge = None,)
                min_cdf_sf = min(cdf, sf)
    if bool_use_decimal_type :
        with localcontext() as ctx :
            ctx.prec = int_decimal_reporting_precision
            # Round the final result back to the reporting precision:
            min_cdf_sf = +Decimal(min_cdf_sf)
    if bool_split_into_coeff_and_base_ten_exponent :
        min_cdf_sf = split_float(number = min_cdf_sf)
    return min_cdf_sf


def approx_lanczos_hypergeom_min_cdf_sf(
        k, M, n, N,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = True,
        int_decimal_computational_precision = INT_DECIMAL_COMPUTATIONAL_PRECISION,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,) :
    with localcontext() as ctx :
        # Perform a high precision calculation
        ctx.prec = int_decimal_computational_precision
        half = Decimal(.5) if bool_use_decimal_type else np.float64(.5)
        if k <= (n - k) : # Rough estimate of what is smaller: "cdf" or "sf"
            cdf = _sum_hypergeom_pmfs(
                k=k, M=M, n=n, N=N,
                bool_use_half_of_boundary_pmf = True,
                bool_split_into_coeff_and_base_ten_exponent = False,
                bool_use_decimal_type = bool_use_decimal_type,
                int_decimal_computational_precision = \
                    int_decimal_computational_precision,
                int_decimal_reporting_precision = \
                    int_decimal_reporting_precision,
                bool_true_cdf_false_sf_none_min = True,
                bool_numerical_approximation = True,
                bool_approx_true_lanczos_false_spouge = True,)
            if cdf < half :
                min_cdf_sf = cdf
            else : # Do NOT use "min_cdf_sf = sf = 1 - cdf" due to precision loss!
                sf = _sum_hypergeom_pmfs(
                    k=k, M=M, n=n, N=N,
                    bool_use_half_of_boundary_pmf = True,
                    bool_split_into_coeff_and_base_ten_exponent = False,
                    bool_use_decimal_type = bool_use_decimal_type,
                    int_decimal_computational_precision = \
                        int_decimal_computational_precision,
                    int_decimal_reporting_precision = \
                        int_decimal_reporting_precision,
                    bool_true_cdf_false_sf_none_min = False,
                    bool_numerical_approximation = True,
                    bool_approx_true_lanczos_false_spouge = True,)
                min_cdf_sf = min(cdf, sf)
        else : # if k > (n - k)
            sf = _sum_hypergeom_pmfs(
                k=k, M=M, n=n, N=N,
                bool_use_half_of_boundary_pmf = True,
                bool_split_into_coeff_and_base_ten_exponent = False,
                bool_use_decimal_type = bool_use_decimal_type,
                int_decimal_computational_precision = \
                    int_decimal_computational_precision,
                int_decimal_reporting_precision = \
                    int_decimal_reporting_precision,
                bool_true_cdf_false_sf_none_min = False,
                bool_numerical_approximation = True,
                bool_approx_true_lanczos_false_spouge = True,)
            if sf < half :
                min_cdf_sf = sf
            else : # Do NOT use "min_cdf_sf = cdf = 1 - sf" due to precision loss!
                cdf = _sum_hypergeom_pmfs(
                    k=k, M=M, n=n, N=N,
                    bool_use_half_of_boundary_pmf = True,
                    bool_split_into_coeff_and_base_ten_exponent = False,
                    bool_use_decimal_type = bool_use_decimal_type,
                    int_decimal_computational_precision = \
                        int_decimal_computational_precision,
                    int_decimal_reporting_precision = \
                        int_decimal_reporting_precision,
                    bool_true_cdf_false_sf_none_min = True,
                    bool_numerical_approximation = True,
                    bool_approx_true_lanczos_false_spouge = True,)
                min_cdf_sf = min(cdf, sf)
    if bool_use_decimal_type :
        with localcontext() as ctx :
            ctx.prec = int_decimal_reporting_precision
            # Round the final result back to the reporting precision:
            min_cdf_sf = +Decimal(min_cdf_sf)
    if bool_split_into_coeff_and_base_ten_exponent :
        min_cdf_sf = split_float(number = min_cdf_sf)
    return min_cdf_sf


def approx_spouge_hypergeom_min_cdf_sf(
        k, M, n, N,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = True,
        int_decimal_computational_precision = INT_DECIMAL_COMPUTATIONAL_PRECISION,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,) :
    with localcontext() as ctx :
        # Perform a high precision calculation
        ctx.prec = int_decimal_computational_precision
        half = Decimal(.5) if bool_use_decimal_type else np.float64(.5)
        if k <= (n - k) : # Rough estimate of what is smaller: "cdf" or "sf"
            cdf = _sum_hypergeom_pmfs(
                k=k, M=M, n=n, N=N,
                bool_use_half_of_boundary_pmf = True,
                bool_split_into_coeff_and_base_ten_exponent = False,
                bool_use_decimal_type = bool_use_decimal_type,
                int_decimal_computational_precision = \
                    int_decimal_computational_precision,
                int_decimal_reporting_precision = \
                    int_decimal_reporting_precision,
                bool_true_cdf_false_sf_none_min = True,
                bool_numerical_approximation = True,
                bool_approx_true_lanczos_false_spouge = False,)
            if cdf < half :
                min_cdf_sf = cdf
            else : # Do NOT use "min_cdf_sf = sf = 1 - cdf" due to precision loss!
                sf = _sum_hypergeom_pmfs(
                    k=k, M=M, n=n, N=N,
                    bool_use_half_of_boundary_pmf = True,
                    bool_split_into_coeff_and_base_ten_exponent = False,
                    bool_use_decimal_type = bool_use_decimal_type,
                    int_decimal_computational_precision = \
                        int_decimal_computational_precision,
                    int_decimal_reporting_precision = \
                        int_decimal_reporting_precision,
                    bool_true_cdf_false_sf_none_min = False,
                    bool_numerical_approximation = True,
                    bool_approx_true_lanczos_false_spouge = False,)
                min_cdf_sf = min(cdf, sf)
        else : # if k > (n - k)
            sf = _sum_hypergeom_pmfs(
                k=k, M=M, n=n, N=N,
                bool_use_half_of_boundary_pmf = True,
                bool_split_into_coeff_and_base_ten_exponent = False,
                bool_use_decimal_type = bool_use_decimal_type,
                int_decimal_computational_precision = \
                    int_decimal_computational_precision,
                int_decimal_reporting_precision = \
                    int_decimal_reporting_precision,
                bool_true_cdf_false_sf_none_min = False,
                bool_numerical_approximation = True,
                bool_approx_true_lanczos_false_spouge = False,)
            if sf < half :
                min_cdf_sf = sf
            else : # Do NOT use "min_cdf_sf = cdf = 1 - sf" due to precision loss!
                cdf = _sum_hypergeom_pmfs(
                    k=k, M=M, n=n, N=N,
                    bool_use_half_of_boundary_pmf = True,
                    bool_split_into_coeff_and_base_ten_exponent = False,
                    bool_use_decimal_type = bool_use_decimal_type,
                    int_decimal_computational_precision = \
                        int_decimal_computational_precision,
                    int_decimal_reporting_precision = \
                        int_decimal_reporting_precision,
                    bool_true_cdf_false_sf_none_min = True,
                    bool_numerical_approximation = True,
                    bool_approx_true_lanczos_false_spouge = False,)
                min_cdf_sf = min(cdf, sf)
    if bool_use_decimal_type :
        with localcontext() as ctx :
            ctx.prec = int_decimal_reporting_precision
            # Round the final result back to the reporting precision:
            min_cdf_sf = +Decimal(min_cdf_sf)
    if bool_split_into_coeff_and_base_ten_exponent :
        min_cdf_sf = split_float(number = min_cdf_sf)
    return min_cdf_sf


def normal_scipy_hypergeom_min_cdf_sf(
        k, M, n, N,
        bool_split_into_coeff_and_base_ten_exponent = False,) :
    cdf = np.float64('nan')
    if 0 < M and 0 <= N <= M and 0 <= n <= M and 0 <= k <= N and \
       0 <= (n - k) <= (M - N) : # (0 <= k <= n) is implied from others
        if (M == n or # => k == N
            M == N or # => k == n
            N == k or
            n == 0 or # => k == 0
            N == 0    # => k == 0
            ) and False: 
            min_cdf_sf = np.float64(0.)
        else :
            p = np.float64(N) / np.float64(M) # probability of success
            q = np.float64(1.) - p # probability of failure
            mu = np.float64(n) * p
            sigma = np.sqrt(np.float64(n) * p * q)
            # Adj. to account for discrete nature of Hypergeom. distr.
            if True or k == 0 :
                k_adj = np.float64(k)
            else :
                k_adj = np.float64(k) - np.float64(.5)
            z = (k_adj - mu) / sigma if sigma != np.float64(0.) else np.float64(0.)
            zero = np.float64(0.)
            one = np.float64(1.)
            half = np.float64(.5)
            cdf = norm.cdf(x=z, loc=zero, scale=one)
            if cdf < half :
                min_cdf_sf = cdf
            else : # Do NOT use "min_cdf_sf = sf = 1 - cdf" due to precision loss!
                sf = norm.sf(x=z, loc=zero, scale=one)
                min_cdf_sf = min(cdf, sf)
    if bool_split_into_coeff_and_base_ten_exponent :
        min_cdf_sf = split_float(number = min_cdf_sf)
    return min_cdf_sf


def normal_taylor_hypergeom_min_cdf_sf(
        k, M, n, N,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = True,
        int_decimal_computational_precision = INT_DECIMAL_COMPUTATIONAL_PRECISION,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,) :
    with localcontext() as ctx :
        # Perform a high precision calculation
        ctx.prec = int_decimal_computational_precision
        half = Decimal(.5) if bool_use_decimal_type else np.float64(.5)
        cdf = _sum_taylor_normal_cdf_terms(
            k=k, M=M, n=n, N=N,
            bool_use_half_of_boundary_pmf = True,
            bool_split_into_coeff_and_base_ten_exponent = False,
            bool_use_decimal_type = bool_use_decimal_type,
            int_decimal_computational_precision = \
                int_decimal_computational_precision,
            int_decimal_reporting_precision = \
                int_decimal_reporting_precision,
            bool_true_cdf_false_sf_none_min = True,
            bool_numerical_approximation = False,)
        if cdf < half :
            min_cdf_sf = cdf
        else : # Do NOT use "min_cdf_sf = sf = 1 - cdf" due to precision loss!
            sf = _sum_taylor_normal_cdf_terms(
                k=k, M=M, n=n, N=N,
                bool_use_half_of_boundary_pmf = True,
                bool_split_into_coeff_and_base_ten_exponent = False,
                bool_use_decimal_type = bool_use_decimal_type,
                int_decimal_computational_precision = \
                    int_decimal_computational_precision,
                int_decimal_reporting_precision = \
                    int_decimal_reporting_precision,
                bool_true_cdf_false_sf_none_min = False,
                bool_numerical_approximation = False,)
            min_cdf_sf = min(cdf, sf)
    if bool_use_decimal_type :
        with localcontext() as ctx :
            ctx.prec = int_decimal_reporting_precision
            # Round the final result back to the reporting precision:
            min_cdf_sf = +Decimal(min_cdf_sf)
    if bool_split_into_coeff_and_base_ten_exponent :
        min_cdf_sf = split_float(number = min_cdf_sf)
    return min_cdf_sf


def normal_approx_hypergeom_min_cdf_sf(
        k, M, n, N,
        bool_split_into_coeff_and_base_ten_exponent = False,
        bool_use_decimal_type = True,
        int_decimal_computational_precision = INT_DECIMAL_COMPUTATIONAL_PRECISION,
        int_decimal_reporting_precision = INT_DECIMAL_REPORTING_PRECISION,) :
    with localcontext() as ctx :
        # Perform a high precision calculation
        ctx.prec = int_decimal_computational_precision
        half = Decimal(.5) if bool_use_decimal_type else np.float64(.5)
        cdf = _sum_taylor_normal_cdf_terms(
            k=k, M=M, n=n, N=N,
            bool_use_half_of_boundary_pmf = True,
            bool_split_into_coeff_and_base_ten_exponent = False,
            bool_use_decimal_type = bool_use_decimal_type,
            int_decimal_computational_precision = \
                int_decimal_computational_precision,
            int_decimal_reporting_precision = \
                int_decimal_reporting_precision,
            bool_true_cdf_false_sf_none_min = True,
            bool_numerical_approximation = True,)
        if cdf < half :
            min_cdf_sf = cdf
        else : # Do NOT use "min_cdf_sf = sf = 1 - cdf" due to precision loss!
            sf = _sum_taylor_normal_cdf_terms(
                k=k, M=M, n=n, N=N,
                bool_use_half_of_boundary_pmf = True,
                bool_split_into_coeff_and_base_ten_exponent = False,
                bool_use_decimal_type = bool_use_decimal_type,
                int_decimal_computational_precision = \
                    int_decimal_computational_precision,
                int_decimal_reporting_precision = \
                    int_decimal_reporting_precision,
                bool_true_cdf_false_sf_none_min = False,
                bool_numerical_approximation = True,)
            min_cdf_sf = min(cdf, sf)
    if bool_use_decimal_type :
        with localcontext() as ctx :
            ctx.prec = int_decimal_reporting_precision
            # Round the final result back to the reporting precision:
            min_cdf_sf = +Decimal(min_cdf_sf)
    if bool_split_into_coeff_and_base_ten_exponent :
        min_cdf_sf = split_float(number = min_cdf_sf)
    return min_cdf_sf
