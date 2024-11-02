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
#
# http://www.vttoth.com/CMS/projects/41
# https://beta.boost.org/doc/libs/1_48_0/libs/math/doc/sf_and_dist/html/math_toolkit/backgrounders/lanczos.html
# http://mtweb.cs.ucl.ac.uk/mus/arabidopsis/xiang/software/boost_1_47_0/libs/math/doc/sf_and_dist/html/math_toolkit/backgrounders/lanczos.html
# https://laplace.physics.ubc.ca/ThesesOthers/Phd/pugh.pdf
#
# https://en.wikipedia.org/wiki/Spouge%27s_approximation
#

import math
import numpy as np
# https://docs.python.org/3/library/decimal.html
from decimal import Decimal, getcontext, localcontext
from scipy.special import loggamma
from functools import reduce
from operator import mul


# Lanczos Approximation to the Log Gamma Function.
# The code is based on the idea from this source:
# http://mrob.com/pub/ries/lanczos-gamma.html#fn_4
def log_gamma(a) :
    return math.log(
        (2.5066282751072974716040095600075 +
         190.95517189307639668340239599573 / (a + 2.) -
         216.83668184372796999227712657839 / (a + 3.) +
         60.194417640233328256162764625455 / (a + 4.) -
         3.0875132392854582647683040300775 / (a + 5.) +
         0.00302963870525306071894639603305 / (a + 6.) -
         1.35238595907259587106470354765e-5 / (a + 7.)) /
         (a + 1.)) - (a + 6.5) + (a + 1.5) * math.log(a + 6.5)

def log_gamma_decimal(
        a,
        int_decimal_computational_precision= 64,
        int_decimal_reporting_precision = 64) :
    with localcontext() as ctx :
        ctx.prec = int_decimal_computational_precision
        # https://en.wikipedia.org/wiki/Euler%27s_constant
        # https://functions.wolfram.com/GammaBetaErf/LogGamma/02/

        # Lanczos Approximation to the Log Gamma Function.
        # The code is based on the idea from this source:
        # http://mrob.com/pub/ries/lanczos-gamma.html#fn_4
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
        log_gamma_lambda = lambda a : ctx.ln(
            (b1+b2/(a+c2)-b3/(a+c3)+b4/(a+c4)-b5/(a+c5)+b6/(a+c6)-b7/(a+c7))/
            (a+c1))-(a+c6p5)+(a+c1p5)*ctx.ln(a+c6p5)
        r = log_gamma_lambda(a)
    getcontext().prec = int_decimal_reporting_precision
    r = +r # Round the final result back to the default precision
    return r


def __combination(n, k) :
    if k < 0 :
        return 0
    else :
        return math.comb(n,k)


def _combination(n, k) :
    r = Decimal(0)
    if k < 0 :
        return r
    r = Decimal(1)
    for l in range(1,n-k+1) :
        r *= n + 1 - l
        r /= l
    return r

# Better version than _combination(n, k) and _combination(n, k)
def combination(n, k) :
    # n! / (k! * (n-k)!)
    if k < 0 :
        c = Decimal(0)
    elif k >= n or k == 0 :
        c = Decimal(1)
    elif k == 1 or k == n - 1:
        c = Decimal(n)
    else :
        one = Decimal(1)
        c = Decimal(1)
        if k >= (n - k) :
            num_min = k+1
            num_max = n
            den_min = 2
            den_max = n-k
        else :
            num_min = n-k+1
            num_max = n
            den_min = 2
            den_max = k
        num_curr = num_min
        den_curr = den_min
        while num_curr <= num_max and den_curr <= den_max :
            if c >= one :
                c /= den_curr
                den_curr += 1
            else :
                c *= num_curr
                num_curr += 1
        while num_curr <= num_max :
            c *= num_curr
            num_curr += 1
        while den_curr <= den_max :
            c /= den_curr
            den_curr += 1
    return c


def generate_B(n) :
    # list of rows (aka lists): n-by-n
    B = [row.copy() for row in [[Decimal(0)] * n] * n]
    B[0] = [Decimal(1) for c in range(n)]
    for r in range(1,n) :
        for c in range(n) :
            B[r][c] = Decimal(combination(n=(r+c-1), k=(c-r)))
            if (c - r) % 2 == 1 :
                B[r][c] = -B[r][c]
    return B


def generate_C(n) :
    # list of rows (aka lists): n-by-n
    C = [row.copy() for row in [[Decimal(0)] * n] * n]
    zero = Decimal(0)
    for i in range(1,n) :
        for j in range(i+1) :
            C[i][j] = zero
            for k in range(i+1) :
                C[i][j] += combination(n=2*i, k=2*k) * \
                    combination(n=k, k=k+j-i)
            if (i - j) % 2 == 1 :
                C[i][j] = -C[i][j]
    C[0][0] = Decimal(1) / Decimal(2)
    return C


def generate_D(n) :
    # list of rows (aka lists): n-by-n
    D = [row.copy() for row in [[Decimal(0)] * n] * n]
    D[0][0] = Decimal(1)
    D[1][1] = Decimal(-1)
    for i in range(2,n) :
        D[i][i] = D[i-1][i-1] * (2 * (2 * i - 1))
        D[i][i] /= (i - 1)
    return D


def generate_F(g, n) :
    # g is a floating point; n is an integer
    # list of rows (aka lists) with 1 column : n-by-1
    F = [row.copy() for row in [[Decimal(0)]] * n]
    g = Decimal(format(g, ".15g"))
    half = Decimal(1) / Decimal(2)
    for a in range(n) :
        F[a][0] = Decimal(2)
        for i in range(a+1, 2*a+1) :
            F[a][0] *= i
            F[a][0] /= 4
        F[a][0] *= getcontext().exp(Decimal(a) + g + half)
        F[a][0] /= getcontext().power(Decimal(a) + g + half, a)
        F[a][0] /= getcontext().sqrt(Decimal(a) + g + half)
    return F


def generate_Z(z, n) :
    # z is an integer; n is an integer
    # list of 1 row (aka lists) with n column : 1-by-n
    Z = [row.copy() for row in [[Decimal(0)] * n] * 1]
    Z[0][0] = Decimal(1)
    one = Decimal(1)
    for i in range(1,n) :
         Z[0][i] = one / (z+i)
    return Z


def mtx_mult(L, R) :
    n = len(L)
    m = len(L[0]) # len(R)
    p = len(R[0])
    P = [row.copy() for row in [[Decimal(0)] * p] * n]
    for i in range(n) :
        for j in range(p) :
            s = Decimal(0)
            for k in range(m) :
                s += L[i][k] * R[k][j]
            P[i][j] = s
    return P


def _test_matrix_multiplication() :
    L = [[1, 2], [2, 3], [4, 5]] 
    R = [[4, 5, 1], [6, 7, 2]] 
    #
    print("Matrix L :") 
    print(L) 
    print("Matrix R :") 
    print(R) 
    #
    result = np.dot(L, R)
    print("The matrix multiplication is:") 
    print(result) 
    #
    result = mtx_mult(L=L, R=R)
    print("The matrix multiplication is:") 
    print(result) 


def log_gamma_lanczos_approx(z, _n, _g, lst_p) :
    r = lst_p[0]
    for i in range(1,_n) :
        r += lst_p[i] / (z+i)
    g = Decimal(format(_g, ".15g"))
    half = Decimal(1) / Decimal(2)
    t = Decimal(z) + g + half
    log_g = getcontext().ln(r) + (Decimal(z) + half) * getcontext().ln(t) - t
    return log_g


def generate_lanczos_approx_coeffs(
        n=13, g=6.02468, int_digits=64, z=None) :
    getcontext().prec = int_digits
    B = generate_B(n=n)
    C = generate_C(n=n)
    D = generate_D(n=n)
    F = generate_F(g=g, n=n)
    P = mtx_mult(mtx_mult(mtx_mult(D,B),C),F)
    lst_p = [c[0] for c in P]
    if z is not None :
        Z = generate_Z(z=z, n=n)
        R = mtx_mult(Z,P)
    
        g = Decimal(format(g, ".15g"))
        half = Decimal(1) / Decimal(2)
    
        r = Decimal(z) + g + half
        r = getcontext().ln(R[0][0]) + (Decimal(z) + half) * getcontext().ln(r) - r
        return (r, lst_p)
    else :
        return lst_p


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


# https://en.wikipedia.org/wiki/Spouge%27s_approximation
def generate_spouge_approx_coeffs(a) :
    lst_c = [Decimal(0)] * a
    half = Decimal(1) / Decimal(2)
    if a > 0 :
        lst_c[0] = getcontext().sqrt(approx_pi(digits=getcontext().prec) * 2)
    if a > 1 :
        lst_c[1] = getcontext().power(-1+a, half) * getcontext().exp(-1+a)
    if a > 2 :
        lst_c[2] = Decimal(-1) * getcontext().power(-2+a, Decimal(2)-half) * \
            getcontext().exp(-2+a)
    for k in range(3, a) :
        lst_c[k] = (
            getcontext().power(-k+a, k-half) *
            getcontext().exp(-k+a) /
            reduce(mul, (i for i in range(2, k))) *
            Decimal(-1 if (k - 1) % 2 == 1 else 1))
    return lst_c


def log_gamma_spouge_approx(z, lst_c) :
    a = len(lst_c)
    half = Decimal(1) / Decimal(2)
    log_g = getcontext().ln(z+a) * (half+z) + \
        (-z-a) + \
        getcontext().ln(sum([lst_c[0]] + [lst_c[k] / (z+k) for k in range(1, a)]))
    return log_g


def main():

    if False :
        _test_matrix_multiplication()

    int_digits = 64
    getcontext().prec = int_digits

    if True :
        z = 102
        n = 13
        g = 6.02468
        print("Generating Lanczos approximation coefficients for log-Gamma " +
              "(n=" + str(n) + "; g=" + str(g) + "):")
        print()
        (log_g, lst_p) = generate_lanczos_approx_coeffs(
            n = n, g = g, int_digits = int_digits, z = z)
        print(lst_p)
        print()
        print("Lanczos    = " + str(log_g))
        print("Benchmark1 = " + str(log_gamma(a=z)))
        print("Benchmark2 = " + str(log_gamma_decimal(a=Decimal(z))))
        print("Benchmark3 = " + str(loggamma(z + 1)))
        print()

    if True :
        z = 102
        n = 13
        g = 6.02468
        print("Reusing hardcoded Lanczos approximation coefficients " +
              "for log-Gamma (n=" + str(n) + "; g=" + str(g) + "):")
        print()
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
            Decimal('9.06038883356544784242812502552213897255901E-8'),]
        log_g = log_gamma_lanczos_approx(z=z, _n=n, _g=g, lst_p=lst_p)
        print("Lanczos    = " + str(log_g))
        print("Benchmark1 = " + str(log_gamma(a=z)))
        print("Benchmark2 = " + str(log_gamma_decimal(a=Decimal(z))))
        print("Benchmark3 = " + str(loggamma(z + 1)))
        print()

    if True :
        z = 102
        a = 20 # for a in range(25,19,-1) :
        print("Generating Spouge approximation coefficients " + 
              "for log-Gamma (a=" + str(a) + "):")
        print()
        lst_c = generate_spouge_approx_coeffs(a=a)
        log_g = log_gamma_spouge_approx(z = z, lst_c=lst_c)
        print(lst_c)
        print()
        print("Spouge     = " + str(log_g))
        print("Benchmark1 = " + str(log_gamma(a=z)))
        print("Benchmark2 = " + str(log_gamma_decimal(a=Decimal(z))))
        print("Benchmark3 = " + str(loggamma(z + 1)))
        print()

    if True :
        z = 102
        print("Reusing hardcoded Spouge approximation coefficients " +
              "for log-Gamma (a=" + str(a) + "):")
        print()
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
        log_g = log_gamma_spouge_approx(z = z, lst_c=lst_c)
        pi = approx_pi(digits = int_digits)
        print("PI ~= " + str(pi))
        print("Spouge     = " + str(log_g))
        print("Benchmark1 = " + str(log_gamma(a=z)))
        print("Benchmark2 = " + str(log_gamma_decimal(a=Decimal(z))))
        print("Benchmark3 = " + str(loggamma(z + 1)))



if __name__ ==  '__main__':
    main()
