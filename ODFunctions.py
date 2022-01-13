from math import radians, copysign, pi, sin, cos, atan2, degrees, acos, asin, sqrt, atan
from numpy import array, cross, dot, linalg
import numpy.polynomial.polynomial as npp

k = 0.01720209895  # Gaussian gravitational constant
mu = 1

def get_data_points(path, number):
    """Takes in the following parang_mometers:
    1. a text file path, path
    2. a number specifying the number of observations wanted, number

    Every line in the file contains the following information for one 
    observation night:
    1. UT date/time at middle of observation (Julian Days)
    2. RA (hh:mm:ss.ss), DEC (dd:mm:ss.s), and
    3. Sun vector from JPL Horizons (AU, equatorial cartesian, J2000, apparent states)
    
    Example:
        2457927.76426 17:27:15.08 -24:44:53.0 -3.092452663004570E-02 9.321463833793008E-01 4.040478235741391E-01"""

    # open the file, read all lines and save every line to a list
    file = open(path, "r")
    observation_data = file.read().splitlines()
    file.close()

    # get the total observations in the file
    total_observations = len(observation_data)

    # return all observations if all them are wanted
    if number == total_observations:
        return observation_data
    # display if the user want more observations than is available
    elif number >= total_observations:
        print("There are only", number, "observations.")
        number = int(input(""))

    # Otherwise, make the user choose the ones they want.
    # list that will holds observation the user chooses
    observations_chosen = []
    # in case the user chooses observations out of order, the indices are 
    # saved first and then sorted to get them in the correct order
    indices = []
    print("There are more than", number, "observations available. Which would you like to use?")
    print("Enter the number next to the observation you want to use to select it.")

    # display the possible observations
    for i in range(1, total_observations+1):
        print( str(i) + ".", observation_data[i-1] )
    
    # while the user has not chosen three observations...
    count = 0
    while count != number:
        index = get_valid_number(0, total_observations, indices, "Observation Number: ")
        indices.append(index)
        count += 1
    
    for index in sorted(indices):
        observations_chosen.append( observation_data[index-1] )
    return observations_chosen

def decimalToSexagesimal(degrees, hour = False):
    """"Takes in decimal degrees and returns tuple holding the sexagesimal
    version. If you want the output in hours, the function takes in a second
    boolean argument that can be set to True. """

    hours, minutes, seconds = degrees, 0, 0
    if hour:
        hours /= 15
    minutes = (hours - int(hours))*60
    hours = int(hours)

    seconds = (minutes - int(minutes))*60
    minutes = int(minutes)

    return hours, minutes, seconds

def sexagesimalToDecimal(degrees, minutes, seconds, radians = False, normalize = False):
    """Converts sexagesimal value to degrees. Takes in a fourth boolean value that specifies
    if the output is to be returned in radians, and fifth boolean value that specifies to 
    normalize the angle"""

    # handle a negative angle
    n_degrees = degrees
    n_minutes = copysign(minutes, degrees)
    n_seconds = copysign(seconds, degrees)

    # perform angle conversion
    n_minutes += n_seconds/60
    n_degrees += n_minutes/60

    # return result
    if normalize:
        n_degrees %= 360
    if radians:
        return n_degrees*pi/180 
    else:
        return n_degrees

def convert_data_line_to_numbers(string):
    """"Reads data line and returns the following numbers in a dictionary:
    1. UT time in Julian Days
    2. Right ascension in radians
    3. Declination in radians
    4. numpy array holding Sun vector components"""

    # separate values in the string
    values = string.split()

    # get UT time
    time = float( values[0] )

    # get RA in sexagesimal hour form and convert to radians
    sexagesimal_RA = [float(a) for a in values[1].split(":")]
    # convert decimal hour into decimal degree by multiplying
    # by 15
    RA = sexagesimalToDecimal( *sexagesimal_RA, True)*15

    # get DEC in sexagesimal form and convert to radians
    sexagesimal_DEC = [float(a) for a in values[2].split(":")]
    DEC = sexagesimalToDecimal( *sexagesimal_DEC, True )

    # Get Sun Vector and convert to numpy array
    sun_vector = array( [float(a) for a in values[3:]] )

    return {"time":time, "RA":RA, "DEC":DEC, "sun vector":sun_vector}

def calculate_taus(k, time1, time2, time3):
    """Takes in a k-value, three Julian day values and returns the tau 
    intervals as a dictionary"""

    tau3 = k * ( time3 - time2 )
    tau1 = k * ( time1 - time2 )
    tau = tau3 - tau1

    return {"tau":tau, "tau1":tau1, "tau3":tau3}

def calculate_rhohat(RA, DEC):
    """Takes in RA and DEC values in radians and returns a numpy array
    holding the components of the rhohat vector"""

    x_comp = cos(RA) * cos(DEC)
    y_comp = sin(RA) * cos(DEC)
    z_comp = sin(DEC)

    return array([x_comp, y_comp, z_comp])

def calculate_Ds(sun_vectors, rhohats):
    """Takes in two dictionaries:
    1. sun vectors
    2. rhohats
    
    Returns dictionary holding D values needed to calculate ranges"""

    D0 = dot(rhohats[1], cross( rhohats[2], rhohats[3] ))
    Ds = {"D0":D0}
    for j in "123":
        Ds["D1" + j] = dot( cross( sun_vectors[int(j)], rhohats[2] ), rhohats[3] )
        Ds["D2" + j] = dot( cross( rhohats[1], sun_vectors[int(j)] ), rhohats[3] )
        Ds["D3" + j] = dot( rhohats[1],  cross( rhohats[2], sun_vectors[int(j)] ) )
    
    return Ds

def getMag(vector):
    """Takes in a list that holds the components of a vector
    and returns the magnitude of the vector"""

    # check if the parameter given is already a scalar
    if type(vector).__name__ == "float64":
        return vector
    
    return (vector[0]**2 + vector[1]**2 + vector[2]**2)**0.5

def SEL(taus, Sun2, rhohat2, Ds):
    """Scalar equation of Lagrange function.
    Takes in the following parang_mometers:
    1. a dictionary holding all tau values
    2. sun vector of second observation
    3. rhohat of second observation
    4. a dictionary holding D0, D21, D22, and D23 in that order.

    Returns an r2 and its corresponding rho value chosen by the user
    if there is more than 1 positive, real root"""

    possible_rhos = []

    A1 = taus["tau3"]/taus["tau"]
    B1 = (A1 / 6) * (taus["tau"]**2 - taus["tau3"]**2)
    A3 = -taus["tau1"]/taus["tau"]
    B3 = (A3 / 6) * (taus["tau"]**2 - taus["tau1"]**2)

    A = (A1 * Ds["D21"] - Ds["D22"] + A3*Ds["D23"]) / -Ds["D0"]
    B = (B1 * Ds["D21"] + B3*Ds["D23"]) / -Ds["D0"]

    E = -2 * (dot(rhohat2, Sun2))
    F = getMag(Sun2)**2
    
    # coefficients of 8th degree polynomial
    a = -(A**2 + A*E + F)
    b = -(2*A*B + B*E)
    c = -B**2

    # find the roots and then filter out the real positive numbers
    possible_roots = [ x.real for x in npp.polyroots([c, 0, 0, b, 0, 0, a, 0, 1]) if x.imag == 0 and x.real > 0]

    # keep track of corresponding rho values with a list
    for root in possible_roots:
        possible_rhos.append( A + B / root**3 )
    
    # filter out roots with positive rhos
    roots = []
    rhos = []
    for index in range(len(possible_roots)):
        # if rho value is positive, add it and its root value to their
        # corresponding lists
        if possible_rhos[index] >= 0:
            roots.append( possible_roots[index] )
            rhos.append( possible_rhos[index] )

    index = 0
    # check if there are more than one possible roots
    if len(roots) > 1:
        print("There are more than one possible r2 values. Which would you like to use?")
        print("Enter the number on the left of the root to select it.")

        # display possible roots
        for index in range(len(possible_roots)):
            print(str(index + 1) + ".", possible_roots[index])
    
        index = get_valid_number(1, len(possible_roots), [], "Root number: ") - 1
    return roots[index], rhos[index]

def fDeltaE(x, semi_major_a, r2, r2dot, n, tau):
    """Takes in the following values:
    1. an initial guess, x
    2. semi major axis, semi_major_a
    3. position vector of second observation, r2
    4. velocity vector of second observation, r2dot
    5. coefficient n
    6. Gaussian interval, tau
    Returns f(initial guess)"""

    second_term = -(1 - getMag(r2)/semi_major_a) * sin(x)
    third_term = ( dot(r2, r2dot) / (n * semi_major_a**2) ) * ( 1 - cos(x) )
    fourth_term = -n * tau

    return x + second_term + third_term + fourth_term

def fDeltaEPrime(x, semi_major_a, r2, r2dot, n, tau):
    """Takes in the following values:
    1. an initial guess, x
    2. semi major axis, semi_major_a
    3. position vector of second observation, r2
    4. velocity vector of second observation, r2dot
    5. coefficient n
    6. Gaussian time interval, tau
    Returns f'(initial guess)"""
    
    second_term = -(1 - getMag(r2)/semi_major_a) * cos(x)
    third_term = ( dot(r2, r2dot) / (n * semi_major_a**2) ) * sin(x)

    return 1 + second_term + third_term

def newtons_method(f, fPrime, initial_numb, tol, args=[]):
    """Takes in a function f, its derivative fPrime, an initial guess (AKA x0), a 
    tolerance value (AKA convergence value), and arguments that f and fPrime need"""
    # initial guesses
    root = initial_numb
    rootP = root - f(initial_numb, *args) / fPrime(initial_numb, *args)

    # while the tolerance has not been met...
    while abs(root - rootP) > tol:
        root = rootP
        rootP = root - f(root, *args) / fPrime(root, *args)

    return rootP

def fg(tau, r2, r2dot, flag = "f"):
    """Takes in the following values:
    1. Gaussian time interval, tau
    2. position vector, r2
    3. velocity vector, r2dot
    4. indicator of 2nd, 3rd, or 4th order or functions, flag
    Returns the values of f and g"""

    mu = 1
    r2_mag = getMag(r2)
    u = mu / r2_mag**3

    # f and g series to the second order
    if flag == "2":
        f_series = [
            1,
            - 0.5 * u * tau**2,
        ]
        g_series = [
            tau,
            -1/6 * u * tau**3
        ]
        return sum(f_series), sum(g_series)
    
    # numbers needed for the third and fourth order of f and g series
    r2dot_2 = -mu * r2 / r2_mag**3
    r2dot_3 = (-mu / r2_mag**5) * ( r2_mag**2 * r2 - 3 * dot(r2, r2dot) * r2)
    z = dot(r2, r2dot) / r2_mag**2
    q = dot( r2dot, r2dot ) / r2_mag**2 - u

    # f and g functions
    if flag == "f":
        # calculate the semi major axis and n with current r2 and r2dot values
        semi_major_a = get_semi_major_a(r2dot, getMag(r2))
        n = sqrt(1 / semi_major_a**3)

        # estimate the change in E using Newton's method
        deltaE = newtons_method(fDeltaE, fDeltaEPrime, n*tau, 1E-12, [semi_major_a, r2, r2dot, n, tau])

        # f and g function definitions
        f = 1 - (semi_major_a/getMag(r2)) * ( 1 - cos(deltaE) )
        g = tau + (1/n) * (sin(deltaE) - deltaE)

        return f, g
    # 3rd order of f and g series
    elif flag == "3":
        f_series = [
            1,
            - 0.5 * u * tau**2,
            0.5 * u * z * tau**3
        ]

        g_series = [
            tau,
            -1/6 * u * tau**3
        ]
        return sum(f_series), sum(g_series)
    
    # fourth order of f and g series
    elif flag == "4":
        f_series = [
            1,
            - 0.5 * u * tau**2,
            0.5 * u * z * tau**3,
           1/24 * ( 3*u*q - 15 * u * z**2 + u**2) * tau**4
        ]

        g_series = [
            tau,
            -1/6 * u * tau**3,
            1/4 * u * z * tau**4
        ]
        return sum(f_series), sum(g_series)
    # displays if flag given was incorrect
    else:
        print("invalid flag value given")

def calculate_scalar_ranges(taus, r2, r2dot, Ds, flag):
    """Takes in the following:
    1. dictionary holding tau values, taus
    2. numpy array representing the position vector for the 2nd observation,
       r2
    3. dictionary holding all D coefficients, Ds
    4. flag to pass on to the fg function

    The f and g series are then calculated according to the flag parameter 
    to then find c1 and c3. c1 and c3 are then used to calculate the scalar ranges.
    Returns a dictionary holding the scalar ranges"""

    f1, g1 = fg(taus["tau1"], r2, r2dot, flag)
    f3, g3 = fg(taus["tau3"], r2, r2dot, flag)

    cs = {
    "c1": g3 / ( f1*g3 - g1*f3 ),
    "c2": -1,
    "c3": -g1 / ( f1*g3 - g1*f3 )
    }

    scalar_ranges = {}
    for index in "123":
        numerator = cs["c1"]*Ds["D" + index + "1"] + cs["c2"]*Ds["D" + index + "2"] + cs["c3"]*Ds["D" + index + "3"]
        scalar_ranges[int(index)] = numerator / ( cs["c"+index]*Ds["D0"] )
    
    return scalar_ranges

def calculate_r2dot(taus, r1, r2, r3, r2dot, flag):
    """Takes in the following parameters:
    1. dictionary holding tau values, taus
    2. all position vectors, r1, r2, and r3
    3. velocity vector of second observation, r2dot
    4. flag that's passed on to the fg function.
     Does the following:
        1. calculates f and g according to the flag
        2. calculates c1 and c3
        3. finds d1 and d3 in the equation r2dot = d1*r1 + d3*r3,
        4. returns r2dot"""

    f1, g1 = fg(taus["tau1"], r2, r2dot, flag)
    f3, g3 = fg(taus["tau3"], r2, r2dot, flag)

    d1 = -f3 / ( f1*g3 - f3*g1 )
    d3 = f1 / ( f1*g3 - f3*g1 )

    return d1*r1 + d3*r3

def get_angular_momentum(r2, r2dot):
    # find angular momentum by finding the cross product
    # of r2 and r2dot
    return cross(r2, r2dot)

def get_perihelion(ang_mom, r2_dot, r2, r2_mag):
    return -cross(ang_mom, r2_dot) - r2/r2_mag

def get_eccentricity(perihelion_vector):
    return getMag(perihelion_vector)

def get_semi_major_a(r2_dot, r2_mag):
    return 1/( (2/r2_mag) - dot(r2_dot,r2_dot))

def get_inclination(ang_mom):
    return atan( (ang_mom[0]**2 + ang_mom[1]**2)**0.5 / ang_mom[2] )

def get_ascending_node(ang_mom, ang_mom_mag, inclination):
    cos_of_AN =  -ang_mom[1] / (ang_mom_mag * sin(inclination)) 
    sin_of_AN = ang_mom[0] / (ang_mom_mag * sin(inclination))
    ascending_node = atan2( sin_of_AN, cos_of_AN )
    if ascending_node < 0:
        ascending_node += 2*pi
    return ascending_node

def get_arg_of_perihelion(r2, r2_mag, inclination, eccentricity, semi_major_a, ascending_node, ang_mom_mag, r2dot):
    sin_of_U = r2[2] / (r2_mag * sin(inclination))
    cos_of_U = ( r2[0]*cos(ascending_node) + r2[1]*sin(ascending_node) ) / r2_mag
    U = atan2( sin_of_U, cos_of_U )

    cos_of_mu = (1/eccentricity)*( (semi_major_a * (1 - eccentricity**2))/r2_mag -1)
    sin_of_mu = semi_major_a*(1 - eccentricity**2) / (eccentricity*ang_mom_mag) * ( dot(r2, r2dot) / r2_mag)
    mu = atan2( sin_of_mu, cos_of_mu )

    if U-mu < 0:
        return U - mu + 2*pi
    return U - mu

def getMeanAnomaly(eccentricity, r2_mag, semi_major_a):
    eccen_anom =  (1/eccentricity) * (1 - r2_mag/semi_major_a) 
    eccen_anom = acos(eccen_anom)
    eccen_anom = atan2(sin(eccen_anom), cos(eccen_anom))
    mean_anom = eccen_anom - eccentricity*sin(eccen_anom)
    return mean_anom

def calculate_percent_diff(expected, calculated):
    return (expected-calculated)/expected*100

def display_results(eccentricity, semi_major_a, inclination, ascending_node, arg_of_perihelion, mean_anom):
    elements = ["eccentricity", "semi-major axis", "inclination", "ascending node", "argument of perihelion", "mean anomaly"]
    expectedVals = [3.442331151611204E-01, 1.056800057440451E+00, radians(2.515524946767665E+01), 
                    radians(2.362379850093818E+02), radians(2.555046043539150E+02), radians(1.404194630208429E+02)]
    calculatedVals = [eccentricity, semi_major_a, inclination, ascending_node, arg_of_perihelion, mean_anom]
    for elem in range(6):
        print("*"*60)
        print(elements[elem])
        print("expected value:", expectedVals[elem])
        print("calculated value:", calculatedVals[elem] )
        print("percent difference:", calculate_percent_diff(expectedVals[elem], calculatedVals[elem]))

def get_orbital_elements(r2, r2_dot):
    """Given the position and velocity vectors, the following occurs:
    1. The magnitudes and unit vectors or the position and velocity 
       velocity vectors are calculated.
    2. The angular momentum is determined.
    3. orbital elements are determined in the following order:
        eccentricity, semi-major axis, inclination, ascending node,
        argument of perihelion, and mean anomaly"""

    # position vector data
    r2_mag = getMag(r2)
    radius_unitV = r2/r2_mag

    # velocity vector data
    velocity_mag = getMag(r2_dot)
    velocity_unitV = r2_dot/velocity_mag

    # angular momentum data
    ang_mom = get_angular_momentum(r2, r2_dot)
    ang_mom_mag = getMag(ang_mom)

    perihelion_vector = get_perihelion(ang_mom, r2_dot, r2, r2_mag)

    # 6 orbital elements
    eccentricity = get_eccentricity(perihelion_vector)

    semi_major_a = get_semi_major_a(r2_dot, r2_mag)

    inclination = get_inclination(ang_mom)

    ascending_node = get_ascending_node(ang_mom, ang_mom_mag, inclination)

    arg_of_perihelion = get_arg_of_perihelion(r2, r2_mag, inclination, eccentricity, \
        semi_major_a, ascending_node, ang_mom_mag, r2_dot)

    meanAnomaly = getMeanAnomaly(eccentricity, r2_mag, semi_major_a)

    # display_results(eccentricity, semi_major_a, inclination, ascending_node, arg_of_perihelion, meanAnomaly)
    return eccentricity, semi_major_a, inclination, ascending_node, arg_of_perihelion, meanAnomaly

def calculate_new_taus(k, times, scalar_ranges, c):
    """Takes in the following:
    1. k, Gaussian gravitational constant
    2. list holding Julian days, in order
    3. dictionary holding scalar ranges
    4. speed of light in AU per mean solar day
    Adjusts for the time it takes light to travel, and returns a dictionary"""

    ctime1 = times[0] - scalar_ranges[1]/c
    ctime2 = times[1] - scalar_ranges[2]/c
    ctime3 = times[2] - scalar_ranges[3]/c

    return calculate_taus(k, ctime1, ctime2, ctime3)
    
def equatorialToEcliptic(obliquity, vector):
    obliquity_matrix = array([
		[1,              0,               0],
		[0, cos(obliquity), -sin(obliquity)],
		[0, sin(obliquity),  cos(obliquity)]
    ])
    return dot(linalg.inv(obliquity_matrix), vector)

def get_valid_number(Min, Max, already_chosen, prompt_message, 
range_error_message = "That is not an option. Try again.", 
invalid_input_message = "You entered invalid input. Try again."):
    """Takes in the following parameters:
    1. An inclusive range described by its maximum and minimum values, Min and Max
    2. Message to prompt the user to enter 
    
    Optional Parameters:
    3. A message to show when the number entered is not in the range described by Min and Max
    4. A message to show when the user enter invalid input, such as a letter
    
    The function then checks if the number given is withing the range, an actual number,
    and if it has not already been chosen.
    Returns valid number"""

    while True:
        try:
            number = int(input(prompt_message))
            if number >= Min and number <= Max:
                return number
            if number in already_chosen:
                print("This number has already been chosen. Try again.")
            else:
                print(range_error_message)
                continue
        except:
            print(invalid_input_message)

def ephemeris_generator_1(r2, r2dot, t, obliquity, sun_vector):
    time_of_perihelion = 2457852.08061

    # calculate orbital elements
    eccentricity, semi_major_a, inclination, ascending_node, arg_of_perihelion, init_mean_anom = get_orbital_elements(r2, r2dot)

    # calculate mean anomaly at ephemeris generation time
    mean_anom = calculate_mean_anom(t, time_of_perihelion, semi_major_a)

    # find eccentric anomaly
    # passed on init_mean_anom because there is no change in mean anomaly
    eccen_anom = newtons_method(f, f_prime, mean_anom, 1E-12, [eccentricity, 0])

    position_vector = getCartesianCoords(semi_major_a, eccen_anom, eccentricity)

    position_vector = spinVector(ascending_node, inclination, arg_of_perihelion, position_vector)

    position_vector = ecliptic_to_equatorial(obliquity, position_vector)

    range_vector = position_vector + sun_vector
    rhohat = range_vector / getMag(range_vector)

    RA, DEC = getRAandDEC(rhohat)

    return RA, DEC

def ephemeris_generator_2(components, t, obliquity, sun_vector):
    time_of_perihelion = 2457852.08061

    # calculate orbital elements
    eccentricity, semi_major_a, inclination, \
        ascending_node, arg_of_perihelion, init_mean_anom = get_orbital_elements(components[:3], components[3:])

    # calculate mean anomaly at ephemeris generation time
    mean_anom = calculate_mean_anom(JDays, time_of_perihelion, semi_major_a)

    # find eccentric anomaly
    eccen_anom = newtons_method(f, f_prime, mean_anom, 1E-12, [eccentricity, 0])

    position_vector = getCartesianCoords(semi_major_a, eccen_anom, eccentricity)

    position_vector = spinVector(ascending_node, inclination, arg_of_perihelion, position_vector)

    position_vector = ecliptic_to_equatorial(obliquity, position_vector)

    range_vector = position_vector + sun_vector
    rhohat = range_vector / getMag(range_vector)

    RA, DEC = getRAandDEC(rhohat)

    return RA, DEC

def getRAandDEC(rhohat):
    DEC = asin(rhohat[2])
    RA = atan2( rhohat[1] / cos(DEC), rhohat[0] / cos(DEC) ) 

    if RA < 0:
        RA += 2*pi

    return RA, DEC

def calculate_mean_anom(JDays, time_of_perihelion, semi_major_a, mean_anom = 0):
    n = k*sqrt(1 / semi_major_a**3)
    return n * (JDays - time_of_perihelion)

def f(x, eccen, mean_anom):
    return x - eccen * sin(x) - mean_anom


def f_prime(x, eccen, mean_anom):
    return 1 - eccen * cos(x)

def getCartesianCoords(semi_major_a, eccen_anom, eccen):
    """Takes in semi-major axis, eccentric anomaly, and eccentricity
       and returns the Cartesian coordinates of the position vector"""
    x = semi_major_a * cos(eccen_anom) - semi_major_a * eccen
    y = semi_major_a * sqrt(1 - eccen**2) * sin(eccen_anom)
    z = 0
    arr = array([x, y, z])
    return arr

def spinVector(ascending_node, inclination, perihelion, vector):
    ascending_node_matrix = array([
        [cos(ascending_node), -sin(ascending_node), 0],
        [sin(ascending_node),  cos(ascending_node), 0],
        [                  0,                    0, 1]
    ])

    inclination_matrix = array([
        [1,                0,                 0],
        [0, cos(inclination), -sin(inclination)],
        [0, sin(inclination),  cos(inclination)]
    ])

    perihelion_matrix = array([
		[cos(perihelion), -sin(perihelion), 0], 
        [sin(perihelion),  cos(perihelion), 0],
        [              0,                0, 1] 
    ])
    newVector = dot( perihelion_matrix, vector )
    newVector = dot( inclination_matrix, newVector)
    newVector = dot(ascending_node_matrix, newVector)
    return newVector

def ecliptic_to_equatorial(obliquity, vector):
    obliquity_matrix = array([
		[1,              0,               0],
		[0, cos(obliquity), -sin(obliquity)],
		[0, sin(obliquity),  cos(obliquity)]
    ])
    return dot(obliquity_matrix, vector)

def calculate_partial_derivatives(RA, DEC, delta, r2, r2dot, t, obliquity, sun_vector, file):
    """Calculates all partial derivatives for an observation"""
    components = [*r2, *r2dot]
    names = [ 
        "x",
        "y",
        "z",
        "xdot",
        "ydot",
        "zdot"
    ]

    pds_RA = {}
    pds_DEC = {}
    for i in range(6):
        file.write('PARTIAL DERIVATIVES FOR ' + names[i] + "\n")

        components[i] = components[i] * (1+delta)
        file.write(names[i] + " plus " + str(components[i]) + "\n")

        var_plus_RA, var_plus_DEC = ephemeris_generator_2( components, t, obliquity, sun_vector )
        file.write("RA-Plus " + str(var_plus_RA) + "\n")
        file.write("DEC-Plus " + str(var_plus_DEC) + "\n")

        components[i] = components[i] / (1+delta) # back to normal

        components[i] = components[i] * (1-delta)
        file.write(names[i] + " minus " + str(components[i]) + "\n")

        var_minus_RA, var_minus_DEC = ephemeris_generator_2( components, t, obliquity, sun_vector )
        file.write("RA-Minus " + str(var_minus_RA) + "\n")
        file.write("DEC-Minus " + str(var_minus_DEC) + "\n")

        components[i] = components[i] / (1-delta) # back to normal

        # find partial derivative for the current variable
        pd_RA = (var_plus_RA - var_minus_RA) / (delta * 2 * components[i])
        file.write("Partial derivative RA" + str(pd_RA) + "\n")

        pd_DEC = (var_plus_DEC - var_minus_DEC) / (delta * 2 * components[i])
        file.write("Partial derivative DEC" + str(pd_DEC) + "\n")

        pds_RA[ names[i] ] = pd_RA
        pds_DEC[ names[i] ] = pd_DEC

        file.write("*"*60 + "\n")
    return pds_RA, pds_DEC

def write_dict(file, Dict, dict_name, dline = True):
    """Function that takes in dictionary and file name and prints
    all values in the dictionary on separate lines in the file"""
    file.write(dict_name + "\n")
    for key, value in Dict.items():
        file.write(str(key) + ": ")
        file.write(str(value) + "\n")
    if dline:
        file.write("*"*60 + "\n")

def civil_date_to_julian_day(Y, M, D, UT):
    JDay = (367*Y)
    JDay -= int( (7 * ( Y + int( (M+9) /12 )))/4)
    JDay += int(275 * M /9)
    JDay += D
    JDay += 1721013.5 
    JDay += (UT/24)
    return JDay
