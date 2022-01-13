from ODFunctions import *
from math import radians
from timeit import timeit

k = 0.01720209895                   # Gaussian gravitational constant
cAU = 173.144632674240              # speed of light in au/ (mean solar) day
obliquity = radians(84381.448/3600) # Earth's obliquity

# get 3 data points that the user chooses
observations_chosen = get_data_points("chaconInput.txt", 3)

# convert all data points into values Python understands
obs1 = convert_data_line_to_numbers( observations_chosen[0] )
obs2 = convert_data_line_to_numbers( observations_chosen[1] )
obs3 = convert_data_line_to_numbers( observations_chosen[2] )

# Save all tau intervals into a dictionary to then pass on to other functions
taus = calculate_taus( k, obs1["time"], obs2["time"], obs3["time"] )

rhohats = {}
rhohats[1] = calculate_rhohat( obs1["RA"], obs1["DEC"] )
rhohats[2] = calculate_rhohat( obs2["RA"], obs2["DEC"] )
rhohats[3] = calculate_rhohat( obs3["RA"], obs3["DEC"] )

# get all sun vectors in a list to then calculate D values
sun_vectors = { 1:obs1["sun vector"], 2:obs2["sun vector"], 3:obs3["sun vector"] }

# dictionary that holds all D coefficients
Ds = calculate_Ds( sun_vectors, rhohats )

# use Scalar Equation of Lagrange to get initial guesses for r2 and rho2
r20, rho20 = SEL(taus, sun_vectors[2], rhohats[2], Ds)

# using the r20 from SEL, get the scalar ranges or rhos into a dictionary
scalar_ranges = calculate_scalar_ranges(taus, r20, [0, 0, 0], Ds, "2")

# position vectors
r1 = scalar_ranges[1]*rhohats[1] - sun_vectors[1]
r20 = scalar_ranges[2]*rhohats[2] - sun_vectors[2]
r3 = scalar_ranges[3]*rhohats[3] - sun_vectors[3]

# initial guess for velocity vector r2dot
r2dot0 = calculate_r2dot(taus, r1, r20, r3, [0, 0, 0], "2")

# adjust for light travel time
taus = calculate_new_taus( k, [ obs1["time"], obs2["time"], obs3["time"] ], scalar_ranges, cAU )

# ITERATION 1
# get new and improved scalar ranges using f and g functions
scalar_ranges = calculate_scalar_ranges(taus, r20, r2dot0, Ds, "f")

# same for position vectors and velocity vector
r1 = scalar_ranges[1] * rhohats[1] - sun_vectors[1]
r2 = scalar_ranges[2] * rhohats[2] - sun_vectors[2]
r3 = scalar_ranges[3] * rhohats[3] - sun_vectors[3]
r2dot = calculate_r2dot(taus, r1, r2, r3, r2dot0, "f")

tol = 1E-12 
# while my position vector has not converged...
while abs( getMag(r2) - getMag(r20) ) > tol:
    # save my current r2 to the variable that holded the previous r2
    # and do the same for r2dot
    r20 = r2
    r2dot0 = r2dot

    # before calculating new values, adjust for light travel time
    taus = calculate_new_taus( k, [obs1["time"], obs2["time"], obs3["time"]], scalar_ranges, cAU )

    # get new and improved values
    scalar_ranges = calculate_scalar_ranges(taus, r20, r2dot0, Ds, "f")
    r1 = scalar_ranges[1]*rhohats[1] - sun_vectors[1]
    r2 = scalar_ranges[2]*rhohats[2] - sun_vectors[2]
    r3 = scalar_ranges[3]*rhohats[3] - sun_vectors[3]
    r2dot = calculate_r2dot(taus, r1, r2, r3, r2dot0, "f")

# convert to cartesian ecliptic coordinates
r2 = equatorialToEcliptic(obliquity, r2)
r2dot = equatorialToEcliptic(obliquity, r2dot)

# find orbital elements
eccentricity, semiMajorA, inclination, ascendingNode, argOfPerihelion, initMeanAnomaly = get_orbital_elements(r2, r2dot)

# Convert July 25, 2020 at 16:00:00 UT to Julian Day
july_25_2020 = civil_date_to_julian_day( 2020, 7, 25, 16 )

# calculate mean anomaly for July 25 2020 at 16 UT
meanAnomaly = calculate_mean_anom(july_25_2020, 2457852.08061, semiMajorA) % (2*pi)

# DISPLAY RESULTS

# find length of longest string to be displayed
length = len("argument of perihelion")

elements = [ eccentricity, semiMajorA, degrees(inclination), 
            degrees(ascendingNode), degrees(argOfPerihelion), degrees(meanAnomaly) ]

strings = [ "eccentricity", "semi major axis", "inclination", 
            "ascending node", "argument of perihelion", "mean anomaly"]

units = [ "", "AU", "degrees", "degrees", "degrees", "degrees on July 15 2020 16:00 UTC" ]

print("\nposition vector: [", *r2, "] =", getMag(r2), "AU")
print("velocity vector: [", *r2dot, "] =", getMag(r2dot)*k, "AU/day")

print("\nORBITAL ELEMENTS")
for i in range(6):
    print( strings[i] + ":", " " * ( length - len(strings[i]) ), elements[i], units[i] )