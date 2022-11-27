"""generate signals purely from a single SPICE netlist"""

# change oscillator's weight before summation with op amps instead of resistors
# so as to not impact frequency response

# control phase with op amp integrator
# https://electronics.stackexchange.com/questions/308461/how-to-electrically-shift-the-phase-of-a-signal

# add final offset with yet another op amp


# TODO: map search algo to netlist updates for MCExploit
#       yields best netlist (contains det args in the form of parameters to the electric components)
