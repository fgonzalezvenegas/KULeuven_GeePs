# KULeuven_GeePs
EVmodel+Occupancy


Test cases and useful functions to check occupancy profiles with Python 3

occupancy.py and occupancyContinuous.py are the base functions that create occupancy profiles from Strobe. These were sent by Christina.

run_profiles.py is the script that calls occupancy().py functions to run profiles and saves them in a .csv. Sent by Christina

run_profiles_evmodel.py is a script that runs the EV model from Felipe (see Gonzalez Venegas, Petit, Perez, "Plug-in behavior of electric vehicles users: Insights from a large-scale trial and impacts for grid integration studies", eTransportation, Aug 2021), using the occupancy schedules from KULeuven.

analyze_load and analyze_powerflow.py do some plots on grid stability indicators and household/transformer load profiles by technology.

util_occupancy.py has some functions to create a occupancy schedule useful for EV model.


