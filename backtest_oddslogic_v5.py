# backtest_oddslogic_v5.py — six-race comparison vs OddsLogic closing lines.
import numpy as np
import pandas as pd

ATLANTA = [
    ("Tyler Reddick", "Ryan Blaney",       110, -130),
    ("Tyler Reddick", "Chase Elliott",     125, -145),
    ("Tyler Reddick", "William Byron",    -140,  120),
    ("Tyler Reddick", "Kyle Larson",      -120,  100),
    ("Ryan Blaney",   "Chase Elliott",    -120,  100),
    ("Ryan Blaney",   "William Byron",    -145,  125),
    ("Ryan Blaney",   "Kyle Larson",      -130,  110),
    ("Chase Elliott", "William Byron",    -135,  115),
    ("Chase Elliott", "Kyle Larson",      -135,  115),
    ("William Byron", "Kyle Larson",      -120,  100),
    ("Denny Hamlin",  "Christopher Bell", -110, -110),
    ("Denny Hamlin",  "Joey Logano",       110, -130),
    ("Denny Hamlin",  "Carson Hocevar",   -105, -115),
    ("Denny Hamlin",  "Chase Briscoe",    -125,  105),
    ("Christopher Bell","Joey Logano",     110, -130),
    ("Christopher Bell","Carson Hocevar", -110, -110),
    ("Christopher Bell","Chase Briscoe",  -115, -105),
    ("Joey Logano",   "Carson Hocevar",   -140,  120),
    ("Joey Logano",   "Chase Briscoe",    -130,  110),
    ("Carson Hocevar","Chase Briscoe",    -110, -110),
    ("Brad Keselowski","Austin Cindric",   100, -120),
    ("Brad Keselowski","Chris Buescher",  -125,  105),
    ("Brad Keselowski","Bubba Wallace",   -115, -105),
    ("Brad Keselowski","Ross Chastain",   -125,  105),
    ("Austin Cindric","Chris Buescher",   -130,  110),
    ("Austin Cindric","Bubba Wallace",    -125,  105),
    ("Austin Cindric","Ross Chastain",    -135,  115),
    ("Chris Buescher","Bubba Wallace",    -115, -105),
    ("Chris Buescher","Ross Chastain",    -140,  120),
    ("Bubba Wallace", "Ross Chastain",    -115, -105),
    ("Ricky Stenhouse Jr", "Ty Gibbs",    -105, -125),
    ("Ricky Stenhouse Jr", "Shane Van Gisbergen", -150,  120),
    ("Ricky Stenhouse Jr", "Daniel Suarez",   115, -145),
    ("Ty Gibbs",      "Shane Van Gisbergen", -130,  100),
    ("Shane Van Gisbergen", "Daniel Suarez", 120, -150),
]

WILKESBORO = [
    ("Denny Hamlin",      "Kyle Larson",        -190,  165),
    ("Denny Hamlin",      "Ryan Blaney",        -115, -105),
    ("Denny Hamlin",      "Christopher Bell",    120, -140),
    ("Denny Hamlin",      "William Byron",      -220,  190),
    ("Kyle Larson",       "Ryan Blaney",         145, -165),
    ("Kyle Larson",       "Christopher Bell",    195, -225),
    ("Kyle Larson",       "William Byron",      -135,  115),
    ("Ryan Blaney",       "William Byron",      -195,  170),
    ("Christopher Bell",  "William Byron",      -270,  230),
    ("Chase Elliott",     "Joey Logano",         140, -160),
    ("Chase Elliott",     "Tyler Reddick",      -145,  125),
    ("Chase Elliott",     "Ty Gibbs",            140, -160),
    ("Chase Elliott",     "Chase Briscoe",      -140,  120),
    ("Joey Logano",       "Tyler Reddick",      -185,  160),
    ("Joey Logano",       "Ty Gibbs",            105, -125),
    ("Joey Logano",       "Chase Briscoe",      -190,  165),
    ("Tyler Reddick",     "Ty Gibbs",            160, -185),
    ("Tyler Reddick",     "Chase Briscoe",      -140,  120),
    ("Ty Gibbs",          "Chase Briscoe",      -210,  180),
    ("Carson Hocevar",    "Ryan Preece",        -170,  150),
    ("Carson Hocevar",    "Chris Buescher",     -135,  115),
    ("Carson Hocevar",    "Ross Chastain",      -180,  155),
    ("Carson Hocevar",    "Bubba Wallace",       120, -140),
    ("Ryan Preece",       "Chris Buescher",      110, -130),
    ("Ryan Preece",       "Ross Chastain",      -165,  145),
    ("Ryan Preece",       "Bubba Wallace",       150, -170),
    ("Chris Buescher",    "Ross Chastain",      -150,  130),
    ("Chris Buescher",    "Bubba Wallace",       130, -150),
    ("Ross Chastain",     "Bubba Wallace",       170, -195),
    ("Brad Keselowski",   "Austin Cindric",      175, -200),
    ("Brad Keselowski",   "Josh Berry",         -125, -105),
    ("Brad Keselowski",   "Daniel Suarez",      -125, -105),
    ("Austin Cindric",    "Josh Berry",         -210,  170),
    ("Austin Cindric",    "Daniel Suarez",      -190,  155),
    ("Josh Berry",        "Daniel Suarez",       105, -135),
    ("Austin Dillon",     "Alex Bowman",        -140,  110),
]

IOWA = [
    ("Denny Hamlin", "Ryan Blaney", -105, -115),
    ("Denny Hamlin", "Christopher Bell", -115, -105),
    ("Denny Hamlin", "Kyle Larson", -155, 135),
    ("Denny Hamlin", "William Byron", -180, 155),
    ("Ryan Blaney", "Christopher Bell", -120, 100),
    ("Ryan Blaney", "Kyle Larson", -160, 140),
    ("Ryan Blaney", "William Byron", -180, 155),
    ("Christopher Bell", "Kyle Larson", -150, 130),
    ("Christopher Bell", "William Byron", -170, 150),
    ("Kyle Larson", "William Byron", -135, 115),
    ("Joey Logano", "Chase Briscoe", -145, 125),
    ("Joey Logano", "Chase Elliott", -150, 130),
    ("Joey Logano", "Ty Gibbs", -145, 125),
    ("Joey Logano", "Tyler Reddick", -170, 150),
    ("Chase Briscoe", "Chase Elliott", -115, -105),
    ("Chase Briscoe", "Ty Gibbs", -110, -110),
    ("Chase Briscoe", "Tyler Reddick", -145, 125),
    ("Chase Elliott", "Ty Gibbs", -105, -115),
    ("Chase Elliott", "Tyler Reddick", -140, 120),
    ("Ty Gibbs", "Tyler Reddick", -145, 125),
    ("Bubba Wallace", "Carson Hocevar", -180, 155),
    ("Bubba Wallace", "Brad Keselowski", -145, 125),
    ("Bubba Wallace", "Chris Buescher", -210, 180),
    ("Bubba Wallace", "Ross Chastain", -235, 200),
    ("Carson Hocevar", "Brad Keselowski", 120, -140),
    ("Carson Hocevar", "Chris Buescher", -140, 120),
    ("Carson Hocevar", "Ross Chastain", -165, 145),
    ("Brad Keselowski", "Chris Buescher", -180, 155),
    ("Brad Keselowski", "Ross Chastain", -200, 175),
    ("Chris Buescher", "Ross Chastain", -135, 115),
]

LOUDON = [
    ("Ryan Blaney", "Christopher Bell", -170, 150),
    ("Ryan Blaney", "Joey Logano", -200, 175),
    ("Ryan Blaney", "Denny Hamlin", -185, 160),
    ("Ryan Blaney", "Chase Briscoe", -270, 230),
    ("Christopher Bell", "Joey Logano", -110, -110),
    ("Christopher Bell", "Denny Hamlin", -125, 105),
    ("Christopher Bell", "Chase Briscoe", -185, 160),
    ("Joey Logano", "Denny Hamlin", -115, -105),
    ("Joey Logano", "Chase Briscoe", -165, 145),
    ("Denny Hamlin", "Chase Briscoe", -170, 150),
    ("Kyle Larson", "William Byron", 145, -165),
    ("Kyle Larson", "Ty Gibbs", 140, -160),
    ("Kyle Larson", "Tyler Reddick", -180, 155),
    ("Kyle Larson", "Chase Elliott", -185, 160),
    ("William Byron", "Ty Gibbs", -135, 115),
    ("William Byron", "Tyler Reddick", -185, 160),
    ("William Byron", "Chase Elliott", -245, 210),
    ("Ty Gibbs", "Tyler Reddick", -170, 150),
    ("Ty Gibbs", "Chase Elliott", -235, 200),
    ("Tyler Reddick", "Chase Elliott", -130, 110),
    ("Josh Berry", "Austin Cindric", -260, 220),
    ("Josh Berry", "Bubba Wallace", -250, 215),
    ("Josh Berry", "Ross Chastain", -265, 225),
    ("Josh Berry", "Chris Buescher", -235, 200),
    ("Austin Cindric", "Bubba Wallace", -105, -115),
    ("Austin Cindric", "Ross Chastain", -130, 110),
    ("Austin Cindric", "Chris Buescher", 100, -120),
    ("Bubba Wallace", "Ross Chastain", -125, 105),
    ("Bubba Wallace", "Chris Buescher", -110, -110),
    ("Ross Chastain", "Chris Buescher", 125, -145),
    ("Brad Keselowski", "Shane Van Gisbergen", 125, -155),
    ("Brad Keselowski", "Carson Hocevar", -110, -120),
    ("Brad Keselowski", "Alex Bowman", -120, -110),
    ("Shane Van Gisbergen", "Carson Hocevar", -155, 125),
    ("Shane Van Gisbergen", "Alex Bowman", -145, 115),
]

DAYTONA = [
    ("Ryan Blaney", "Joey Logano", -110, -110),
    ("Ryan Blaney", "William Byron", -120, 100),
    ("Ryan Blaney", "Tyler Reddick", -125, 105),
    ("Ryan Blaney", "Christopher Bell", -130, 110),
    ("Joey Logano", "William Byron", -125, 105),
    ("Joey Logano", "Tyler Reddick", -125, 105),
    ("Joey Logano", "Christopher Bell", -130, 110),
    ("William Byron", "Tyler Reddick", -115, -105),
    ("William Byron", "Christopher Bell", -120, 100),
    ("Tyler Reddick", "Christopher Bell", -110, -110),
    ("Chase Elliott", "Kyle Larson", -120, 100),
    ("Chase Elliott", "Carson Hocevar", -115, -105),
    ("Chase Elliott", "Austin Cindric", -105, -115),
    ("Chase Elliott", "Bubba Wallace", -115, -105),
    ("Kyle Larson", "Carson Hocevar", -105, -115),
    ("Kyle Larson", "Austin Cindric", 105, -125),
    ("Kyle Larson", "Bubba Wallace", -105, -115),
    ("Carson Hocevar", "Austin Cindric", 100, -120),
    ("Carson Hocevar", "Bubba Wallace", -110, -110),
    ("Austin Cindric", "Bubba Wallace", -120, 100),
    ("Chase Briscoe", "Denny Hamlin", -110, -110),
    ("Chase Briscoe", "Chris Buescher", -110, -110),
    ("Chase Briscoe", "Brad Keselowski", -105, -115),
    ("Chase Briscoe", "Ty Gibbs", -120, 100),
    ("Denny Hamlin", "Chris Buescher", -110, -110),
    ("Denny Hamlin", "Brad Keselowski", -105, -115),
    ("Denny Hamlin", "Ty Gibbs", -120, 100),
    ("Chris Buescher", "Brad Keselowski", -105, -115),
    ("Chris Buescher", "Ty Gibbs", -120, 100),
    ("Brad Keselowski", "Ty Gibbs", -125, 105),
]

DARLINGTON = [
    ("Denny Hamlin", "Tyler Reddick", -110, -110),
    ("Denny Hamlin", "Chase Briscoe", -135, 115),
    ("Denny Hamlin", "Kyle Larson", -150, 130),
    ("Denny Hamlin", "Ryan Blaney", -160, 140),
    ("Tyler Reddick", "Chase Briscoe", -135, 115),
    ("Tyler Reddick", "Kyle Larson", -150, 130),
    ("Tyler Reddick", "Ryan Blaney", -160, 140),
    ("Chase Briscoe", "Kyle Larson", -125, 105),
    ("Chase Briscoe", "Ryan Blaney", -135, 115),
    ("Kyle Larson", "Ryan Blaney", -120, 100),
    ("William Byron", "Christopher Bell", -120, 100),
    ("William Byron", "Joey Logano", -165, 145),
    ("William Byron", "Ty Gibbs", -120, 100),
    ("William Byron", "Chase Elliott", -160, 140),
    ("Christopher Bell", "Joey Logano", -155, 135),
    ("Christopher Bell", "Ty Gibbs", -115, -105),
    ("Christopher Bell", "Chase Elliott", -150, 130),
    ("Joey Logano", "Ty Gibbs", 130, -150),
    ("Joey Logano", "Chase Elliott", -110, -110),
    ("Ty Gibbs", "Chase Elliott", -145, 125),
    ("Bubba Wallace", "Chris Buescher", -120, 100),
    ("Bubba Wallace", "Brad Keselowski", 100, -120),
    ("Bubba Wallace", "Ross Chastain", -210, 180),
    ("Bubba Wallace", "Erik Jones", -150, 130),
    ("Chris Buescher", "Brad Keselowski", 110, -130),
    ("Chris Buescher", "Ross Chastain", -200, 175),
    ("Chris Buescher", "Erik Jones", -140, 120),
    ("Brad Keselowski", "Ross Chastain", -210, 180),
    ("Brad Keselowski", "Erik Jones", -155, 135),
    ("Ross Chastain", "Erik Jones", 130, -150),
]

CHICAGOLAND = [
    ("Denny Hamlin", "Tyler Reddick", -155, 125),
    ("Denny Hamlin", "Kyle Larson", -143, 114),
    ("Denny Hamlin", "Christopher Bell", -233, 181),
    ("Denny Hamlin", "Ryan Blaney", -260, 200),
    ("Tyler Reddick", "Ryan Blaney", -210, 170),
    ("Kyle Larson", "Christopher Bell", -165, 135),
    ("Kyle Larson", "Ryan Blaney", -210, 170),
    ("William Byron", "Chase Briscoe", -120, -110),
    ("William Byron", "Chris Buescher", -135, 105),
    ("Chase Elliott", "Chase Briscoe", 115, -145),
    ("Chase Elliott", "Ty Gibbs", 105, -135),
    ("Chase Elliott", "Chris Buescher", -135, 105),
    ("Chase Briscoe", "Chris Buescher", -125, -105),
    ("Chase Briscoe", "Carson Hocevar", -200, 160),
    ("Ty Gibbs", "Carson Hocevar", -190, 155),
    ("Ty Gibbs", "Chris Buescher", -135, 105),
    ("Carson Hocevar", "Bubba Wallace", 160, -200),
    ("Carson Hocevar", "Joey Logano", -200, 160),
    ("Carson Hocevar", "Brad Keselowski", 115, -145),
    ("Bubba Wallace", "Brad Keselowski", -140, 110),
    ("Erik Jones", "Daniel Suarez", -140, 110),
    ("Erik Jones", "Corey Heim", 185, -225),
    ("Erik Jones", "Alex Bowman", 110, -140),
    ("Daniel Suarez", "Corey Heim", 175, -215),
    ("Corey Heim", "Alex Bowman", -175, 145),
    ("Austin Cindric", "Ryan Preece", -120, -110),
    ("Shane Van Gisbergen", "Ryan Preece", 150, -180),
    ("Zane Smith", "Riley Herbst", -125, -105),
    ("Riley Herbst", "Connor Zilisch", -165, 135),
    ("Michael McDowell", "Austin Dillon", 150, -180),
    ("Josh Berry", "John Hunter Nemechek", 150, -180),
]

SONOMA = [
    ("Connor Zilisch", "Tyler Reddick", 100, -130),
    ("Connor Zilisch", "Kyle Larson", 130, -160),
    ("Connor Zilisch", "Ryan Blaney", -105, -125),
    ("Connor Zilisch", "Michael McDowell", -120, -110),
    ("Tyler Reddick", "Kyle Larson", 115, -145),
    ("Tyler Reddick", "Ryan Blaney", -175, 145),
    ("Tyler Reddick", "Michael McDowell", -160, 130),
    ("Kyle Larson", "Ryan Blaney", -200, 160),
    ("Kyle Larson", "Michael McDowell", -215, 175),
    ("Ryan Blaney", "Michael McDowell", -125, -105),
    ("Ty Gibbs", "Chase Briscoe", -150, 120),
    ("Ty Gibbs", "AJ Allmendinger", -145, 115),
    ("Chase Briscoe", "Chris Buescher", -155, 125),
    ("Chase Briscoe", "Chase Elliott", -190, 155),
    ("Chase Briscoe", "William Byron", -130, 100),
    ("Chris Buescher", "AJ Allmendinger", 125, -155),
    ("Chris Buescher", "William Byron", 100, -130),
    ("AJ Allmendinger", "Chase Elliott", -190, 155),
    ("William Byron", "Chase Elliott", -190, 155),
    ("Christopher Bell", "Chase Elliott", 145, -175),
    ("Christopher Bell", "Daniel Suarez", -130, 100),
    ("Christopher Bell", "Denny Hamlin", 145, -175),
    ("Daniel Suarez", "Ross Chastain", 160, -200),
    ("Daniel Suarez", "Denny Hamlin", 160, -200),
    ("Ross Chastain", "Carson Hocevar", -115, -115),
    ("Ross Chastain", "Denny Hamlin", -130, 100),
    ("Ryan Preece", "Bubba Wallace", -130, 100),
    ("Ryan Preece", "Austin Hill", -150, 120),
    ("Ryan Preece", "Zane Smith", -150, 120),
    ("Bubba Wallace", "Austin Hill", -180, 150),
    ("Bubba Wallace", "Zane Smith", -165, 135),
    ("Austin Hill", "Zane Smith", -105, -125),
    ("Joey Logano", "Alex Bowman", -215, 175),
    ("Joey Logano", "Austin Cindric", -190, 155),
    ("Alex Bowman", "Austin Cindric", -130, 100),
    ("Riley Herbst", "Todd Gilliland", 170, -210),
    ("Riley Herbst", "John Hunter Nemechek", 100, -130),
]

SAN_DIEGO = [
    ("Connor Zilisch", "Tyler Reddick", -160, 130),
    ("Connor Zilisch", "Michael McDowell", -130, 100),
    ("Connor Zilisch", "Ty Gibbs", -155, 125),
    ("Connor Zilisch", "Kyle Larson", -135, 105),
    ("Tyler Reddick", "Michael McDowell", 110, -140),
    ("Tyler Reddick", "Ty Gibbs", 120, -150),
    ("Tyler Reddick", "Kyle Larson", -105, -125),
    ("Michael McDowell", "Ty Gibbs", -115, -115),
    ("Michael McDowell", "Kyle Larson", -140, 110),
    ("Ty Gibbs", "Kyle Larson", -115, -115),
    ("Chris Buescher", "AJ Allmendinger", -115, -115),
    ("Chris Buescher", "Christopher Bell", -200, 160),
    ("Chris Buescher", "Chase Elliott", -120, -110),
    ("Chris Buescher", "Chase Briscoe", -110, -120),
    ("AJ Allmendinger", "Christopher Bell", -205, 165),
    ("AJ Allmendinger", "Chase Elliott", -140, 110),
    ("AJ Allmendinger", "Chase Briscoe", -115, -115),
    ("Christopher Bell", "Chase Elliott", 170, -210),
    ("Christopher Bell", "Chase Briscoe", 170, -210),
    ("Chase Elliott", "Chase Briscoe", -110, -120),
    ("Ryan Blaney", "Daniel Suarez", -165, 135),
    ("Ross Chastain", "Kevin Magnussen", -150, 120),
    ("Ross Chastain", "Denny Hamlin", -140, 110),
    ("Kevin Magnussen", "Denny Hamlin", -115, -115),
    ("Alex Bowman", "Corey Heim", 130, -160),
    ("Corey Heim", "Joey Logano", -190, 155),
    ("Austin Cindric", "Joey Logano", 135, -165),
]

NASHVILLE = [
    ("Denny Hamlin", "Tyler Reddick", -175, 145),
    ("Denny Hamlin", "Christopher Bell", -150, 120),
    ("Denny Hamlin", "Ryan Blaney", -180, 150),
    ("Denny Hamlin", "Kyle Larson", -180, 150),
    ("Tyler Reddick", "Christopher Bell", -105, -125),
    ("Tyler Reddick", "Ryan Blaney", -155, 125),
    ("Tyler Reddick", "Kyle Larson", -135, 105),
    ("Christopher Bell", "Ryan Blaney", -150, 120),
    ("Christopher Bell", "Kyle Larson", -150, 120),
    ("Ryan Blaney", "Kyle Larson", -130, 100),
    ("Chase Briscoe", "William Byron", 100, -130),
    ("Chase Briscoe", "Chase Elliott", -125, -105),
    ("Chase Briscoe", "Ty Gibbs", 130, -160),
    ("Chase Briscoe", "Joey Logano", -160, 130),
    ("William Byron", "Chase Elliott", -125, -105),
    ("William Byron", "Ty Gibbs", 145, -175),
    ("William Byron", "Joey Logano", -165, 135),
    ("Chase Elliott", "Ty Gibbs", 135, -165),
    ("Chase Elliott", "Joey Logano", -200, 160),
    ("Carson Hocevar", "Ross Chastain", 140, -170),
    ("Carson Hocevar", "Chris Buescher", -145, 115),
    ("Carson Hocevar", "Bubba Wallace", -140, 110),
    ("Carson Hocevar", "Brad Keselowski", -115, -115),
    ("Ross Chastain", "Chris Buescher", -165, 135),
    ("Ross Chastain", "Bubba Wallace", -165, 135),
    ("Ross Chastain", "Brad Keselowski", -165, 135),
    ("Chris Buescher", "Bubba Wallace", 105, -135),
    ("Chris Buescher", "Brad Keselowski", 103, -133),
    ("Bubba Wallace", "Brad Keselowski", 105, -135),
    ("Daniel Suarez", "Austin Cindric", 105, -135),
    ("Daniel Suarez", "Alex Bowman", -110, -120),
    ("Daniel Suarez", "Josh Berry", -175, 145),
    ("Austin Cindric", "Alex Bowman", -130, 100),
    ("Austin Cindric", "Josh Berry", -200, 160),
    ("Alex Bowman", "Josh Berry", -170, 140),
    ("Ryan Preece", "Zane Smith", -160, 130),
    ("Ryan Preece", "Erik Jones", 115, -145),
    ("Ryan Preece", "Shane Van Gisbergen", -135, 105),
    ("Zane Smith", "Erik Jones", 125, -155),
    ("Zane Smith", "Corey Heim", -105, -125),
    ("Shane Van Gisbergen", "Erik Jones", 160, -200),
    ("Shane Van Gisbergen", "Corey Heim", 130, -160),
    ("Connor Zilisch", "Austin Dillon", -125, -105),
    ("Connor Zilisch", "AJ Allmendinger", 100, -130),
]

POCONO = [
    ("Denny Hamlin", "Kyle Larson", -190, 155),
    ("Denny Hamlin", "Tyler Reddick", -190, 150),
    ("Kyle Larson", "Tyler Reddick", -150, 120),
    ("Tyler Reddick", "Christopher Bell", -215, 175),
    ("Tyler Reddick", "Ryan Blaney", -170, 140),
    ("Christopher Bell", "Ryan Blaney", 125, -155),
    ("Chase Briscoe", "William Byron", -125, -105),
    ("Chase Briscoe", "Chase Elliott", -125, -105),
    ("Chase Briscoe", "Ty Gibbs", 115, -145),
    ("Chase Briscoe", "Carson Hocevar", -150, 120),
    ("William Byron", "Chase Elliott", -135, 105),
    ("William Byron", "Ty Gibbs", 100, -130),
    ("William Byron", "Carson Hocevar", -150, 120),
    ("Chase Elliott", "Ty Gibbs", 140, -170),
    ("Chase Elliott", "Carson Hocevar", -140, 110),
    ("Ty Gibbs", "Carson Hocevar", -175, 145),
    ("Bubba Wallace", "Chris Buescher", 230, -290),
    ("Bubba Wallace", "Joey Logano", 110, -140),
    ("Bubba Wallace", "Ross Chastain", -115, -115),
    ("Bubba Wallace", "Brad Keselowski", 100, -130),
    ("Chris Buescher", "Joey Logano", -260, 200),
    ("Chris Buescher", "Brad Keselowski", -280, 220),
    ("Joey Logano", "Ross Chastain", -145, 115),
    ("Joey Logano", "Brad Keselowski", -130, 100),
    ("Ross Chastain", "Brad Keselowski", 105, -135),
    ("Erik Jones", "Daniel Suarez", -125, -105),
    ("Erik Jones", "Alex Bowman", -260, 200),
    ("Daniel Suarez", "Alex Bowman", -215, 175),
    ("Austin Cindric", "Alex Bowman", -160, 130),
    ("Ryan Preece", "Zane Smith", 125, -155),
    ("Ryan Preece", "Shane Van Gisbergen", -130, 100),
    ("Ryan Preece", "Connor Zilisch", -130, 100),
    ("Zane Smith", "Shane Van Gisbergen", -170, 140),
    ("Zane Smith", "Connor Zilisch", -160, 130),
    ("Shane Van Gisbergen", "Connor Zilisch", 100, -130),
    ("Josh Berry", "Riley Herbst", -125, -105),
    ("Josh Berry", "Michael McDowell", 120, -150),
    ("Michael McDowell", "Ricky Stenhouse Jr", -115, -115),
    ("Austin Dillon", "AJ Allmendinger", 120, -150),
]

CHARLOTTE = [
    ("Denny Hamlin", "Tyler Reddick", -135, 105),
    ("Denny Hamlin", "Kyle Larson", -160, 130),
    ("Denny Hamlin", "Ryan Blaney", -180, 150),
    ("Denny Hamlin", "Christopher Bell", -170, 140),
    ("Tyler Reddick", "Kyle Larson", -145, 115),
    ("Tyler Reddick", "Ryan Blaney", -155, 125),
    ("Tyler Reddick", "Christopher Bell", -155, 125),
    ("Kyle Larson", "Ryan Blaney", -115, -115),
    ("Kyle Larson", "Christopher Bell", -120, -110),
    ("Ryan Blaney", "Christopher Bell", -105, -125),
    ("William Byron", "Chase Elliott", -150, 120),
    ("William Byron", "Chase Briscoe", -135, 105),
    ("William Byron", "Ty Gibbs", -140, 110),
    ("William Byron", "Carson Hocevar", -155, 125),
    ("Chase Elliott", "Chase Briscoe", -130, 100),
    ("Chase Elliott", "Ty Gibbs", -105, -125),
    ("Chase Elliott", "Carson Hocevar", -132, 109),
    ("Chase Briscoe", "Ty Gibbs", 100, -130),
    ("Chase Briscoe", "Carson Hocevar", -125, -105),
    ("Ty Gibbs", "Carson Hocevar", -115, -115),
    ("Chris Buescher", "Bubba Wallace", -130, 100),
    ("Chris Buescher", "Brad Keselowski", -125, 104),
    ("Chris Buescher", "Ross Chastain", -155, 125),
    ("Bubba Wallace", "Joey Logano", -160, 130),
    ("Bubba Wallace", "Brad Keselowski", 117, -142),
    ("Bubba Wallace", "Ross Chastain", -160, 130),
    ("Joey Logano", "Brad Keselowski", 125, -155),
    ("Joey Logano", "Ross Chastain", -105, -125),
    ("Brad Keselowski", "Ross Chastain", -130, 100),
    ("Alex Bowman", "Ryan Preece", -145, 115),
    ("Alex Bowman", "Austin Cindric", -140, 110),
    ("Ryan Preece", "Connor Zilisch", -155, 125),
    ("Austin Cindric", "Connor Zilisch", -175, 145),
    ("Corey Heim", "Erik Jones", -115, -115),
    ("Corey Heim", "Daniel Suarez", -125, -105),
    ("Erik Jones", "Daniel Suarez", -125, -105),
    ("AJ Allmendinger", "Shane Van Gisbergen", -125, -105),
    ("AJ Allmendinger", "Riley Herbst", -140, 110),
    ("Shane Van Gisbergen", "Zane Smith", -125, -105),
    ("Riley Herbst", "Zane Smith", 100, -130),
    ("Todd Gilliland", "Ricky Stenhouse Jr", 155, -190),
    ("Noah Gragson", "John Hunter Nemechek", 120, -150),
    ("Cole Custer", "Ty Dillon", -130, 100),
]

TEXAS = [
    ("Denny Hamlin", "Tyler Reddick", -105, -125),
    ("Denny Hamlin", "Kyle Larson", -140, 110),
    ("Denny Hamlin", "Ryan Blaney", -220, 180),
    ("Denny Hamlin", "Christopher Bell", -170, 140),
    ("Tyler Reddick", "Kyle Larson", -145, 115),
    ("Tyler Reddick", "Christopher Bell", -150, 130),
    ("Kyle Larson", "Ryan Blaney", -200, 160),
    ("Kyle Larson", "Christopher Bell", -140, 110),
    ("Ryan Blaney", "Christopher Bell", 115, -145),
    ("William Byron", "Chase Elliott", -115, -115),
    ("William Byron", "Chase Briscoe", -110, -120),
    ("William Byron", "Joey Logano", -210, 170),
    ("William Byron", "Ty Gibbs", -120, -110),
    ("Chase Elliott", "Chase Briscoe", -130, 100),
    ("Chase Elliott", "Joey Logano", -200, 160),
    ("Chase Elliott", "Ty Gibbs", -125, -105),
    ("Chase Briscoe", "Joey Logano", -200, 160),
    ("Chase Briscoe", "Ty Gibbs", -115, -115),
    ("Joey Logano", "Ty Gibbs", 150, -180),
    ("Carson Hocevar", "Bubba Wallace", -165, 135),
    ("Carson Hocevar", "Chris Buescher", 105, -135),
    ("Carson Hocevar", "Brad Keselowski", -165, 135),
    ("Bubba Wallace", "Chris Buescher", 145, -175),
    ("Bubba Wallace", "Brad Keselowski", -125, -105),
    ("Bubba Wallace", "Ross Chastain", -135, 105),
    ("Chris Buescher", "Brad Keselowski", -200, 160),
    ("Chris Buescher", "Ross Chastain", -180, 150),
    ("Brad Keselowski", "Ross Chastain", -145, 115),
    ("Austin Cindric", "Ryan Preece", -115, -115),
    ("Austin Cindric", "Alex Bowman", -125, -105),
    ("Austin Cindric", "Daniel Suarez", -125, -105),
    ("Ryan Preece", "Alex Bowman", -130, 100),
    ("Ryan Preece", "Daniel Suarez", -115, -115),
    ("Alex Bowman", "Daniel Suarez", -120, -110),
    ("Kyle Busch", "Josh Berry", -135, 105),
    ("Kyle Busch", "Corey Heim", -105, -125),
    ("Josh Berry", "Corey Heim", 105, -135),
    ("Noah Gragson", "Erik Jones", 170, -210),
    ("AJ Allmendinger", "Erik Jones", 140, -170),
    ("AJ Allmendinger", "Riley Herbst", -135, 105),
    ("Shane Van Gisbergen", "Riley Herbst", 110, -140),
    ("Ricky Stenhouse Jr", "John Hunter Nemechek", -130, 100),
    ("Zane Smith", "Cole Custer", -155, 125),
]

WATKINS_GLEN = [
    ("Tyler Reddick", "Connor Zilisch", 148, -182),
    ("Tyler Reddick", "Christopher Bell", 115, -145),
    ("Christopher Bell", "William Byron", -190, 155),
    ("Chase Elliott", "Kyle Larson", -170, 140),
    ("Chase Elliott", "Chris Buescher", 185, -225),
    ("William Byron", "Kyle Larson", -200, 160),
    ("William Byron", "Chris Buescher", 100, -130),
    ("Kyle Larson", "Chris Buescher", 190, -230),
    ("AJ Allmendinger", "Chase Briscoe", 140, -170),
    ("AJ Allmendinger", "Ryan Blaney", 150, -180),
    ("AJ Allmendinger", "Ty Gibbs", 165, -205),
    ("Chase Briscoe", "Ryan Blaney", -120, -110),
    ("Chase Briscoe", "Ty Gibbs", 125, -155),
    ("Chase Briscoe", "Michael McDowell", -105, -125),
    ("Ryan Blaney", "Ty Gibbs", 125, -155),
    ("Ryan Blaney", "Michael McDowell", 100, -130),
    ("Ross Chastain", "Carson Hocevar", -180, 150),
    ("Ross Chastain", "Daniel Suarez", -210, 170),
    ("Ross Chastain", "Alex Bowman", -200, 160),
    ("Ross Chastain", "Austin Cindric", -115, -115),
    ("Daniel Suarez", "Carson Hocevar", 105, -135),
    ("Daniel Suarez", "Alex Bowman", -110, -120),
    ("Denny Hamlin", "Joey Logano", 100, -130),
    ("Denny Hamlin", "Kyle Busch", 105, -135),
    ("Denny Hamlin", "Bubba Wallace", -160, 130),
    ("Joey Logano", "Kyle Busch", -110, -120),
    ("Ryan Preece", "Zane Smith", -200, 160),
    ("Brad Keselowski", "Austin Dillon", -155, 125),
    ("Erik Jones", "John Hunter Nemechek", 135, -165),
]

TALLADEGA = [
    ("Ryan Blaney", "Joey Logano", -120, -110),
    ("Ryan Blaney", "Austin Cindric", -130, 100),
    ("Ryan Blaney", "Denny Hamlin", -150, 120),
    ("Ryan Blaney", "Tyler Reddick", -130, 100),
    ("Joey Logano", "Austin Cindric", -125, -105),
    ("Joey Logano", "Denny Hamlin", -135, 105),
    ("Joey Logano", "Tyler Reddick", -130, 100),
    ("Austin Cindric", "Denny Hamlin", -125, -105),
    ("Austin Cindric", "Tyler Reddick", -125, -105),
    ("Denny Hamlin", "Tyler Reddick", -115, -115),
    ("Bubba Wallace", "Carson Hocevar", -140, 110),
    ("Bubba Wallace", "Brad Keselowski", -115, -115),
    ("Bubba Wallace", "Chase Elliott", -115, -115),
    ("Bubba Wallace", "Kyle Larson", -135, 105),
    ("Carson Hocevar", "Brad Keselowski", 110, -140),
    ("Carson Hocevar", "Chase Elliott", 100, -130),
    ("Carson Hocevar", "Kyle Larson", -105, -125),
    ("Brad Keselowski", "Chase Elliott", -115, -115),
    ("Brad Keselowski", "Kyle Larson", -125, -105),
    ("Chase Elliott", "Kyle Larson", -130, 100),
    ("William Byron", "Chase Briscoe", -135, 105),
    ("William Byron", "Chris Buescher", -125, -105),
    ("William Byron", "Ryan Preece", -150, 120),
    ("William Byron", "Christopher Bell", -115, -115),
    ("Chase Briscoe", "Chris Buescher", -105, -125),
    ("Chase Briscoe", "Ryan Preece", -125, -105),
    ("Chase Briscoe", "Kyle Busch", -140, 110),
    ("Chris Buescher", "Ryan Preece", -130, 100),
    ("Chris Buescher", "Christopher Bell", -135, 105),
    ("Ryan Preece", "Kyle Busch", -130, 100),
    ("Ty Gibbs", "Ricky Stenhouse Jr", -115, -115),
    ("Ty Gibbs", "Ross Chastain", -115, -115),
    ("Ty Gibbs", "Erik Jones", -120, -110),
    ("Ricky Stenhouse Jr", "Ross Chastain", -135, 105),
    ("Ricky Stenhouse Jr", "Erik Jones", -135, 105),
    ("Ross Chastain", "Erik Jones", -130, -110),
    ("Noah Gragson", "Austin Dillon", -115, -115),
    ("Noah Gragson", "Zane Smith", -140, 110),
    ("Noah Gragson", "Michael McDowell", -115, -115),
    ("Austin Dillon", "Zane Smith", -105, -125),
    ("Austin Dillon", "Alex Bowman", 105, -135),
    ("Michael McDowell", "Alex Bowman", -120, -110),
    ("Josh Berry", "Daniel Suarez", -120, -110),
    ("Josh Berry", "Connor Zilisch", -145, 115),
    ("Shane Van Gisbergen", "Daniel Suarez", 115, -145),
    ("Shane Van Gisbergen", "Connor Zilisch", -105, -125),
    ("Todd Gilliland", "Riley Herbst", -160, 130),
]

KANSAS = [
    ("Denny Hamlin", "Kyle Larson", -142, 117),
    ("Denny Hamlin", "Christopher Bell", -150, 120),
    ("Denny Hamlin", "Tyler Reddick", 105, -135),
    ("Denny Hamlin", "Ryan Blaney", -165, 135),
    ("Kyle Larson", "Christopher Bell", -120, -101),
    ("Kyle Larson", "Tyler Reddick", 122, -148),
    ("Kyle Larson", "Ryan Blaney", -140, 110),
    ("Christopher Bell", "Tyler Reddick", 115, -145),
    ("Christopher Bell", "Ryan Blaney", -155, 125),
    ("Tyler Reddick", "Ryan Blaney", -180, 150),
    ("Chase Briscoe", "Chase Elliott", -118, -103),
    ("Chase Briscoe", "William Byron", -115, -115),
    ("Chase Briscoe", "Ty Gibbs", -103, -118),
    ("Chase Briscoe", "Bubba Wallace", -130, 100),
    ("Chase Elliott", "William Byron", -115, -115),
    ("Chase Elliott", "Ty Gibbs", 110, -140),
    ("Chase Elliott", "Bubba Wallace", -120, -110),
    ("William Byron", "Ty Gibbs", -101, -120),
    ("William Byron", "Bubba Wallace", -110, -120),
    ("Ty Gibbs", "Bubba Wallace", -140, 110),
    ("Joey Logano", "Chris Buescher", 175, -215),
    ("Joey Logano", "Carson Hocevar", 145, -175),
    ("Joey Logano", "Ryan Preece", 120, -150),
    ("Joey Logano", "Brad Keselowski", 130, -160),
    ("Chris Buescher", "Carson Hocevar", -135, 105),
    ("Chris Buescher", "Ryan Preece", -165, 135),
    ("Chris Buescher", "Brad Keselowski", -165, 135),
    ("Carson Hocevar", "Ryan Preece", -160, 130),
    ("Carson Hocevar", "Brad Keselowski", -155, 125),
    ("Ryan Preece", "Brad Keselowski", 120, -150),
    ("Ross Chastain", "Alex Bowman", 135, -165),
    ("Ross Chastain", "Austin Cindric", -110, -120),
    ("Ross Chastain", "Kyle Busch", -130, 100),
    ("Alex Bowman", "Austin Cindric", -145, 115),
    ("Alex Bowman", "Kyle Busch", -190, 155),
    ("Josh Berry", "Corey Heim", 105, -135),
    ("Josh Berry", "Zane Smith", -115, -115),
    ("Josh Berry", "Erik Jones", -115, -115),
    ("Corey Heim", "Zane Smith", -155, 125),
    ("Corey Heim", "Daniel Suarez", -110, -120),
    ("Zane Smith", "Daniel Suarez", 115, -145),
    ("Connor Zilisch", "Michael McDowell", 115, -145),
    ("Shane Van Gisbergen", "Austin Dillon", -130, 100),
    ("Michael McDowell", "Austin Dillon", -120, -110),
    ("AJ Allmendinger", "Todd Gilliland", -180, 150),
    ("AJ Allmendinger", "Ricky Stenhouse Jr", -140, 110),
]

BRISTOL = [
    ("Kyle Larson", "Denny Hamlin", -130, 100),
    ("Kyle Larson", "Ryan Blaney", -105, -125),
    ("Kyle Larson", "Christopher Bell", -145, 115),
    ("Denny Hamlin", "Ryan Blaney", 130, -160),
    ("Denny Hamlin", "Christopher Bell", -130, 100),
    ("Denny Hamlin", "William Byron", -220, 180),
    ("Christopher Bell", "Ryan Blaney", 145, -175),
    ("Christopher Bell", "William Byron", -205, 165),
    ("Ty Gibbs", "Chase Elliott", -200, 160),
    ("Ty Gibbs", "Chase Briscoe", -155, 125),
    ("Chris Buescher", "Chase Elliott", -105, -125),
    ("Chris Buescher", "Chase Briscoe", 140, -170),
    ("Chris Buescher", "Brad Keselowski", -120, -110),
    ("Chase Elliott", "Chase Briscoe", 135, -165),
    ("Chase Elliott", "Brad Keselowski", -115, 100),
    ("Chase Briscoe", "Brad Keselowski", -175, 145),
    ("Tyler Reddick", "Carson Hocevar", -135, 105),
    ("Tyler Reddick", "Joey Logano", -150, 120),
    ("Tyler Reddick", "Bubba Wallace", -170, 140),
    ("Tyler Reddick", "Ryan Preece", -165, 135),
    ("Carson Hocevar", "Joey Logano", -140, 110),
    ("Carson Hocevar", "Bubba Wallace", -165, 135),
    ("Carson Hocevar", "Ryan Preece", -175, 145),
    ("Joey Logano", "Bubba Wallace", -120, -110),
    ("Joey Logano", "Ryan Preece", -130, 100),
    ("Bubba Wallace", "Ryan Preece", -125, -105),
    ("Kyle Busch", "Ross Chastain", -110, -120),
    ("Kyle Busch", "Austin Cindric", 155, -190),
    ("Kyle Busch", "Josh Berry", 104, -125),
    ("Ross Chastain", "Austin Cindric", -140, 110),
    ("Ross Chastain", "Josh Berry", -180, 150),
    ("Austin Cindric", "Josh Berry", -170, 140),
    ("Zane Smith", "Michael McDowell", -125, -105),
    ("Zane Smith", "Connor Zilisch", -120, -110),
    ("Zane Smith", "Erik Jones", -125, -105),
    ("Michael McDowell", "Connor Zilisch", -155, 125),
    ("Michael McDowell", "Erik Jones", -115, -115),
    ("Connor Zilisch", "Erik Jones", 130, -165),
    ("AJ Allmendinger", "Shane Van Gisbergen", -160, 130),
    ("AJ Allmendinger", "Ricky Stenhouse Jr", 120, -150),
    ("Shane Van Gisbergen", "Ricky Stenhouse Jr", -125, -105),
    ("Noah Gragson", "Daniel Suarez", 100, -130),
]

MARTINSVILLE = [
    ("Ryan Blaney", "Denny Hamlin", 105, -135),
    ("Ryan Blaney", "William Byron", -135, 105),
    ("Ryan Blaney", "Kyle Larson", -140, 110),
    ("Ryan Blaney", "Christopher Bell", -175, 145),
    ("Denny Hamlin", "William Byron", -155, 125),
    ("Denny Hamlin", "Kyle Larson", -135, 105),
    ("Denny Hamlin", "Christopher Bell", -190, 155),
    ("Kyle Larson", "William Byron", -125, -105),
    ("Kyle Larson", "Christopher Bell", -155, 125),
    ("William Byron", "Christopher Bell", -140, 110),
    ("Chase Elliott", "Tyler Reddick", -220, 180),
    ("Chase Elliott", "Chase Briscoe", -215, 175),
    ("Chase Elliott", "Joey Logano", -125, -105),
    ("Chase Elliott", "Ty Gibbs", -155, 125),
    ("Tyler Reddick", "Chase Briscoe", -105, -125),
    ("Tyler Reddick", "Joey Logano", 160, -200),
    ("Tyler Reddick", "Ty Gibbs", 145, -175),
    ("Chase Briscoe", "Joey Logano", 155, -190),
    ("Chase Briscoe", "Ty Gibbs", 150, -180),
    ("Joey Logano", "Ty Gibbs", -180, 150),
    ("Ryan Preece", "Bubba Wallace", 105, -135),
    ("Ryan Preece", "Ross Chastain", -140, 110),
    ("Ryan Preece", "Chris Buescher", -160, 130),
    ("Ryan Preece", "Brad Keselowski", -145, 115),
    ("Bubba Wallace", "Ross Chastain", -165, 135),
    ("Bubba Wallace", "Chris Buescher", -156, 126),
    ("Bubba Wallace", "Brad Keselowski", -135, 105),
    ("Ross Chastain", "Chris Buescher", -140, 110),
    ("Ross Chastain", "Brad Keselowski", 105, -135),
    ("Chris Buescher", "Brad Keselowski", 115, -145),
    ("Carson Hocevar", "Kyle Busch", -205, 165),
    ("Carson Hocevar", "Josh Berry", 125, -155),
    ("Carson Hocevar", "Austin Cindric", 115, -145),
    ("Kyle Busch", "Josh Berry", -115, -115),
    ("Kyle Busch", "Austin Cindric", 190, -230),
    ("Josh Berry", "Austin Cindric", -135, 105),
    ("Daniel Suarez", "Shane Van Gisbergen", 125, -155),
    ("Daniel Suarez", "Michael McDowell", -125, -105),
    ("Daniel Suarez", "Connor Zilisch", -150, 120),
    ("Shane Van Gisbergen", "Michael McDowell", -150, 120),
    ("Shane Van Gisbergen", "Austin Dillon", -105, -125),
    ("Michael McDowell", "Austin Dillon", 115, -145),
    ("Connor Zilisch", "Erik Jones", -115, -115),
    ("Todd Gilliland", "Erik Jones", -145, 115),
    ("Todd Gilliland", "Justin Allgaier", -145, 115),
    ("AJ Allmendinger", "Justin Allgaier", 110, -140),
    ("Noah Gragson", "Zane Smith", 155, -190),
    ("Riley Herbst", "John Hunter Nemechek", 130, -160),
    ("Cole Custer", "Austin Hill", -135, 105),
]

DARLINGTON_SPRING = [
    ("Denny Hamlin", "Kyle Larson", 100, -130),
    ("Denny Hamlin", "Tyler Reddick", 135, -165),
    ("Denny Hamlin", "Chase Briscoe", -120, -110),
    ("Denny Hamlin", "William Byron", -125, -105),
    ("Kyle Larson", "Tyler Reddick", 140, -170),
    ("Kyle Larson", "Chase Briscoe", -165, 135),
    ("Kyle Larson", "William Byron", -150, 120),
    ("Tyler Reddick", "Chase Briscoe", -115, -115),
    ("Tyler Reddick", "William Byron", -170, 140),
    ("Chase Briscoe", "William Byron", -115, -115),
    ("Ryan Blaney", "Christopher Bell", -150, 120),
    ("Ryan Blaney", "Chase Elliott", -170, 140),
    ("Ryan Blaney", "Joey Logano", -175, 145),
    ("Ryan Blaney", "Ross Chastain", -185, 155),
    ("Christopher Bell", "Chase Elliott", -105, -125),
    ("Christopher Bell", "Joey Logano", -200, 160),
    ("Christopher Bell", "Ross Chastain", -175, 145),
    ("Chase Elliott", "Joey Logano", -200, 160),
    ("Chase Elliott", "Ross Chastain", -200, 160),
    ("Joey Logano", "Ross Chastain", -115, -115),
    ("Bubba Wallace", "Ty Gibbs", -130, 100),
    ("Bubba Wallace", "Chris Buescher", -145, 115),
    ("Bubba Wallace", "Kyle Busch", -160, 130),
    ("Bubba Wallace", "Brad Keselowski", -120, -110),
    ("Ty Gibbs", "Chris Buescher", -145, 115),
    ("Ty Gibbs", "Kyle Busch", -190, 155),
    ("Chris Buescher", "Kyle Busch", -120, -110),
    ("Brad Keselowski", "Kyle Busch", -180, 130),
    ("Carson Hocevar", "Erik Jones", -150, 120),
    ("Carson Hocevar", "Ryan Preece", -145, 115),
    ("Carson Hocevar", "Austin Cindric", -115, -115),
    ("Erik Jones", "Ryan Preece", -115, -115),
    ("Erik Jones", "Austin Cindric", 125, -155),
    ("Ryan Preece", "Austin Cindric", 135, -165),
    ("Josh Berry", "Connor Zilisch", -155, 125),
    ("Josh Berry", "Justin Allgaier", -125, -105),
    ("Josh Berry", "Austin Cindric", 135, -165),
    ("Connor Zilisch", "Justin Allgaier", 115, -145),
    ("Connor Zilisch", "Ryan Preece", 145, -175),
    ("John Hunter Nemechek", "AJ Allmendinger", -155, 125),
    ("John Hunter Nemechek", "Shane Van Gisbergen", -155, 125),
    ("AJ Allmendinger", "Zane Smith", 140, -170),
    ("Shane Van Gisbergen", "Zane Smith", 125, -155),
    ("Daniel Suarez", "Noah Gragson", -155, 125),
    ("Daniel Suarez", "Michael McDowell", -180, 150),
    ("Noah Gragson", "Austin Dillon", 145, -175),
    ("Todd Gilliland", "Ricky Stenhouse Jr", -135, 105),
    ("Riley Herbst", "Ricky Stenhouse Jr", -150, 120),
    ("Ty Dillon", "Cole Custer", 100, -130),
]

LAS_VEGAS = [
    ("Kyle Larson", "Denny Hamlin", -115, -105),
    ("Kyle Larson", "Christopher Bell", -115, -105),
    ("Kyle Larson", "William Byron", -155, 135),
    ("Kyle Larson", "Tyler Reddick", -180, 160),
    ("Denny Hamlin", "Christopher Bell", -110, -110),
    ("Denny Hamlin", "William Byron", -160, 140),
    ("Denny Hamlin", "Tyler Reddick", -180, 160),
    ("Christopher Bell", "William Byron", -145, 125),
    ("Christopher Bell", "Tyler Reddick", -180, 160),
    ("William Byron", "Tyler Reddick", -150, 130),
    ("Ryan Blaney", "Joey Logano", -165, 145),
    ("Ryan Blaney", "Chase Elliott", -130, 110),
    ("Ryan Blaney", "Chase Briscoe", -105, -115),
    ("Ryan Blaney", "Ross Chastain", -165, 145),
    ("Joey Logano", "Chase Elliott", 140, -160),
    ("Joey Logano", "Chase Briscoe", 165, -185),
    ("Joey Logano", "Ross Chastain", -115, -105),
    ("Chase Elliott", "Chase Briscoe", 100, -125),
    ("Chase Elliott", "Ross Chastain", -130, 110),
    ("Chase Briscoe", "Ross Chastain", -175, 155),
    ("Bubba Wallace", "Ty Gibbs", 130, -150),
    ("Bubba Wallace", "Chris Buescher", -175, 155),
    ("Bubba Wallace", "Carson Hocevar", -175, 155),
    ("Bubba Wallace", "Kyle Busch", -185, 165),
    ("Ty Gibbs", "Chris Buescher", -215, 185),
    ("Ty Gibbs", "Carson Hocevar", -175, 155),
    ("Ty Gibbs", "Kyle Busch", -210, 180),
    ("Chris Buescher", "Carson Hocevar", 105, -125),
    ("Chris Buescher", "Kyle Busch", -110, -110),
    ("Carson Hocevar", "Kyle Busch", -130, 110),
    ("Josh Berry", "Austin Cindric", -130, 100),
    ("Josh Berry", "Ryan Preece", 168, -215),
    ("Josh Berry", "Connor Zilisch", -155, 125),
    ("Austin Cindric", "Ryan Preece", 108, -137),
    ("Austin Cindric", "Connor Zilisch", -148, 116),
    ("Ryan Preece", "Connor Zilisch", -175, 139),
    ("Justin Allgaier", "Brad Keselowski", -113, -117),
    ("Justin Allgaier", "Erik Jones", -123, -107),
    ("Justin Allgaier", "Daniel Suarez", -130, 100),
    ("Brad Keselowski", "Erik Jones", -125, -105),
    ("Brad Keselowski", "Shane Van Gisbergen", -155, 124),
    ("Erik Jones", "Daniel Suarez", -115, -115),
    ("Shane Van Gisbergen", "Daniel Suarez", 107, -138),
    ("Shane Van Gisbergen", "Michael McDowell", -130, 100),
    ("AJ Allmendinger", "Michael McDowell", -107, -123),
    ("Noah Gragson", "Ricky Stenhouse Jr", -105, -125),
    ("Noah Gragson", "Austin Dillon", -105, -125),
    ("Riley Herbst", "Zane Smith", 143, -180),
    ("John Hunter Nemechek", "Todd Gilliland", -180, 140),
    ("Ricky Stenhouse Jr", "Zane Smith", 170, -205),
]

PHOENIX = [
    ("Ryan Blaney", "Denny Hamlin", -200, 160),
    ("Ryan Blaney", "Kyle Larson", -175, 145),
    ("Ryan Blaney", "Christopher Bell", -155, 125),
    ("Ryan Blaney", "William Byron", -200, 160),
    ("Denny Hamlin", "Kyle Larson", -115, -115),
    ("Denny Hamlin", "Christopher Bell", 140, -170),
    ("Denny Hamlin", "William Byron", -115, -115),
    ("Kyle Larson", "Christopher Bell", 120, -150),
    ("Kyle Larson", "William Byron", -125, -105),
    ("Christopher Bell", "William Byron", -145, 115),
    ("Tyler Reddick", "Joey Logano", 170, -210),
    ("Tyler Reddick", "Chase Elliott", -150, 120),
    ("Tyler Reddick", "Ross Chastain", -120, -110),
    ("Joey Logano", "Ross Chastain", -180, 150),
    ("Chase Elliott", "Chase Briscoe", -120, -110),
    ("Chase Elliott", "Ross Chastain", 120, -150),
    ("Chase Briscoe", "Ross Chastain", 110, -140),
    ("Chris Buescher", "Carson Hocevar", 110, -140),
    ("Chris Buescher", "Bubba Wallace", -115, -115),
    ("Chris Buescher", "Brad Keselowski", -150, 120),
    ("Kyle Busch", "Carson Hocevar", 145, -175),
    ("Kyle Busch", "Brad Keselowski", -115, -115),
    ("Carson Hocevar", "Bubba Wallace", -135, 105),
    ("Carson Hocevar", "Brad Keselowski", -175, 145),
    ("Bubba Wallace", "Brad Keselowski", -155, 125),
    ("Josh Berry", "Connor Zilisch", -165, 135),
    ("Josh Berry", "Austin Cindric", 135, -165),
    ("Ty Gibbs", "Ryan Preece", -145, 115),
    ("Michael McDowell", "Shane Van Gisbergen", -180, 150),
    ("Ryan Preece", "Daniel Suarez", 105, -135),
    ("Shane Van Gisbergen", "Daniel Suarez", 170, -210),
    ("Erik Jones", "Noah Gragson", -115, -115),
    ("Noah Gragson", "Zane Smith", -135, 105),
]

COTA = [
    ("Shane Van Gisbergen", "Connor Zilisch", -285, 215),
    ("Christopher Bell", "Connor Zilisch", 105, -135),
    ("Christopher Bell", "William Byron", -125, -105),
    ("Christopher Bell", "Tyler Reddick", 130, -160),
    ("Christopher Bell", "Kyle Larson", -165, 135),
    ("Connor Zilisch", "William Byron", -155, 125),
    ("Tyler Reddick", "William Byron", -170, 140),
    ("Tyler Reddick", "Kyle Larson", -210, 140),
    ("William Byron", "Kyle Larson", -150, 120),
    ("Chase Briscoe", "Chase Elliott", 130, -160),
    ("Chase Briscoe", "Chris Buescher", -150, 110),
    ("Chase Briscoe", "Ross Chastain", 145, -175),
    ("Chase Briscoe", "AJ Allmendinger", -115, -115),
    ("Chase Elliott", "Chris Buescher", -180, 150),
    ("Chase Elliott", "Ross Chastain", 105, -135),
    ("Chase Elliott", "AJ Allmendinger", -155, 125),
    ("Chris Buescher", "Ross Chastain", 150, -190),
    ("Chris Buescher", "AJ Allmendinger", 115, -145),
    ("Ross Chastain", "AJ Allmendinger", -160, 130),
    ("Kyle Busch", "Michael McDowell", 175, -215),
    ("Kyle Busch", "Ryan Blaney", 160, -205),
    ("Kyle Busch", "Ty Gibbs", 130, -165),
    ("Kyle Busch", "Alex Bowman", 110, -140),
    ("Michael McDowell", "Ryan Blaney", 115, -145),
    ("Michael McDowell", "Alex Bowman", -165, 135),
    ("Ty Gibbs", "Alex Bowman", -110, -120),
    ("Daniel Suarez", "Joey Logano", -210, 170),
    ("Daniel Suarez", "Denny Hamlin", -170, 140),
    ("Daniel Suarez", "Carson Hocevar", -155, 125),
    ("Joey Logano", "Denny Hamlin", -105, -125),
    ("Joey Logano", "Austin Cindric", -125, -105),
    ("Denny Hamlin", "Carson Hocevar", -110, -120),
    ("Denny Hamlin", "Austin Cindric", -145, 115),
    ("Carson Hocevar", "Austin Cindric", -145, 115),
    ("Ryan Preece", "Bubba Wallace", 130, -160),
    ("Ryan Preece", "Brad Keselowski", -145, 115),
    ("Ryan Preece", "Todd Gilliland", -120, -110),
    ("Bubba Wallace", "Brad Keselowski", -150, 120),
    ("Bubba Wallace", "Todd Gilliland", -135, 105),
    ("Noah Gragson", "John Hunter Nemechek", 105, -135),
    ("Noah Gragson", "Zane Smith", 160, -200),
    ("Austin Dillon", "Jesse Love", 100, -130),
]

# 2026 Autotrader 400 @ Atlanta (Feb 22), BetOnline closing lines.
ATLANTA_SPRING_2026 = [
    ("Ryan Blaney", "Joey Logano", 100, -130),
    ("Ryan Blaney", "Chase Elliott", -125, -105),
    ("Ryan Blaney", "Kyle Larson", -145, 115),
    ("Ryan Blaney", "William Byron", -140, 110),
    ("Joey Logano", "Chase Elliott", -115, -115),
    ("Joey Logano", "Kyle Larson", -135, 105),
    ("Joey Logano", "William Byron", -135, 105),
    ("Chase Elliott", "Kyle Larson", -125, -105),
    ("Chase Elliott", "William Byron", -120, -110),
    ("Kyle Larson", "William Byron", -110, -120),
    ("Austin Cindric", "Kyle Busch", -135, 105),
    ("Austin Cindric", "Denny Hamlin", -140, 110),
    ("Austin Cindric", "Christopher Bell", -120, -110),
    ("Kyle Busch", "Denny Hamlin", -105, -125),
    ("Kyle Busch", "Brad Keselowski", 100, -130),
    ("Kyle Busch", "Christopher Bell", -105, -125),
    ("Denny Hamlin", "Brad Keselowski", 100, -130),
    ("Denny Hamlin", "Christopher Bell", 100, -130),
    ("Brad Keselowski", "Christopher Bell", -115, -115),
    ("Tyler Reddick", "Chase Briscoe", -110, -120),
    ("Ty Gibbs", "Chris Buescher", -115, -115),
    ("Ty Gibbs", "Ricky Stenhouse Jr", -120, -105),
    ("Ty Gibbs", "Ryan Preece", -110, -125),
    ("Chris Buescher", "Ricky Stenhouse Jr", -115, -115),
    ("Chris Buescher", "Ryan Preece", -135, 105),
    ("Ricky Stenhouse Jr", "Ryan Preece", -115, -115),
    ("Michael McDowell", "Daniel Suarez", 100, -130),
    ("Michael McDowell", "Erik Jones", -120, 100),
    ("Daniel Suarez", "Erik Jones", -145, 125),
]

RACES = [
    ("2026-02-22", "Autotrader 400",   ATLANTA_SPRING_2026),
    ("2026-03-01", "Duramax Grand Prix", COTA),
    ("2026-03-08", "Straight Talk",    PHOENIX),
    ("2026-03-15", "Pennzoil 500",     LAS_VEGAS),
    ("2026-03-22", "Goodyear 400",     DARLINGTON_SPRING),
    ("2026-03-29", "Cook Out 400",     MARTINSVILLE),
    ("2026-04-12", "Food City 500",    BRISTOL),
    ("2026-04-19", "AdventHealth 400", KANSAS),
    ("2026-04-26", "Jack Link",        TALLADEGA),
    ("2026-05-03", "Würth 400",         TEXAS),
    ("2026-05-10", "Go Bowling",        WATKINS_GLEN),
    ("2026-05-24", "Coca-Cola 600",     CHARLOTTE),
    ("2026-05-31", "Cracker Barrel 400", NASHVILLE),
    ("2026-06-14", "Great American Gateway", POCONO),
    ("2026-06-21", "Anduril 250",      SAN_DIEGO),
    ("2026-06-28", "Toyota Save Mart", SONOMA),
    ("2026-07-05", "EERO 400",         CHICAGOLAND),
    ("2026-07-12", "Quaker State 400", ATLANTA),
    ("2026-07-19", "Window World",     WILKESBORO),
    ("2026-08-09", "Iowa Corn",        IOWA),
    ("2026-08-23", "Dollar Tree 301",  LOUDON),
    ("2026-08-29", "Coke Zero",        DAYTONA),
    ("2026-09-06", "Southern 500",     DARLINGTON),
]

TIGHT_REG = {"num_leaves": 31, "min_data_in_leaf": 25, "lambda_l2": 2.0}
ALPHA = 0.85
N_SAMPLES = 30_000
HAZARD = {"superspeedway": 0.255, "intermediate": 0.142, "short": 0.086,
          "road": 0.093, "unique": 0.189}
# Per-track-type DNF dispersion: shared Gamma frailty (mean 1, var = dispersion)
# multiplies every driver's hazard within a sample. Values below are empirical
# — fit_dispersion.py measures per-race field-DNF-fraction variance from ≤2025
# entries.parquet, not tuned against 2026 backtest outcomes. Earlier values in
# this dict were hand-set to explain 2026 Atlanta outliers — that was test-set
# tuning of the exact shape kimi's audit flagged. These replace them.
DNF_DISPERSION = {
    "superspeedway": 0.10,  # DNF-share variance is much lower than the "big one"
                            # imagination suggests; historical range 8-54%,
                            # tight around 25%, not the wreck-ocalypse v=3.0 encodes
    "intermediate": 0.28,
    "short": 0.30,
    "road": 0.43,
    "unique": 0.17,
}

# Damage-hazard: probability a driver has a "damaged but continues" day at
# that track type. Distinct from DNF — car limps to end, driver loses 10-25
# positions but still finishes. Values fit from ≤2025 entries.parquet via
# fit_damage.py (finish_pos > qual_pos + 12, is_dnf=False, per track type,
# 5,327 driver-races across 4 seasons).
DAMAGE_HAZARD: dict[str, float | None] = {
    "superspeedway": 0.070,
    "intermediate": 0.079,
    "short": 0.082,
    "road": 0.103,
    "unique": None,  # dead classification; kept as safety fallback
}
DAMAGE_PENALTY = 3.0  # score-std units subtracted when damaged
# Empirical-Bayes shrinkage constant for per-driver hazard: how many prior
# same-type races we need to trust the driver-specific rate over the track
# baseline. Lower = trust driver rate more.
HAZARD_SHRINK_K = 5.0

# CV refit speed knobs. T is a 1-parameter fit; the val races just need to
# score honestly. Fewer boosters and a per-type val cap cost ~nothing on T
# quality but cut wall-clock ~10x.
CV_N_ESTIMATORS = 5
VAL_CAP_PER_TYPE = 30


def per_driver_hazards(target_df, track_type: str) -> np.ndarray:
    """Blend track-type baseline with driver-specific hazard rates.

    hazard_i = (n_i * driver_rate + k * baseline) / (n_i + k)

    Uses dnf_rate_at_type_10 (all-cause DNF at this track type) as the
    primary signal — it's the empirical measurement of how often the driver
    doesn't finish here. crash_dnf_rate_at_type_10 is used as a secondary
    bump for drivers whose crash rate is notably elevated above the field
    baseline (~half of typical DNFs are crashes).
    """
    base = HAZARD.get(track_type, 0.08)
    n = target_df.get("n_races_at_type_for_variance",
                      target_df.get("races_at_type_last_10", 0))
    if hasattr(n, "to_numpy"):
        n = n.to_numpy()
    n = np.nan_to_num(np.asarray(n, dtype=float), nan=0.0)

    dnf = target_df.get("dnf_rate_at_type_10")
    crash = target_df.get("crash_dnf_rate_at_type_10")

    # Primary driver rate: empirical DNF rate at this track type.
    if dnf is not None:
        d = np.asarray(dnf, dtype=float)
        driver_rate = np.where(np.isnan(d), base, d)
    else:
        driver_rate = np.full(len(target_df), base)

    # Secondary bump for elevated crash rate (only for drivers meaningfully
    # above the crash baseline). Assume half of typical DNFs are crashes, so
    # baseline crash rate ~ 0.5 * base. A driver with 2x that gets a hazard
    # bump of the excess.
    if crash is not None:
        c = np.asarray(crash, dtype=float)
        crash_baseline = 0.5 * base
        bump = np.where(
            np.isnan(c), 0.0,
            np.maximum(0.0, c - crash_baseline),  # only positive
        )
        driver_rate = driver_rate + bump

    with np.errstate(invalid="ignore"):
        blended = (n * driver_rate + HAZARD_SHRINK_K * base) \
                  / (n + HAZARD_SHRINK_K)
    # Clip: no hazard below 0.5x baseline, no more than 3.0x baseline.
    return np.clip(blended, 0.5 * base, 3.0 * base)


def per_driver_damage_hazards(
    target_df, track_type: str
) -> np.ndarray | None:
    """Per-driver damage-hazard array, or None if damage layer is disabled
    for this track type (base rate is None in DAMAGE_HAZARD).

    First pass: uniform per-driver rate at the track-type baseline. Later
    we can shrink by driver-specific historical damage rate the same way
    per_driver_hazards does for DNF.
    """
    base = DAMAGE_HAZARD.get(track_type)
    if base is None:
        return None
    return np.full(len(target_df), float(base))


def american_to_prob(odds):
    return 100.0 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def _log_clip(p, floor=1e-9):
    return float(np.log(np.clip(p, floor, 1.0 - floor)))


# Race IDs this backtest evaluates. Filled as each race is processed and
# excluded from every calibration pool, so no evaluated race ever helps
# calibrate another (walk-forward purity across the whole backtest).
_EVALUATED_RIDS: set = set()


def run_race(target_date, name_match, matchups, races, entries, sessions,
             loopstats, laptimes, features=None):
    """If `features` is passed in, skip the (expensive) rebuild — main() builds
    once and reuses across all 14 races."""
    from src.features.build_features import build_features
    from src.models import distribution as dist
    from src.models.calibrate import RaceScoresGT, find_best_temperature_by_type
    from src.models.gbm_ranker import GBMEnsemble

    target_ts = pd.Timestamp(target_date)
    r = races[
        ((races["date"] == target_ts)
         | (races["race_name"].str.contains(name_match, case=False, na=False)))
        & (races["season"] == 2026)
    ]
    if r.empty:
        return None
    target_rid = r.iloc[0]["race_id_short"]
    _EVALUATED_RIDS.add(target_rid)
    target_date_ts = r.iloc[0]["date"]
    # tracks.py is the source of truth for track_type. Parquet values can be
    # stale after a reclassification (e.g., Pocono/Indy moved to intermediate).
    from src.features.tracks import resolve_track_type
    tt = resolve_track_type(r.iloc[0].get("track_name", ""),
                            fallback=r.iloc[0]["track_type"])

    if features is None:
        features = build_features(races, entries, sessions, loopstats=loopstats, laptimes=laptimes)
    features["date"] = pd.to_datetime(features["date"])
    train = features[(features["date"] < target_date_ts) & (features["finish_pos"] > 0)]
    target = features[features["race_id_short"] == target_rid].reset_index(drop=True)
    if target.empty:
        return None

    model = GBMEnsemble()
    model.fit(train, n_estimators=15, **TIGHT_REG)

    race_order = train.groupby("race_id_short")["date"].first().sort_values().index.tolist()
    # Guardrail: never calibrate on races this backtest evaluates, and
    # never silently fall back to default temperatures on an empty pool.
    val_ids = [rid for rid in race_order[30:] if rid not in _EVALUATED_RIDS]
    if len(val_ids) < 4:
        print(f"WARNING [{name_match}]: validation pool has only {len(val_ids)} "
              f"races (of {len(race_order)} train races) - calibration falls back "
              f"to T=1.0/Tm=1.0. Check training-data span!")

    # Cap val pool per track type. T is a 1-parameter fit — 30 per type is
    # ample. Keep the MOST RECENT races of each type so calibration reflects
    # current conditions. Race chronology preserved within each type.
    tt_by_rid = train.groupby("race_id_short")["track_type"].first().to_dict()
    per_type: dict[str, list] = {}
    for rid in val_ids:
        per_type.setdefault(tt_by_rid[rid], []).append(rid)
    capped = set()
    for tt_pool, rids in per_type.items():
        capped.update(rids[-VAL_CAP_PER_TYPE:])
    val_ids = [rid for rid in val_ids if rid in capped]

    # --- Honest out-of-sample calibration: refit without each val race ---
    val_races, tt_list = [], []
    for rid in val_ids:
        sub = train[train["race_id_short"] == rid]
        train_minus = train[train["race_id_short"] != rid]
        m_cv = GBMEnsemble()
        m_cv.fit(train_minus, n_estimators=CV_N_ESTIMATORS, **TIGHT_REG)
        raw = m_cv.predict_scores(sub)
        gbm_z = (raw - raw.mean()) / (raw.std() + 1e-9)
        pl_z = ((sub["pl_effective"].to_numpy() - sub["pl_effective"].mean())
                / (sub["pl_effective"].std() + 1e-9))
        b = ALPHA * gbm_z + (1 - ALPHA) * pl_z
        b = b / max(b.std(), 1e-6)
        val_races.append(RaceScoresGT(
            scores=b, finishes=sub["finish_pos"].to_numpy(),
            is_dnf=sub["is_dnf"].to_numpy(),
            hazards=per_driver_hazards(sub, sub["track_type"].iloc[0]),
        ))
        tt_list.append(sub["track_type"].iloc[0])
    T_by_type = find_best_temperature_by_type(
        val_races, tt_list, default_T=1.0, n_samples=1500,
        dnf_dispersion_by_type=DNF_DISPERSION,
    )
    T = T_by_type.get(tt, 1.0)

    raw = model.predict_scores(target)
    gbm_z = (raw - raw.mean()) / (raw.std() + 1e-9)
    pl_z = ((target["pl_effective"].to_numpy() - target["pl_effective"].mean())
            / (target["pl_effective"].std() + 1e-9))
    blended = ALPHA * gbm_z + (1 - ALPHA) * pl_z
    blended = blended / max(blended.std(), 1e-6)
    haz = per_driver_hazards(target, tt)

    # Single calibrated temperature — no second layer.
    T_effective = T
    rng = np.random.default_rng(42)
    dam = per_driver_damage_hazards(target, tt)
    positions = dist.sample_finishing_orders(
        blended / T_effective, haz,
        n_samples=N_SAMPLES, rng=rng,
        dnf_dispersion=DNF_DISPERSION.get(tt, 0.0),
        damage_hazards=dam,
        damage_penalty=DAMAGE_PENALTY,
    )
    matchup_mtx = dist.matchup_probs(positions)

    # Accent- and format-insensitive driver lookup. The hand-entered matchup
    # lists use ASCII spellings ("Daniel Suarez", "John Hunter Nemechek") but
    # entries.parquet has "Daniel Suárez" and "John H. Nemechek". Without
    # normalization these matchups silently drop (~6% of the sample) and the
    # aggregate Delta is measured on a biased subsample. See
    # diag_unseen_categories.py-adjacent silent-drop check.
    import unicodedata
    def _norm(s):
        # Accent-fold and lower.
        norm = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().strip().lower()
        # Canonicalize middle-name variants by mapping each middle token to
        # its first letter, so "john hunter nemechek" and "john h. nemechek"
        # both become "john h nemechek". Preserves suffix tokens like "jr":
        # "ricky stenhouse jr" stays "ricky s jr" on both sides of the compare,
        # so the pair still matches.
        parts = norm.split()
        if len(parts) >= 3:
            first, middles, last = parts[0], parts[1:-1], parts[-1]
            middles_short = " ".join(m.rstrip(".")[0] for m in middles if m)
            norm = f"{first} {middles_short} {last}".strip()
        return norm
    driver_to_idx = {_norm(d): i for i, d in enumerate(target["driver"].values)}
    actual = entries[entries["race_id_short"] == target_rid][
        ["driver", "finish_pos"]
    ].copy()
    actual["nkey"] = actual["driver"].map(_norm)
    finish = dict(zip(actual["nkey"], actual["finish_pos"]))

    market_ll, model_ll = [], []
    market_correct = model_correct = n = 0
    matchup_cache_rows: list[dict] = []  # exported at end for downstream analysis
    dropped_matchups: list[tuple[str, str, str]] = []
    for a, b, oa, ob in matchups:
        ka, kb = _norm(a), _norm(b)
        if ka not in driver_to_idx or kb not in driver_to_idx:
            dropped_matchups.append((a, b, "not_in_target_entries"))
            continue
        if ka not in finish or kb not in finish:
            dropped_matchups.append((a, b, "no_finish_pos"))
            continue
        ma_raw = american_to_prob(oa); mb_raw = american_to_prob(ob)
        vig = ma_raw + mb_raw
        ma, mb = ma_raw / vig, mb_raw / vig
        i, j = driver_to_idx[ka], driver_to_idx[kb]
        p_model_a = float(matchup_mtx[i, j]); p_model_b = 1 - p_model_a
        fa, fb = int(finish[ka]), int(finish[kb])
        # Guard: NASCAR always assigns finished drivers a positive position,
        # but a scratched/partial-data row with finish_pos <= 0 would auto-win
        # via `fa < fb` (0 < 30). Skip rather than credit a phantom win.
        if fa <= 0 or fb <= 0: continue
        a_won = fa < fb
        matchup_cache_rows.append({
            "race_idx": None,  # filled by main() after concat
            "race": name_match,
            "a": a, "b": b,
            "p_a_raw": p_model_a,
            "outcome_a": int(a_won),
            "odds_a": oa, "odds_b": ob,
        })
        p_market_won = ma if a_won else mb
        p_model_won = p_model_a if a_won else p_model_b
        market_correct += int((ma > mb and a_won) or (mb > ma and not a_won))
        model_correct += int((p_model_a > p_model_b and a_won) or (p_model_b > p_model_a and not a_won))
        market_ll.append(-_log_clip(p_market_won))
        model_ll.append(-_log_clip(p_model_won))
        n += 1

    mkt_mean = float(np.mean(market_ll)); mdl_mean = float(np.mean(model_ll))
    drop_str = f"  DROPPED={len(dropped_matchups)}" if dropped_matchups else ""
    print(f"{name_match} ({tt}, T={T}): n={n}  mkt={mkt_mean:.4f}({market_correct}/{n})  "
    f"mdl={mdl_mean:.4f}({model_correct}/{n})  Delta={mdl_mean-mkt_mean:+.4f}{drop_str}")
    if dropped_matchups:
        for a, b, reason in dropped_matchups[:5]:
            print(f"    drop: {a} vs {b} ({reason})")
        if len(dropped_matchups) > 5:
            print(f"    ...and {len(dropped_matchups) - 5} more")
    return {"race": name_match, "tt": tt, "n": n, "market_ll": mkt_mean, "model_ll": mdl_mean,
            "market_correct": market_correct, "model_correct": model_correct, "T": T,
            "matchup_rows": matchup_cache_rows}


def main():
    from src.features.build_features import build_features
    races = pd.read_parquet("data/processed/races.parquet")
    entries = pd.read_parquet("data/processed/entries.parquet")
    sessions = pd.read_parquet("data/processed/sessions.parquet")
    loopstats = pd.read_parquet("data/processed/loopstats.parquet")
    laptimes = pd.read_parquet("data/processed/laptimes.parquet")
    races["date"] = pd.to_datetime(races["date"])
    entries["date"] = pd.to_datetime(entries["date"])

    print("Building features once (this is the slow part)...")
    features = build_features(races, entries, sessions, loopstats=loopstats, laptimes=laptimes)
    features["date"] = pd.to_datetime(features["date"])
    print(f"Done. {len(features)} rows.\n")

    summary = []
    for date, name, matchups in RACES:
        result = run_race(date, name, matchups, races, entries, sessions,
                          loopstats, laptimes, features=features)
        if result:
            summary.append(result)

    print("\n" + "=" * 70)
    print(f"SUMMARY ({len(summary)} races vs OddsLogic closing lines)")
    print("=" * 70)
    summary_df = pd.DataFrame([{k: v for k, v in s.items() if k != "matchup_rows"}
                               for s in summary])
    print(summary_df.round(4).to_string(index=False))
    total_n = sum(s["n"] for s in summary)
    total_mkt = sum(s["market_ll"] * s["n"] for s in summary) / total_n
    total_mdl = sum(s["model_ll"] * s["n"] for s in summary) / total_n
    total_mkt_c = sum(s["market_correct"] for s in summary)
    total_mdl_c = sum(s["model_correct"] for s in summary)
    print(f"\nCombined ({total_n} matchups, weighted mean):")
    print(f"  Market: log-loss {total_mkt:.4f}, correct {total_mkt_c}/{total_n} ({total_mkt_c/total_n*100:.1f}%)")
    print(f"  Model:  log-loss {total_mdl:.4f}, correct {total_mdl_c}/{total_n} ({total_mdl_c/total_n*100:.1f}%)")
    print(f"  Δ: {total_mdl - total_mkt:+.4f}")

    # Cache per-matchup probabilities for downstream calibration analysis
    # (beta_calibrate_backtest.py, calibrate_sweep.py). Writing here means
    # the cache is guaranteed consistent with THIS run's honest calibration.
    all_rows = []
    for race_idx, s in enumerate(summary):
        for row in s["matchup_rows"]:
            row = dict(row)
            row["race_idx"] = race_idx
            all_rows.append(row)
    if all_rows:
        import os
        out_path = "data/processed/backtest_matchups.parquet"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame(all_rows).to_parquet(out_path, index=False)
        print(f"\nCached {len(all_rows)} matchup rows -> {out_path}")


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
