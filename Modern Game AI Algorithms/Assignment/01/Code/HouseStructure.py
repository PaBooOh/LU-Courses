
from gdpc import geometry as GEO
from gdpc import interface as INTF
from gdpc import toolbox as TB
from gdpc import worldLoader as WL

import numpy

Start_X, Start_Y, Start_Z, End_X, End_Y, End_Z = INTF.requestPlayerArea(50, 50)
print(Start_X, Start_Y, Start_Z, End_X, End_Y, End_Z)

WORLDSLICE = WL.WorldSlice(Start_X, Start_Z,
                           End_X + 1, End_Z + 1)  # this takes a while

ROADHEIGHT = 0

def buildfoundation():

    pass

def buildPerimeter():
    """Build a wall along the build area border.

    In this function we're building a simple wall around the build area
        pillar-by-pillar, which means we can adjust to the terrain height
    """
    # HEIGHTMAP
    # Heightmaps are an easy way to get the uppermost block at any coordinate
    # There are four types available in a world slice:
    # - 'WORLD_SURFACE': The top non-air blocks
    # - 'MOTION_BLOCKING': The top blocks with a hitbox or fluid
    # - 'MOTION_BLOCKING_NO_LEAVES': Like MOTION_BLOCKING but ignoring leaves
    # - 'OCEAN_FLOOR': The top solid blocks
    heights = WORLDSLICE.heightmaps["MOTION_BLOCKING_NO_LEAVES"]


    print("Building east-west walls...")
    # building the east-west walls  

    for x in range(Start_X, End_X + 1):
        # the northern wall
        print((x, Start_Z))
        y = heights[(x, Start_Z)]
        GEO.placeCuboid(x, y - 2, Start_Z, x, y, Start_Z, "granite")
        GEO.placeCuboid(x, y + 1, Start_Z, x, y + 4, Start_Z, "granite_wall")
        # the southern wall
        y = heights[(x, End_Z)]
        GEO.placeCuboid(x, y - 2, End_Z, x, y, End_Z, "red_sandstone")
        GEO.placeCuboid(x, y + 1, End_Z, x, y + 4, End_Z, "red_sandstone_wall")

    print("Building north-south walls...")
    # building the north-south walls
    for z in range(Start_Z, End_Z + 1):
        # the western wall
        y = heights[(Start_X, z)]
        GEO.placeCuboid(Start_X, y - 2, z, Start_X, y, z, "sandstone")
        GEO.placeCuboid(Start_X, y + 1, z, Start_X, y + 4, z, "sandstone_wall")
        # the eastern wall
        y = heights[(End_X, z)]
        GEO.placeCuboid(End_X, y - 2, z, End_X, y, z, "prismarine")
        GEO.placeCuboid(End_X, y + 1, z, End_X, y + 4, z, "prismarine_wall")



height = WORLDSLICE.heightmaps["MOTION_BLOCKING"][(Start_X, Start_Y)]
INTF.runCommand(f"tp @a {Start_X} {height} {Start_Z}")
print(f"/tp @a {Start_X} {height} {Start_Z}")

buildPerimeter()