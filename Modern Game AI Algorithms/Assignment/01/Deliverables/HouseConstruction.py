

import interfaceUtils as INTF
import direct_interface as DINTF
import mapUtils as MapU
import worldLoader as WL
import numpy as np


""" Setting """
# Buffer
INTF.setBuffering(True)
INTF.setBufferLimit(1024)

# BuildArea
# INTF.setBuildArea(0, 0, 0, 200, 200, 200)
# INTF.setBuildArea(200, 0, 200, 400, 200, 400)
# INTF.setBuildArea(400, 0, 400, 600, 200, 600)
Start_X, Start_Y, Start_Z, End_X, End_Y, End_Z = INTF.requestBuildArea()
print(Start_X, Start_Y, Start_Z, End_X, End_Y, End_Z)

worldslice = WL.WorldSlice(Start_X, Start_Z, End_X, End_Z)
heightmap = MapU.calcGoodHeightmap(worldslice)
# print(heightmap.shape)

# construction
layer_counts = 3

def heightAt(x, z):
    return heightmap[(x - Start_X, z - Start_Z)]

def getAllUppermostBlocksHeight(X, Z, size_X, size_Z):
    heights = []
    for x in range(X, X + size_X + 1):
        for z in range(Z, Z + size_Z + 1):
            y = heightAt(x, z)
            heights.append(y)
    return heights

def check_water_lava_detailed(size_min, n_places = None):
    flag = 2
    feasible_place = []
    # upper and bottom boundaries
    O_X = Start_X + size_min + 1
    O_Z = Start_Z + size_min + 1
    E_X = End_X - size_min
    E_Z = End_Z - size_min
    print('---------> Checking whether the targeted temple will be on lava or water...')
    for Z in range(O_Z, E_Z): # area
        for X in range(O_X, E_X): # area
            if flag == 0:
                feasible_place.append((X, Z))
            if n_places is not None and len(feasible_place) >= n_places:
                    return feasible_place
            flag = 0
            for z in range(Z, Z + size_min + 1):
                if flag == 1:
                    break
                for x in range(X, X + size_min + 1):
                    y = heightAt(x, z) - 1
                    if "minecraft:water" == worldslice.getBlockAt(x, y, z) or "minecraft:lava" == worldslice.getBlockAt(x, y, z):
                        flag = 1
                        break

    return feasible_place

def initialize():
    # set temple size (random)
    temple_size_min = 27
    temple_size_max = 35
    temple_size_range = list(range(temple_size_min, temple_size_max + 1))

    # # 1) scan environment using the minimum temple to check whether the targeted temple will be on water or lava.
    # feasible_place = check_water_lava_detailed(temple_size_min)
    # if len(feasible_place) == 0:
    #     print('*** Not found any fitting place to construct! Plese try other areas.')
    #     return
    # else:
    #     print('---------> A fitting place may exist! Starting a deep search...')

    # 1) deep search: scan environment to check whether the targeted temple will be on a sea.
    while True:
        # randomly choose a size (for both width and length) while removing it, or the iteration would otherwise not stop if there are no fitting places.
        if len(temple_size_range) == 0:
            print('*** Not found any fitting place to construct! Plese try other areas.')
            return
        temple_size = np.random.choice(temple_size_range)
        temple_size_range.remove(temple_size)
        # determine a fitting construction coordinates in case out of index
        # temple_X = np.random.randint(Start_X + temple_size + 1, End_X - temple_size)
        # temple_Z = np.random.randint(Start_Z + temple_size + 1, End_Z - temple_size)
        # temple_Square = (temple_X, temple_Z, temple_size, temple_size)
        feasible_place = check_water_lava_detailed(temple_size)
        if len(feasible_place) == 0:
            # print('This is not a fitting place. New Try... ')
            continue
        else:
            layer_size_ratio = temple_size // (layer_counts ** 2) 
            layer_height_ratio = temple_size // 5 
            temple_X, temple_Z = feasible_place[np.random.choice(len(feasible_place))]
            print('---------> Temple will be (x, z): (', temple_X, ',', temple_Z, ') with size (', temple_size, ',', temple_size, ') ...')
            break

    temple_min_height = min(
            heightAt(temple_X, temple_Z),
            heightAt(temple_X + temple_size - 1, temple_Z),
            heightAt(temple_X, temple_Z + temple_size - 1),
            heightAt(temple_X + temple_size - 1, temple_Z + temple_size - 1)
        ) - 1

    # Get the minimum height
    uppermost_blocks_heights = getAllUppermostBlocksHeight(temple_X, temple_Z, temple_size, temple_size)
    temple_Y = temple_min_height
#     temple_Y = min(uppermost_blocks_heights) - 1
#     temple_Y_min = min(uppermost_blocks_heights) - 1
#     temple_Y_max = max(uppermost_blocks_heights) - 1
#     error_Y = abs(temple_Y_max -  temple_Y_min) // 2
    temple_length = temple_X + temple_size
    temple_width = temple_Z + temple_size
    
    
    # Start building!
    buildTemple(temple_X, temple_Y, temple_Z, temple_length, temple_width)
   
    
    

def buildTemple(x1, y1, z1, x2, z2):

    central_x = x1 + (x2 - x1) // 2
    central_z = z1 + (z2 - z1) // 2
    pillar_height = np.random.choice([4, 5, 6])
    
#     print('Triming leaves and trees ...')
#     INTF.fill(x1, y1 + 1, z1, 
#             x2 - 1, y1 + 50, z2 - 1, "air")
    
    # ===================================== build foundation
    print('Building foudation ...')
    INTF.fill(x1, y1 + 1, z1, 
            x2 - 1, y1 + 2, z2 - 1, "air")
    INTF.fill(x1, y1, z1, 
            x2 - 1, y1, z2 - 1, "crimson_nylium")
    # ===================================== build layers
    print('Building layers ...')
    INTF.fill(x1 + 2, y1 + 1, z1 + 2, 
            x2 - 3, y1 + 1, z2 - 3, "chiseled_stone_bricks")
    INTF.fill(x1 + 2, y1 + 2, z1 + 2, 
            x2 - 3, y1 + 2, z2 - 3, "air")

    INTF.fill(x1 + 3, y1 + 2, z1 + 3, 
            x2 - 4, y1 + 2, z2 - 4, "chiseled_stone_bricks")
    INTF.fill(x1 + 3, y1 + 3, z1 + 3, 
            x2 - 4, y1 + 3, z2 - 4, "air")

    INTF.fill(x1 + 4, y1 + 3, z1 + 4, 
            x2 - 5, y1 + 3, z2 - 5, "chiseled_stone_bricks")
    INTF.fill(x1 + 4, y1 + 4, z1 + 4, 
            x2 - 5, y1 + 5, z2 - 5, "air")

    INTF.fill(x1 + 5, y1 + 4, z1 + 5, 
            x2 - 6, y1 + 5, z2 - 6, "chiseled_stone_bricks")
    INTF.fill(x1 + 5, y1 + 6, z1 + 5, 
            x2 - 6, y1 + 7, z2 - 6, "air")

    INTF.fill(x1 + 6, y1 + 6, z1 + 6, 
            x2 - 7, y1 + 7, z2 - 7, "chiseled_stone_bricks")
    INTF.fill(x1 + 6, y1 + 8, z1 + 6, 
            x2 - 7, y1 + 9, z2 - 7, "air")

    INTF.fill(x1 + 7, y1 + 8, z1 + 7, 
            x2 - 8, y1 + 9, z2 - 8, "chiseled_stone_bricks")
    INTF.fill(x1 + 7, y1 + 10, z1 + 7, 
            x2 - 8, y1 + 10 + pillar_height + 5, z2 - 8, "air")
    
    # ===================================== build stairs
    print('Building stairs ...')
    if abs(x2 - x1) & 1 == 1: #  if odd
        stairs = 5 // 2
        INTF.fill(central_x - stairs + 1, y1 + 1, z1, 
            central_x + stairs - 1, y1 + 9, central_z, "air")
        for i in range(1, 6):
            INTF.fill(central_x - stairs, y1 + i, z1 + i, 
                central_x + stairs, y1 + i, central_z, "mossy_stone_brick_stairs[facing=south]")
        for i in range(6, 9):
            INTF.fill(central_x - stairs + 1, y1 + i, z1 + i, 
                central_x + stairs - 1, y1 + i, central_z, "mossy_stone_brick_stairs[facing=south]")

        INTF.fill(central_x - stairs + 1, y1, central_z - 1, 
                central_x + stairs - 1, y1 + 9, central_z, "chiseled_stone_bricks")
    else: # if even
        stairs = 6 // 2
        INTF.fill(central_x - stairs + 1, y1 + 1, z1, 
            central_x + stairs - 2, y1 + 9, central_z, "air")
        for i in range(1, 6):
            INTF.fill(central_x - stairs, y1 + i, z1 + i, 
                central_x + stairs - 1, y1 + i, central_z, "mossy_stone_brick_stairs[facing=south]")
        for i in range(6, 9):
            INTF.fill(central_x - stairs + 1, y1 + i, z1 + i, 
                central_x + stairs - 2, y1 + i, central_z, "mossy_stone_brick_stairs[facing=south]")

        INTF.fill(central_x - stairs + 1, y1, central_z - 1, 
                central_x + stairs - 2, y1 + 9, central_z, "chiseled_stone_bricks")
        
    
    # ===================================== build temple
    print('Building temple ...')
    # build plinthes and pillars
    
    # north-west
    INTF.setBlock(x2 - 8, y1 + 10, z2 - 8, "polished_blackstone_brick_stairs[facing=south]")
    INTF.setBlock(x2 - 9, y1 + 10, z2 - 8, "polished_blackstone_brick_stairs[facing=south]")
    INTF.setBlock(x2 - 10, y1 + 10, z2 - 8, "polished_blackstone_brick_stairs[facing=south]")
    INTF.setBlock(x2 - 8, y1 + 10, z2 - 10, "polished_blackstone_brick_stairs[facing=east]")
    # INTF.setBlock(x2 - 9, y1 + 10, z2 - 10, "polished_diorite_stairs[facing=north]")
    # INTF.setBlock(x2 - 10, y1 + 10, z2 - 10, "polished_diorite_stairs[facing=north]")
    INTF.setBlock(x2 - 8, y1 + 10, z2 - 9, "polished_blackstone_brick_stairs[facing=east]")
    # INTF.setBlock(x2 - 10, y1 + 10, z2 - 9, "polished_diorite_stairs[facing=west]")
    INTF.fill(x2 - 9, y1 + 10, z2 - 9, 
            x2 - 9, y1 + 10 + pillar_height, z2- 9, "damaged_anvil[facing=west]") 
    INTF.fill(x2 - 9, y1 + 10, z2 - 11, 
            x2 - 9, y1 + 10 + pillar_height, z2- 10, "damaged_anvil[facing=north]") # west wall
    INTF.fill(x2 - 10, y1 + 10, z2 - 9, 
            x2 - 10, y1 + 10 + pillar_height, z2- 9, "damaged_anvil[facing=west]") 
    
    # north-east
    INTF.setBlock(x1 + 7, y1 + 10, z2 - 8, "polished_blackstone_brick_stairs[facing=south]")
    INTF.setBlock(x1 + 8, y1 + 10, z2 - 8, "polished_blackstone_brick_stairs[facing=south]")
    INTF.setBlock(x1 + 9, y1 + 10, z2 - 8, "polished_blackstone_brick_stairs[facing=south]")
    INTF.setBlock(x1 + 7, y1 + 10, z2 - 10, "polished_blackstone_brick_stairs[facing=west]")
    # INTF.setBlock(x1 + 8, y1 + 10, z2 - 10, "polished_diorite_stairs[facing=north]")
    # INTF.setBlock(x1 + 9, y1 + 10, z2 - 10, "polished_diorite_stairs[facing=north]")
    INTF.setBlock(x1 + 7, y1 + 10, z2 - 9, "polished_blackstone_brick_stairs[facing=west]")
    # INTF.setBlock(x1 + 9, y1 + 10, z2 - 9, "polished_diorite_stairs[facing=east]")
    INTF.fill(x1 + 8, y1 + 10, z2 - 9, 
            x2- 9, y1 + 10 + pillar_height, z2- 9, "damaged_anvil[facing=west]") # north wall
    INTF.fill(x1 + 9, y1 + 10, z2 - 9, 
            x1 + 9, y1 + 10 + pillar_height, z2- 9, "damaged_anvil[facing=west]")
    INTF.fill(x1 + 8, y1 + 10, z2 - 11, 
            x1 + 8, y1 + 10 + pillar_height, z2- 10, "damaged_anvil[facing=north]") # east wall

    # south-west
    INTF.setBlock(x2 - 8, y1 + 10, z1 + 7 + 4, "polished_blackstone_brick_stairs[facing=north]")
    INTF.setBlock(x2 - 9, y1 + 10, z1 + 7 + 4, "polished_blackstone_brick_stairs[facing=north]")
    INTF.setBlock(x2 - 10, y1 + 10, z1 + 7 + 4, "polished_blackstone_brick_stairs[facing=north]")
    INTF.setBlock(x2 - 8, y1 + 10, z1 + 9 + 4, "polished_blackstone_brick_stairs[facing=east]")
    # INTF.setBlock(x2 - 9, y1 + 10, z1 + 9 + 4, "polished_diorite_stairs[facing=south]")
    # INTF.setBlock(x2 - 10, y1 + 10, z1 + 9 + 4, "polished_diorite_stairs[facing=south]")
    INTF.setBlock(x2 - 8, y1 + 10, z1 + 8 + 4, "polished_blackstone_brick_stairs[facing=east]")
    # INTF.setBlock(x2 - 10, y1 + 10, z1 + 8 + 4, "polished_diorite_stairs[facing=west]")
    INTF.fill(x2 - 9, y1 + 10, z1 + 8 + 4, 
            x2 - 9, y1 + 10 + pillar_height, z1 + 8 + 4, "damaged_anvil[facing=west]")
    INTF.fill(x2 - 9, y1 + 10, z1 + 9 + 4, 
            x2 - 9, y1 + 10 + pillar_height, z1 + 10 + 4, "damaged_anvil[facing=north]") # west wall 
    INTF.fill(x2 - 10, y1 + 10, z1 + 8 + 4, 
            x2 - 10, y1 + 10 + pillar_height, z1 + 8 + 4, "damaged_anvil[facing=west]")

    # south-east
    INTF.setBlock(x1 + 7, y1 + 10, z1 + 7 + 4, "polished_blackstone_brick_stairs[facing=north]")
    INTF.setBlock(x1 + 8, y1 + 10, z1 + 7 + 4, "polished_blackstone_brick_stairs[facing=north]")
    INTF.setBlock(x1 + 9, y1 + 10, z1 + 7 + 4, "polished_blackstone_brick_stairs[facing=north]")
    INTF.setBlock(x1 + 7, y1 + 10, z1 + 9 + 4, "polished_blackstone_brick_stairs[facing=west]")
    # INTF.setBlock(x1 + 8, y1 + 10, z1 + 9 + 4, "polished_diorite_stairs[facing=south]")
    # INTF.setBlock(x1 + 9, y1 + 10, z1 + 9 + 4, "polished_diorite_stairs[facing=south]")
    INTF.setBlock(x1 + 7, y1 + 10, z1 + 8 + 4, "polished_blackstone_brick_stairs[facing=west]")
    # INTF.setBlock(x1 + 9, y1 + 10, z1 + 8 + 4, "polished_diorite_stairs[facing=east]")
    INTF.fill(x1 + 8, y1 + 10, z1 + 8 + 4, 
            x1 + 8, y1 + 10 + pillar_height, z1 + 8 + 4, "damaged_anvil[facing=west]")
    INTF.fill(x1 + 8, y1 + 10, z1 + 9 + 4, 
            x1 + 8, y1 + 10 + pillar_height, z1 + 10 + 4, "damaged_anvil[facing=north]") # east wall
    INTF.fill(x1 + 9, y1 + 10, z1 + 8 + 4, 
            x1 + 9, y1 + 10 + pillar_height, z1 + 8 + 4, "damaged_anvil[facing=west]")
    
    current_Y = y1 + 10 + pillar_height
    
    # build top floor (1)
    INTF.fill(x1 + 7, current_Y + 1, z1 + 7 + 4, 
            x2 - 8, current_Y + 1, z2 - 8, "chiseled_stone_bricks") 

    # build top tiny wall (1)
    INTF.fill(x1 + 8, current_Y + 2, z1 + 8 + 4, 
            x2 - 9, current_Y + 2, z2 - 9, "polished_blackstone_brick_wall") # stone_brick_wall
    
    # build top floor (2)
    INTF.fill(x1 + 7, current_Y + 3, z1 + 7 + 4, 
            x2 - 8, current_Y + 3, z2 - 8, "chiseled_stone_bricks") 


    # ===================================== Decoration 
    print('Start decoration ...')

    # set lantern under the four corner of ceiling
    INTF.setBlock(x1 + 7, current_Y, z1 + 7 + 4, "lantern")
    INTF.setBlock(x2 - 8, current_Y, z1 + 7 + 4, "lantern")
    INTF.setBlock(x2 - 8, current_Y, z2 - 8, "lantern")
    INTF.setBlock(x1 + 7, current_Y, z2 - 8, "lantern")
    # latern under the central ceiling
    if abs(x2 - x1) & 1 == 1: #  if odd
        INTF.fill(central_x - 1, current_Y, central_z, 
                central_x + 1, current_Y, central_z + 2, "chain")
        INTF.fill(central_x - 1, current_Y - 1, central_z, 
                central_x + 1, current_Y - 1, central_z + 2, "soul_lantern")
    else:
        INTF.fill(central_x - 2, current_Y, central_z, 
                central_x + 1, current_Y, central_z + 3, "chain")
        INTF.fill(central_x - 2, current_Y - 1, central_z, 
                central_x + 1, current_Y - 1, central_z + 3, "soul_lantern")
        
    # set dragon head on the 4 corner of the top layer
    INTF.setBlock(x2 - 8, y1 + 9, z1 + 7, "dragon_head[rotation=2]") # south-west
    INTF.setBlock(x1 + 7, y1 + 9, z1 + 7, "dragon_head[rotation=14]") # south-east
    INTF.setBlock(x2 - 8, y1 + 9, z2 - 8, "dragon_head[rotation=6]") # north-west
    INTF.setBlock(x1 + 7, y1 + 9, z2 - 8, "dragon_head[rotation=10]") # north-east

    # build patterns of protrusion located in the 4 corner on the top floor
    # south-west
    INTF.fill(x2 - 9, current_Y + 4, z1 + 7 + 4, 
            x2 - 8, current_Y + 4, z1 + 8 + 4, "chiseled_stone_bricks")
    INTF.fill(x2 - 9, current_Y + 5, z1 + 6 + 4, 
            x2 - 7, current_Y + 5, z1 + 8 + 4, "chiseled_stone_bricks")
    INTF.fill(x2 - 9, current_Y + 4, z1 + 8 + 4, 
            x2 - 9, current_Y + 5, z1 + 8 + 4, "air") 
    # south-east
    INTF.fill(x1 + 7, current_Y + 4, z1 + 7 + 4, 
            x1 + 8, current_Y + 4, z1 + 8 + 4, "chiseled_stone_bricks")
    INTF.fill(x1 + 6, current_Y + 5, z1 + 6 + 4, 
            x1 + 8, current_Y + 5, z1 + 8 + 4, "chiseled_stone_bricks")
    INTF.fill(x1 + 8, current_Y + 4, z1 + 8 + 4, 
            x1 + 8, current_Y + 5, z1 + 8 + 4, "air") 
    # north-west
    INTF.fill(x2 - 9, current_Y + 4, z2 -9, 
            x2 - 8, current_Y + 4, z2 - 8, "chiseled_stone_bricks")
    INTF.fill(x2 - 9, current_Y + 5, z2 - 9, 
            x2 - 7, current_Y + 5, z2 - 7, "chiseled_stone_bricks")
    INTF.fill(x2 - 9, current_Y + 4, z2 - 9, 
            x2 - 9, current_Y + 5, z2 - 9, "air") 
    # north-east
    INTF.fill(x1 + 7, current_Y + 4, z2 - 9, 
            x1 + 8, current_Y + 4, z2 - 8, "chiseled_stone_bricks")
    INTF.fill(x1 + 6, current_Y + 5, z2 - 9, 
            x1 + 8, current_Y + 5, z2 - 7, "chiseled_stone_bricks")
    INTF.fill(x1 + 8, current_Y + 4, z2 - 9, 
            x1 + 8, current_Y + 5, z2 - 9, "air")

    # build two campfires
    if abs(x2 - x1) & 1 == 1: #  if odd
        INTF.setBlock(central_x - 2, y1 + 9, z1 + 7, "campfire[facing=south]") # left
        INTF.setBlock(central_x + 2, y1 + 9, z1 + 7, "campfire[facing=south]") # right
    else:
        INTF.setBlock(central_x - 3, y1 + 9, z1 + 7, "campfire[facing=south]") # left
        INTF.setBlock(central_x + 2, y1 + 9, z1 + 7, "campfire[facing=south]") # right  
    
    # build glass and tribute
    if abs(x2 - x1) & 1 == 1: #  if odd
        INTF.fill(central_x - 1, y1 + 10, central_z + 3, 
                central_x + 1, y1 + 10, central_z + 3, "cyan_stained_glass")
        INTF.fill(central_x - 1, y1 + 11, central_z + 3, 
                central_x + 1, y1 + 11, central_z + 3, "yellow_carpet")
        INTF.setBlock(central_x, y1 + 11, central_z + 3, "turtle_egg[eggs=4]")  
    else:
        INTF.fill(central_x - 2, y1 + 10, central_z + 3, 
                central_x + 1, y1 + 10, central_z + 3, "cyan_stained_glass")
        INTF.fill(central_x - 2, y1 + 11, central_z + 3, 
                central_x + 1, y1 + 11, central_z + 3, "yellow_carpet")       
        INTF.fill(central_x - 1, y1 + 11, central_z + 3, 
                central_x, y1 + 11, central_z + 3, "turtle_egg[eggs=3]")
    
    # debug
    # a part of top floor
    if abs(x2 - x1) & 1 == 1: #  if odd
        INTF.fill(central_x - stairs + 1, y1 + 9, z1 + 9, 
                central_x + stairs - 1, y1 + 9, central_z, "smooth_stone_slab")
    else:
        INTF.fill(central_x - stairs + 1, y1 + 9, z1 + 9, 
                central_x + stairs - 2, y1 + 9, central_z, "smooth_stone_slab")

    if INTF.isBuffering():
        INTF.sendBlocks()

initialize()

    




    
    

