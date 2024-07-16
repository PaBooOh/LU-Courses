#! /usr/bin/python3
"""### Test various aspects of the framework automatically.

The tests contained in this file include:

-

It is not meant to be imported.
"""

__all__ = []
__version__ = "v4.2_dev"

import random
import sys
import time

# import example
import interfaceUtils
# import bitarray
import lookup
import toolbox

# import timeit

# import mapUtils
# import visualizeMap
# import worldLoader


class TestException(Exception):
    def __init__(self, *args):
        super().__init__(*args)

    # inherited __repr__ from OrderedDict is sufficient


def verifyPaletteBlocks():
    """Check blockColours blocks."""
    print(f"\n{lookup.TCOLORS['yellow']}Running blockColours palette test...")

    print(f"\t{lookup.TCOLORS['grey']}Preparing...", end="\r")
    tester = interfaceUtils.Interface()
    counter = 0
    badcounter = 0
    passed = []
    tocheck = [block for i in lookup.PALETTE.values()
               for block in i] + list(lookup.MAPTRANSPARENT)
    print(f"\t{lookup.TCOLORS['grey']}Preparing done.")

    for block in tocheck:
        if block in passed:
            badcounter += 1
            print()
            print(f"\t\t{lookup.TCOLORS['grey']}{block} is duplicated")
        elif not tester.placeBlock(0, 0, 0, block).isnumeric():
            badcounter += 1
            print()
            print(tester.placeBlock(0, 0, 0, block))
            print(f"\t\t{lookup.TCOLORS['orange']}Cannot verify {block}")
        counter += 1
        passed.append(block)
        print(f"\t{lookup.TCOLORS['blue']}{counter}"
              f"{lookup.TCOLORS['CLR']} blocks verified.", end='\r')
    tester.placeBlock(0, 0, 0, 'air')
    if badcounter > 0:
        raise TestException(f"{lookup.TCOLORS['red']}{badcounter}/"
                            f"{lookup.TCOLORS['grey']}{counter}"
                            f"{lookup.TCOLORS['red']} blocks duplicate "
                            "or could not be verified.\n"
                            f"{lookup.TCOLORS['orange']}"
                            "Please check you are running"
                            f" on Minecraft {lookup.VERSION}")

    print(f"{lookup.TCOLORS['green']}"
          f"All {counter} blocks successfully verified!")


def testBooks():
    """**Check book creation and storage**."""
    print(f"\n{lookup.TCOLORS['yellow']}Running book test...")
    TITLE = 'Testonomicon'
    AUTHOR = 'Dr. Blinkenlights'
    DESCRIPTION = 'All is doomed that enters here.'
    DESCRIPTIONCOLOR = 'aqua'

    CRITERIA = ('Lectern is at 0 255 0 and displays correctly?',
                'Lectern is facing east?', 'Book is legible?',
                'Book contents is correct?', 'Final page is displayed?')

    text = ('New line:\n'
            'Automatic sentence line breaking\n'
            'Automatic_word_line_breaking\n'
            '\\cCenter-aligned\n'
            '\\rRight-aligned\n'
            'New page:\f'
            '§6gold text§r\n'
            '§k█§r < obfuscated text§r\n'
            '§lbold§r text§r\n'
            '§mstriked§r text§r\n'
            '§nunderlined§r text§r\n'
            '§oitalic§r text§r\n'
            '\f\\\\s╔══════════╗\\\\n'
            '║                      `║\\\\n'
            '║                      `║\\\\n'
            '║:   Preformatted  :║\\\\n'
            '║         Page       `║\\\\n'
            '║                      `║\\\\n'
            '║                      `║\\\\n'
            '║        ☞⛏☜       .║\\\\n'
            '║                      `║\\\\n'
            '║                      `║\\\\n'
            '║                      `║\\\\n'
            '║                      `║\\\\n'
            '║                      `║\\\\n'
            '╚══════════╝')

    print(f"\t{lookup.TCOLORS['grey']}Writing book...", end="\r")
    book = toolbox.writeBook(text, TITLE, AUTHOR,
                             DESCRIPTION, DESCRIPTIONCOLOR)
    print("\tWriting book done.")

    print("\tPlacing lectern...", end="\r")
    toolbox.placeLectern(0, 255, 0, book, 'east')
    print("\tPlacing lectern done.")

    print("\tPrompting user...", end="\r")
    for no, prompt in enumerate(CRITERIA):
        reply = input(f"\t{lookup.TCOLORS['blue']}{no}/{len(CRITERIA)} "
                      f"{prompt} (y/*): {lookup.TCOLORS['CLR']}")
        if reply == '' or reply[0].lower() != 'y':
            raise TestException(f"Book criteria #{no} was failed:\n"
                                f"\t{prompt}: {reply}")
    print(f"{lookup.TCOLORS['green']}Book test complete!")
    interfaceUtils.globalinterface.placeBlock(0, 255, 0, "air")


def testCache():
    """**Check Interface cache functionality**."""
    print(f"\n{lookup.TCOLORS['yellow']}Running Interface cache test...")
    SIZE = 16
    PALETTES = (("birch_fence", "stripped_birch_log"),
                ("dark_oak_fence", "stripped_dark_oak_log"))

    def clearTestbed():
        """Clean testbed for placement from memory."""
        print("\t\tWiping blocks...", end="\r")
        tester.fill(0, 1, 0, SIZE - 1, 1, SIZE - 1, "shroomlight")
        tester.sendBlocks()
        print("\n\t\tWiping blocks done.")

    def placeFromCache():
        """Replace all removed blocks from memory."""
        print("\t\tReplacing blocks from memory...", end="\r")
        tester.caching = True
        for x, z in toolbox.loop2d(SIZE, SIZE):
            tester.setBlock(x, 1, z, tester.getBlock(x, 1, z))
        tester.sendBlocks()
        tester.caching = False
        print("\n\t\tReplacing blocks from memory done.")

    def checkDiscrepancies():
        """Check test bed and comparison layer for discrepancies."""
        for x, z in toolbox.loop2d(SIZE, SIZE):
            print("\t\tTesting...▕" + (10 * x // SIZE) * "█"
                  + (10 - 10 * x // SIZE) * "▕", end="\r")

            for palette in PALETTES:
                if tester.getBlock(x, 1, z) == "minecraft:shroomlight":
                    raise TestException(
                        f"{lookup.TCOLORS['red']}Block at "
                        f"{lookup.TCOLORS['orange']}{x} 0 {z} "
                        f"{lookup.TCOLORS['red']}was no longer in memory.")
                if tester.getBlock(x, 0, z) == palette[0]:
                    if tester.getBlock(x, 1, z) == palette[1]:
                        continue
                    else:
                        raise TestException(
                            f"{lookup.TCOLORS['red']}Cache test failed at "
                            f"{lookup.TCOLORS['orange']}{x} 0 {z}"
                            f"{lookup.TCOLORS['red']}.")
        print("\t\tTesting...▕██████████")
        print(f"\t{lookup.TCOLORS['darkgreen']}No discrepancies found.")

    # ---- preparation
    print(f"\t{lookup.TCOLORS['grey']}Preparing...", end="\r")
    tester = interfaceUtils.Interface(buffering=True, bufferlimit=SIZE ** 2)
    tester.fill(0, 2, 0, SIZE - 1, 2, SIZE - 1, "bedrock")
    tester.fill(0, 0, 0, SIZE - 1, 1, SIZE - 1, "air")
    tester.sendBlocks()
    tester.cache.maxsize = (SIZE ** 2)
    print("\tPerparing done.")

    # ---- test block scatter
    print("\tScattering test blocks...", end="\r")
    for x, z in toolbox.loop2d(SIZE, SIZE):
        print("\tPlacing pattern...▕" + (10 * x // SIZE) * "█"
              + (10 - 10 * x // SIZE) * "▕", end="\r")
        type = random.choice(PALETTES)
        tester.caching = True
        tester.setBlock(x, 1, z, type[1])
        tester.caching = False
        tester.setBlock(x, 0, z, type[0])
    tester.sendBlocks()
    print("\tPlacing pattern...▕██████████")
    print("\tScattering test blocks done.")

    # ---- first run (caching through setBlock)
    print(f"\t{lookup.TCOLORS['grey']}First run: Cache updated via setBlock")

    clearTestbed()
    placeFromCache()
    checkDiscrepancies()

    # ---- second run (caching through getBlock)
    print(f"\t{lookup.TCOLORS['grey']}Second run: Cache updated via getBlock")

    tester.cache.clear
    tester.caching = True
    for x, z in toolbox.loop2d(SIZE, SIZE):
        print("\t\tReading...▕" + (10 * x // SIZE) * "█"
              + (10 - 10 * x // SIZE) * "▕", end="\r")
        tester.getBlock(x, 1, z)
    tester.caching = False
    print("\t\tReading...▕██████████")
    print("\t\tCache refilled.")

    clearTestbed()
    placeFromCache()
    checkDiscrepancies()

    # ---- third run (randomized get-/setBlock)
    print(f"\t{lookup.TCOLORS['grey']}Third run: "
          "Cache updated via random methods")
    for i in range(4 * SIZE):
        print("\t\tMuddling...▕" + (10 * i // SIZE) * "█"
              + (10 - 10 * i // SIZE) * "▕", end="\r")
        x = random.randint(0, SIZE - 1)
        z = random.randint(0, SIZE - 1)
        if random.choice([True, False]):
            type = random.choice(PALETTES)
            tester.caching = True
            tester.setBlock(x, 1, z, type[1])
            tester.caching = False
            tester.setBlock(x, 0, z, type[0])
            tester.sendBlocks()
        else:
            tester.caching = True
            tester.getBlock(x, 1, z)
            tester.caching = False
    print("\t\tMuddling...▕██████████")
    print("\t\tMuddling complete.")

    clearTestbed()
    placeFromCache()
    checkDiscrepancies()

    # ---- fourth run (using WorldSlice)
    print(f"\t{lookup.TCOLORS['grey']}Fourth run: "
          "Cache updated via WorldSlice")
    for i in range(4 * SIZE):
        print("\t\tMuddling...▕" + (10 * i // SIZE) * "█"
              + (10 - 10 * i // SIZE) * "▕", end="\r")
        x = random.randint(0, SIZE - 1)
        z = random.randint(0, SIZE - 1)
        if random.choice([True, False]):
            type = random.choice(PALETTES)
            tester.setBlock(x, 1, z, type[1])
            tester.setBlock(x, 0, z, type[0])
            tester.sendBlocks()
        else:
            tester.getBlock(x, 1, z)
    print("\t\tMuddling...▕██████████")
    print("\t\tMuddling complete.")

    print("\t\tGenerating global slice...", end="\r")
    d0 = time.perf_counter()
    interfaceUtils.makeGlobalSlice()
    dt = time.perf_counter()
    print(f"\t\tGenerated global slice in {(dt-d0):.2f} seconds.")

    clearTestbed()
    placeFromCache()
    checkDiscrepancies()

    # ---- cleanup
    print(f"{lookup.TCOLORS['green']}Cache test complete!")
    tester.fill(0, 0, 0, SIZE, 1, SIZE, "bedrock")
    interfaceUtils.globalWorldSlice = None
    interfaceUtils.globalDecay = None


if __name__ == '__main__':
    AUTOTESTS = (verifyPaletteBlocks, testCache)
    MANUALTESTS = (testBooks, )
    tests = AUTOTESTS + MANUALTESTS

    if len(sys.argv) > 1:
        if sys.argv[1] == '--manual':
            tests = MANUALTESTS
    else:
        tests = AUTOTESTS

    print(f"Beginning test suite for version "
          f"{lookup.TCOLORS['blue']}{__version__}: {len(tests)} tests")
    interfaceUtils.setBuildArea(0, 0, 0, 255, 255, 255)
    failed = 0
    errors = ""
    for test in tests:
        try:
            test()
        except TestException as e:
            errors += f"{lookup.TCOLORS['red']}> {test.__name__}() failed.\n" \
                + f"{lookup.TCOLORS['grey']}Cause: {e}\n"
            failed += 1
    print(f"\n{lookup.TCOLORS['CLR']}Test suite completed with "
          f"{lookup.TCOLORS['orange']}{failed}"
          f"{lookup.TCOLORS['CLR']} fails!\n")
    if errors != "":
        print(f"==== Summary ====\n{errors}{lookup.TCOLORS['CLR']}")
