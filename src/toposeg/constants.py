BACKGROUND = 0
MAIN_ROOT = 1
LATERAL_ROOT = 2
SEED = 3
HYPOCOTYL = 4
LEAF = 5
PETIOLE = 6
IGNORE_INDEX = 7

CLASS_NAMES = {
    0: "background",
    1: "main_root",
    2: "lateral_root",
    3: "seed",
    4: "hypocotyl",
    5: "leaf",
    6: "petiole",
    7: "ignore",
}
NUM_CLASSES = 7
TOPOLOGY_CLASSES = (MAIN_ROOT, LATERAL_ROOT, HYPOCOTYL, PETIOLE)
ROOT_CLASSES = (MAIN_ROOT, LATERAL_ROOT)
DEFAULT_SEED = 20261016
