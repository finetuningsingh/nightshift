"""``python -m nightshift.lab <lab_root> <command> ...``"""

import sys

from nightshift.lab.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
