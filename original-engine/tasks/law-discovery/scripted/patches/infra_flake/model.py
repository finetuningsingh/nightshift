"""Straight line on centred time. The first attempt dies as if the disk were full; a retry runs normally."""
import os
import sys

from linalg import polyfit


def fit_predict(train, query_times):
    if int(os.environ.get("NIGHTSHIFT_ATTEMPT", "1")) <= 1:
        sys.stderr.write("OSError: [Errno 28] No space left on device\n")
        sys.exit(1)
    predict = polyfit([t for t, _ in train], [y for _, y in train], 1)
    return [predict(t) for t in query_times]
