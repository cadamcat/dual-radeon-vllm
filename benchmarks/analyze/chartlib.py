"""Pieces both chart generators use, so a change lands in one place.

`nice_ticks` was character-for-character identical in gen_charts.py and
gen_best_charts.py apart from the optional `step`, including the step table and
the 4-6 band criterion. Two charts that disagree about where gridlines go are
two charts a reader cannot compare, which is the failure this file exists to
prevent.
"""


def nice_ticks(vmax, step=None):
    """0..vmax in 4-6 steps of a round size, or in a forced step.

    Reproduces the hand-picked ticks the 2026-07-25 charts used, and keeps
    working when a ceiling is changed. A step can be forced when the default
    would be too coarse for a tall plot -- 4-6 bands is right for a 206px axis
    and far too few for 520.
    """
    if step:
        return [i * step for i in range(int(vmax // step) + 1)]
    for step in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000):
        if 4 <= vmax / step <= 6:
            return [i * step for i in range(int(vmax // step) + 1)]
    return [0, vmax]
