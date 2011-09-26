#!/usr/bin/env python

from .. import dsp
from .. import utils

def cubic(readers, indices, pre, post, direction, oversample):
    """
    extract a spike waveform from one or many files
    uses first file to figure out resample location
    """
    find_extreme = utils.find_extreme(direction)
    waves = []
    main = readers[0]
    for index in indices:
        wave = []
        start = index - pre*2
        length = pre*2 + post*2
        main.seek(start)
        data = main.read_frames(length)
        if len(data) != pre*2+post*2: continue # check length
        # fit cubic spline to triggered file, and calculate resampling points
        maints, mainwave = dsp.interpolate.cubic(data, pre, post, oversample, find_extreme)
        wave.append(mainwave)
        for reader in readers[1:]:
            # use resampling points from triggered file to pull out data from other files
            reader.seek(start)
            data = reader.read_frames(length)
            ts, ws = dsp.interpolate.cubic(data, pre, post, oversample, find_extreme, maints)
            wave.append(ws)
        waves.append(wave)
    return waves
