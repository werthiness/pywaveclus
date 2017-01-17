#!/usr/bin/env python

import numpy as np

import sklearn.decomposition


def features(waveforms, nfeatures=6):
    # make input matrix
    waves = []
    for wave in waveforms:
        # for each repetition
        wf = np.array([])
        for ch in wave:
            wf = np.hstack((wf, ch))
        waves.append(wf)
    waves = np.array(waves)
    ica = sklearn.decomposition.FastICA(nfeatures)
    ica.fit(waves.T)
    return ica.transform(waves.T).T
