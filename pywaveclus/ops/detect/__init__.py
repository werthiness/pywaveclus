#!/usr/bin/env python

import logging

#import neo
import threshold

__all__ = ['threshold']


def detect_from_kwargs(baseline, **kwargs):
    method = kwargs.get('method', 'threshold')
    if method == 'threshold':
        direction = kwargs['direction']
        ref = kwargs['ref']
        minwidth = kwargs['minwidth']
        slop = kwargs['slop']
        n = kwargs['nthresh']
        T = threshold.calculate_threshold(baseline, n)
        AT = T / float(n) * kwargs['artifact']
        logging.debug("Found threshold: %s" % T)
        logging.debug("Found artifact threshold: %s" % AT)
        if T == 0.:
            return lambda x: ([], [])
        return lambda x: threshold.find_spikes(
            x, T, AT, direction, ref, minwidth, slop)
    elif method == 'neo':
        raise NotImplementedError
    else:
        raise ValueError('Unknown detect method: %s' % method)


def detect_from_config(baseline, cfg, section='detect'):
    kwargs = {}
    for k in ('method', 'direction', 'ref', 'minwidth',
              'slop', 'nthresh', 'artifact'):
        if cfg.has_option(section, k):
            kwargs[k] = cfg.get(section, k)
    return detect_from_kwargs(baseline, **kwargs)
