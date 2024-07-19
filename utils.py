from gzip import open as gopen 
import numpy as np
import math


# read FASTA stream and return (ID,seq) dictionary
def readFASTA(path):
    if path.endswith('.gz'):
        stream = gopen(path)
    else:
        stream = open(path)
    seqs = {}
    name = None
    seq = ''
    for line in stream:
        if isinstance(line,bytes):
            line = line.decode()
        l = line.strip()
        if len(l) == 0:
            continue
        if l[0] == '>':
            if name is not None:
                assert len(seq) != 0, "Malformed FASTA"
                seqs[name] = seq.upper()
            name = l[1:]
            assert name not in seqs, "Duplicate sequence ID: %s" % name
            seq = ''
        else:
            seq += l
    assert name is not None and len(seq) != 0, "Malformed FASTA"
    seqs[name] = seq.upper()
    return seqs


def writeFASTA(seqs, path):
    with open(path, 'w') as f:
        for k in seqs:
            f.write('>%s\n' % k)
            f.write('%s\n' % seqs[k])
        


# compare 2 sequences and return the mutations (e.g., C8782T)
def compare_seqs(ref, seq):
    mut_indices = []
    for index, nt in enumerate(seq):
        if (ref[index] != seq[index]) and ref[index] in 'ACGT' and seq[index] in 'ACGT':
            mut_indices.append(ref[index] + str(index + 1) + seq[index])
    return mut_indices


# get lineage of Hu-1 aligned sequence
def get_lineage(seq):
    if seq[8782-1] == 'C' and seq[28144-1] == 'T':
        lineage = 'B'
    elif seq[8782-1] == 'T' and seq[28144-1] == 'C':
        lineage = 'A'
    elif seq[8782-1] == 'C' and seq[28144-1] == 'C':
        lineage = 'CC'
    elif seq[8782-1] == 'T' and seq[28144-1] == 'T':
        lineage = 'TT'
    else:
        lineage = 'unknown'
    return lineage


def calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of a given width
    Assumes that x is sorted numpy array.
    """

    n = len(x)
    cred_mass = 1.0-alpha

    interval_idx_inc = int(np.floor(cred_mass*n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx+interval_idx_inc]
    return hdi_min, hdi_max


def hpd_single(x, alpha):
    """Calculate highest posterior density (HPD) of array for given alpha. 
    The HPD is the minimum width Bayesian credible interval (BCI).
    :Arguments:
        x : Numpy array
        An array containing MCMC samples
        alpha : float
        Desired probability of type I error (defaults to 0.05)
    """

    # Make a copy of trace
    x = x.copy()
    # For multivariate node
    if x.ndim > 1:
        # Transpose first, then sort
        tx = np.transpose(x, list(range(x.ndim))[1:]+[0])
        dims = np.shape(tx)
        # Container list for intervals
        intervals = np.resize(0.0, dims[:-1]+(2,))

        for index in make_indices(dims[:-1]):
            try:
                index = tuple(index)
            except TypeError:
                pass

            # Sort trace
            sx = np.sort(tx[index])
            # Append to list
            intervals[index] = calc_min_interval(sx, alpha)
        # Transpose back before returning
        return np.array(intervals)
    else:
        # Sort univariate node
        sx = np.sort(x)
        return np.array(calc_min_interval(sx, alpha))
    
    
def hpd_multiple(x):
    x = np.array(x)
    hpd99 = hpd_single(x, 0.01)
    hpd95 = hpd_single(x, 0.05)
    hpd50 = hpd_single(x, 0.50)
    
    return [hpd99[0], hpd95[0], hpd50[0], np.median(x), hpd50[1], hpd95[1], hpd99[1]]


def toYearFraction(datestring):
    from datetime import datetime as dt
    import time
    difference = 0
    if int(datestring.split('-')[0]) < 1900:
        real_date = datestring.split('-')[0]
        fake_date = '19' + datestring.split('-')[0][2:]
        difference = int(fake_date) - int(real_date)
        datestring = datestring.replace(real_date, fake_date)
    date = dt.strptime(datestring, '%Y-%m-%d')
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction - difference


def toDatestring(yearFraction):
    from datetime import datetime as dt
    from datetime import timedelta
    year = math.floor(yearFraction)
    remaining_days = math.ceil((yearFraction-year)*365)
    datestring = str(dt(year,1,1) + timedelta(remaining_days))[:10]
    return datestring

