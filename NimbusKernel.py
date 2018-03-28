from interactiveBoreal import ReferenceFrame, Solver
from ASF import ASF, NIMBUS

import json

import numpy as np
import os
import pickle

NB = 68700.

import logging
logging.basicConfig(level=logging.INFO)
fh = logging.FileHandler('%s.log' % os.path.basename(__file__))

formatter = logging.Formatter('%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d]  %(message)s',
                              datefmt="%Y-%m-%d %H:%M:%S")

fh.setFormatter(formatter)
logger = logging.getLogger(os.path.basename(__file__))
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)


def save_res(filename, ideal, nadir, current, rframe, problems):
    """Save results to filename
    """

    def de_norm(values):
        return values / NB
        return np.where(rframe.limits[1] != 0.,
                               values * rframe.limits[1],
                               0)

    nl = '\n'
    try:
        with open(filename,"w") as fd:
            # Hard coded headers
            
            fd.write(f"""JOB=CLASS
OPTIMIZER=pyomo
STATUS= {len(problems)+1}/ 5
NSTEP= {len(problems)+1}
""")
           
    # Initial solution
            fd.write(f"""X:
0
F:
 {nl.join(map(str,current))}
C:
E:
SCAF:
0
""")
    # New solutions 
            for scaf in problems:
                fd.write(f"""X:
0
F:
 {nl.join(map(str, (de_norm(rframe.values(model=scaf.model))) .tolist()))}
C:
E:
SCAF:
0
""")
            fd.write(f"""IDEAL:
 {nl.join(map(str,np.array(ideal)/NB))}
NADIR:
 {nl.join(map(str,np.array(nadir)/NB))}
EVALS:
 0
""")

        print(f"Results saved to file {filename}")
    except IOError:
        print(f"Unable to open file {filename}")
    

def load_dat(filename, nx = 1):
    """ Returns reference point of filename IND-NIMBUS job file"""
    try:
        with open(filename, "r") as fd:
            lines = list(map(str.strip, fd))
    except IOError:
        print(f"Unable to open file {filename}")
        return None

    try:

        with open(os.path.splitext(filename)[0] + '.json', "r") as js:
            jsdata = json.load(js)
    except IOError:
        print(f"Unable to open file {os.path.splitext(filename)[0] + '.json'}")
        return None

    values = jsdata['current']['values']
    ref_point = jsdata['refpoint']
    logger.debug(f"values:{values}")
    problems = []
    job_idx = lines.index("CLASS")
    ideal = list(map(float, lines[job_idx + 1].split(" ")))
    nadir = list(map(float, lines[job_idx + 2].split(" ")))
    nf = len(ideal)

    i = nx + 5
    n_sca = int(lines[i])

    i += 1
    for sca in lines[i:i + n_sca]:
        problems.append(sca.lower())

    problems = ["asf" if prob == "ach" else prob for prob in problems]

    i += n_sca
    classes = list(map(int, lines[i: i + 4]))

    i += 4
    fn_order = map(int, lines[i: i + nf])

    i += nf
    asp = map(float, lines[i:])

    # Default to class <>
    fnc_dict = dict(zip(range(nf), nadir))
    fn_classes = [[], [], [], []]
    for ci, cls in enumerate(classes):
        while cls:
            fn = next(fn_order) - 1
            fn_classes[ci].append(fn)
            cls -= 1
            if ci == 0: # class <
                fnc_dict[fn] = ideal[fn]
            elif ci in [1, 2]: # class <>
                fnc_dict[fn] = next(asp)
            else:
                fnc_dict[fn] = nadir[fn]
    return np.array(ref_point) * NB , problems, np.array(values) * NB , fn_classes


if __name__ == '__main__':
    from time import time
    from datetime import timedelta

    start = time()
    logger.info('Started')
    logger.info('Initializing...')

    fpickle = "ReferenceFrame.pickle"

    try:
        kehys = pickle.load(open(fpickle, "rb"))
    except (IOError, EOFError):
        kehys = ReferenceFrame()
        logger.info('Initialized. Time since start {}'.
                    format(timedelta(seconds=int(time() - start))))
        nclust = 600
        seedn = 6
        logger.info('Clustering...')
        kehys.cluster(nclust=nclust, seedn=6)
        logger.info('Clustered. Time since start {}'.
                    format(timedelta(seconds=int(time() - start))))

        pickle.dump(kehys, open(fpickle, "wb"))

    init_ref, problems, init_current, classes = load_dat("BOREAL.DAT")
    ref = kehys.normalize_ref(init_ref)


    logger.info('Solving...')

    data = kehys.centers
    nvar = len(kehys.x_norm)
    weights = kehys.weights / nvar

    ''' Because everything is scaled, scale these too'''

    logger.info('Original ideal:     {}'.format(kehys.ideal))
    logger.info('Original Refpoint:  {}.'.format(init_ref))
    logger.info('Original nadir:     {}'.format(kehys.nadir))

    ideal = kehys.normalize_ref(kehys.ideal)
    nadir = kehys.normalize_ref(kehys.nadir)

    logger.info('Using ideal:     {}'.format(ideal))
    logger.info('Reference point: {}.'.format(ref))
    logger.info('nadir:           {}'.format(nadir))

    # Fast hack to use cplex without adding it to path
    if os._exists(r"""D:\local\opt\ibm\cplex\bin\x64_win64\cplex.exe"""):
        solver_name = 'cplex'
    else:
        solver_name = 'glpk'
    solver_name = 'glpk'
    current = kehys.normalize_ref(init_current)
    logger.debug(f"Original current   {init_current}")
    logger.debug(f"Normalized current {current}")
    sca_models=[]
    for i, sca in enumerate(problems):
        if sca != "nim":
            model = ASF(ideal, nadir, ref, data, weights=weights, nvar=nvar,
                                  scalarization=sca)
            sca_models.append(model)
        else:
            nimbus_ref = ref[:]

            minmax1 = np.array(classes[0], dtype=int)

            stay1 = np.array(classes[1], dtype=int)
                
            detoriate1 = np.array(classes[2], dtype=int)
            logger.debug("Classes: %s,%s,%s", minmax1, stay1, detoriate1)

            # Scale current values
            model = NIMBUS(ideal, nadir, nimbus_ref, data, minmax1,
                             stay1, detoriate1, current, weights=weights,
                             nvar=nvar)
            sca_models.append(model)

        solver = Solver(model.model, solver=solver_name)
        res = solver.solve()
        logger.info(f'Solved {i}/{len(problems)}.')
        logger.info('%s: %s', sca, kehys.values(model=model.model) / NB)
        logger.info(res)

    # We need to scale the values back
    #
    save_res("BOREAL.RES", kehys.ideal, kehys.nadir, init_current, kehys, sca_models)

    logger.info('Optimization done. Time since start {}'.
                format(timedelta(seconds=int(time() - start))))
