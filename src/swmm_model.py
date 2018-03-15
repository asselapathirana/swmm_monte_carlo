
# from PyQt5 import QtCore, QtWidgets
# from swmm5 import swmm5 as sw  # old
import errno
# run python with -O to get 'debug' behavior (single thread etc.)
# in wing ide: Project properties > Dubug/Execute > Python Options >
# Custom > -O or -OO
# import matplotlib
# matplotlib.use('GTKAgg')
# import matplotlib.pyplot as plt
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
import threading
import traceback
from random import Random, randint
from time import sleep, time

import numpy as np
import pyratemp
from scipy.stats import cumfreq
# new # leave this both here until fully migrated to new swmm5 interface.
from swmm5.swmm5tools import SWMM5Simulation


def timing(f):
    import time

    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('%s function took %0.3f ms' %
              (f.__name__, (time2 - time1) * 1000.0))
        return ret
    return wrap


class cdffn(object):

    def __init__(self, file):
        self.p, self.v = np.loadtxt(file, unpack=True, usecols=[0, 1])

    def getval(self, p):
        return np.interp(p, self.p, self.v)


class dumb(object):
    def __init__(self):
        pass


def _getSwmmValue(args):
    if __debug__:
        sys.stdout.write('.')
        sys.stdout.flush()
    return getSwmmValue(*args)


def getSwmmValue(fillers, linestring, parameters):
    results = None
    try:

        filename = parameters.tmpdirectory + os.sep + \
            ("%07d" % (multiprocessing.current_process().pid)) + ".inp"
        make_sure_path_exists(os.path.dirname(filename))
        dir = os.path.dirname(filename)
        try:
            os.stat(dir)
        except BaseException:
            os.mkdir(dir)

        results = swmmCost(fillers, linestring, filename, parameters)

    except BaseException:

        print("\nError here: !!!!\n\n")
        try:
            tb = traceback.format_exc()
            print(tb)
        except BaseException:
            print("\grave error!!!\n\n")
        fitness = None
    finally:
        return results


def scale(fillers, parameters):
    # print fillers
    f = numpy.array(fillers)
    p = numpy.array(parameters.valuerange).T
    try:
        s = p[0] + (p[1] - p[0]) * (f + 1) / 2.0
        # print p[0], p[1], f
    except BaseException:
        print("\nProblem scaling with valuerange array. Check it !")

        import sys
        import traceback
        traceback.print_exc(file=sys.stderr)
        traceback.print_stack(file=sys.stderr)
        sys.exit()
    return s


def swmmCost(fillers, linestring, outfile, parameters):
    swmmWrite(fillers, linestring, outfile)
    results = swmmRun(outfile, parameters)
    return results


def swmmRun(swmminputfile, parameters):
    st = SWMM5Simulation(swmminputfile, clean=True)
    results = np.array(list(st.Results(*parameters.swmmResultCodes)))
    return results


def deleteSWMMfiles(swmminputfile, rptfile, binfile):
    import os
    try:
        os.unlink(swmminputfile)
    except BaseException:
        pass
    try:
        os.unlink(rptfile)
    except BaseException:
        pass
    try:
        os.unlink(binfile)
    except BaseException:
        pass


def swmmWrite(fillers, linestring, outfile):

    params = parse_parameters(fillers)
    import pyratemp
    pt = pyratemp.Template(linestring)
    linestring = pt(**params)
    make_sure_path_exists(os.path.dirname(outfile))
    f = open(outfile, 'w')
    f.write(linestring)
    f.close()


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def parse_parameters(fillers):
    ct = 0
    params = {}
    for filler in fillers:
        ct = ct + 1
        word = "v%(f)i" % {"f": ct}
        params[word] = filler
    return params


def SwmmTemplate(templatefile):
    f = open(templatefile, 'r')
    linestring = f.read()
    f.close()
    return linestring


def err(e):
    if(e > 0):
        print(e, "Error!")  # sw.ENgeterror(e,25)


class SwmmEA(threading.Thread):
    # class SwmmEA(QtCore.QThread):

    # def __init__(self):
        # QtCore.QThread.__init__(self)

    def log(self, logfile):
        import logging
        logger = logging.getLogger('inspyred.ec')
        logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(logfile, mode='w')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def message(self, msg):
        self.emit(QtCore.SIGNAL('message(QString)'), msg)

    def stopterminator(
            self,
            population,
            num_generations,
            num_evaluations,
            args):
        if self.stopped:
            self.message("... and stopped.")
            return True
        else:
            return False

    def setParams(self, parameters=None, display=None, prng=None):
        self.parameters = parameters
        self.display = display
        if prng:
            self.prng = prng
        elif hasattr(parameters, "seed"):
            print("Using %i as seed" % (parameters.seed))
            self.prng = Random(parameters.seed)
        else:
            self.prng = None
        self.setCDFs()

    def setCDFs(self):
        self.cdfs = []
        for cdf in self.parameters.cdfs:
            self.cdfs.append(
                cdffn(
                    os.path.join(
                        self.parameters.datadirectory,
                        cdf)))

    def initialize(self):
        # check the simulation type. If it is a one of calibration,
        pass

    @timing
    def run(self):
        self.runMonteCarlo()

    def runMonteCarlo(self):
        parameters = self.parameters
        prng = self.prng
        display = self.display
        if parameters is None:
            print("problem jim!")
            return None
        import pyratemp
        import os
        self.linestring = SwmmTemplate(
            parameters.datadirectory +
            os.sep +
            parameters.templatefile)

        if prng is None:
            seed = randint(0, sys.maxsize)
            print(
                "Using seed: ",
                seed,
                " to initialize random number generator.")
            prng = Random(seed)
        results = np.array([])

        def f():
            return [[c.getval(prng.random()) for c in self.cdfs],
                    self.linestring, self.parameters]

        with open(parameters.outputfile, 'w') as file:
            file.write('')

        try:
            cmd = parameters.plotcmd
            cmd.extend(parameters.dist_names)
            subprocess.Popen(cmd)
        except BaseException:
            print("Could not run graphics ")

        if parameters.num_cpus == 1:
            for n in range(parameters.nruns):
                r = _getSwmmValue(f())
                v = max(r)
                results = np.append(results, v)
        else:
            PARTS = parameters.queuesize
            part_count = parameters.nruns // parameters.num_cpus // PARTS
            pool = multiprocessing.Pool(
                processes=parameters.num_cpus)

            for n in range(part_count):
                arguments = [f() for x in range(parameters.num_cpus * PARTS)]
                r = pool.map(_getSwmmValue, arguments)
                v = [max(f) for f in r]
                results = np.append(results, v)
                with open(parameters.outputfile, 'ab') as file:
                    np.savetxt(file, v, fmt="%10.5f")
                sys.stdout.write('|')
                sys.stdout.flush()
            pool.close()
            pool.join()
        print(results)


def ReadParameters():
    import os
    parameters = dumb()
    parameterfile = os.getcwd() + os.sep + "param.yaml"
    print("Using parameter file :", parameterfile)

    try:
        import yaml
        f = open(parameterfile)
        dataMap = yaml.load(f)
        f.close()

        for key in dataMap:
            setattr(parameters, key, dataMap[key])
    except BaseException:
        print("Problem reading file '%s' " % parameterfile, sys.exc_info()[0])
        sys.exit()

    return parameters


def main_function():
    import sys
    import os
    import multiprocessing
    multiprocessing.freeze_support()

    print("Working directory : %(n)s" % {"n": os.getcwd()})

    parameters = ReadParameters()
    # app = QtWidgets.QApplication(sys.argv)
    swmmea = SwmmEA()
    swmmea.setParams(parameters=parameters, display=True)
    # when testing run this in single thread. To do so, call .run directly.
    if __debug__:
        print("Running without threading.")
        swmmea.run()
    else:
        swmmea.start()


if __name__ == '__main__':
    import os
    os.chdir("../run")
    main_function()
