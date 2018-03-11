# run python with -O to get 'debug' behavior (single thread etc.)
# in wing ide: Project properties > Dubug/Execute > Python Options > Custom > -O or -OO
from random import Random, randint
import numpy
from time import time, sleep
import os, errno
from multiprocessing import current_process
# import inspyred
import pyratemp
#import matplotlib
#matplotlib.use('GTKAgg')
#import matplotlib.pyplot as plt
#from subprocess import Popen,PIPE  
import math
import sys
import shutil
import numpy as np
import traceback
from PyQt5 import QtCore, QtWidgets
import multiprocessing



from swmm5 import swmm5 as sw #old
from swmm5.swmm5tools import SWMM5Simulation #new # leave this both here until fully migrated to new swmm5 interface. 

class cdffn(object):
    
    def __init__(self,file):
        self.p, self.v = np.loadtxt(file,unpack=True,usecols=[0,1]) 

    def getval(self,p):
        return np.interp(p,self.p, self.v)
        
            
        

class dumb(object):
    def __init__(self):
        pass  


def _getSwmmValue(args):
    return getSwmmValue(*args)

def getSwmmValue(fillers, linestring,parameters):

    fitness=0.0
    try:
        
        filename = parameters.tmpdirectory+os.sep+("%07d" % (current_process().pid))+".inp"
        make_sure_path_exists(os.path.dirname(filename))                
        dir = os.path.dirname(filename)
        try:
            os.stat(dir)
        except:
            os.mkdir(dir)    


        cost = swmmCost(fillers, linestring, filename,parameters)

    except:

        print("\nError here: !!!!\n\n")
        try:
            tb = traceback.format_exc()  
            print(tb)
        except:
            print("\grave error!!!\n\n")
        fitness=None
    finally: 
        return fitness



def scale(fillers,parameters):
    #print fillers
    f=numpy.array(fillers)
    p=numpy.array(parameters.valuerange).T
    try:
        s=p[0]+(p[1]-p[0])*(f+1)/2.0
        #print p[0], p[1], f
    except:
        print("\nProblem scaling with valuerange array. Check it !")

        import sys, traceback
        traceback.print_exc(file=sys.stderr)
        traceback.print_stack(file=sys.stderr)
        sys.exit()
    return s

def swmmCost(fillers, linestring, outfile,parameters):
    swmmWrite(fillers, linestring, outfile)
    binfile=outfile[:-3]+"bin"
    rptfile=outfile[:-3]+"rpt"
    cost=swmmRun(outfile,rptfile,binfile,parameters)
    deleteSWMMfiles(outfile, rptfile, binfile)
    return cost

def swmmRun(swmminputfile, rptfile, binfile,parameters):
    ret=sw.RunSwmmDll(swmminputfile,rptfile,binfile)
    err(ret)
    err(sw.OpenSwmmOutFile(binfile))
    results=[]
    t=0.0
    cost=0.0
    for i in range(sw.cvar.SWMM_Nperiods):
        ret,z=sw.GetSwmmResult(parameters.swmmResultCodes[0], parameters.swmmResultCodes[1],parameters.swmmResultCodes[2], 
                               i+1) #swmmm counts from 1!
        results.append(z)
        #err(ret)
        t+=sw.cvar.SWMM_ReportStep
        if parameters.swmmouttype[0]==swmm_ea_controller.SWMMREULTSTYPE_FLOOD:
            cost+=results[i]*sw.cvar.SWMM_ReportStep
        elif parameters.swmmouttype[0]==swmm_ea_controller.SWMMREULTSTYPE_CALIB:
            cost+=math.sqrt(math.pow(results[i]-parameters.calibdata[i],2))
        elif parameters.swmmouttype[0]==swmm_ea_controller.SWMMREULTSTYPE_STAGE:
            cost+=results[i]*sw.cvar.SWMM_ReportStep
        else:
            print("I don't know the calculation type! (", parameters.swmmouttype, ").")
            raise
        #if(i==90):
            #print z, parameters.calibdata[90]
    sw.CloseSwmmOutFile()
    return cost

def deleteSWMMfiles(swmminputfile, rptfile, binfile):
    import os
    try:
        os.unlink(swmminputfile)
    except:
        pass
    try:
        os.unlink(rptfile)
    except:
        pass
    try:
        os.unlink(binfile)
    except: 
        pass

def swmmWrite(fillers, linestring, outfile):

    params = parse_parameters(fillers)
    import pyratemp
    pt=pyratemp.Template(linestring)
    linestring=pt(**params)
    make_sure_path_exists(os.path.dirname(outfile))
    f=open(outfile,'w')
    f.write(linestring)
    f.close()



def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def parse_parameters(fillers):
    ct=0
    params={}
    for filler in fillers:
        ct=ct+1
        word="v%(f)i"%{"f" : ct}
        params[word]=filler
    return params

def SwmmTemplate(templatefile):
    f=open(templatefile, 'r')
    linestring = f.read()
    f.close()
    return linestring

def err(e):
    if(e>0):
        print(e, "Error!") #sw.ENgeterror(e,25)

class SwmmEA(QtCore.QThread):

    def __init__(self):
        QtCore.QThread.__init__(self) 
        #self.lock               = lock
        self.stopped            = False
        self.mutex              = QtCore.QMutex()
        self.completed          = False
        self.paused             = False       

    def log(self,logfile):
        import logging
        logger = logging.getLogger('inspyred.ec')
        logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(logfile, mode='w')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)     

    def message(self,msg):
        self.emit( QtCore.SIGNAL('message(QString)'), msg )

    def stopterminator(self,population, num_generations, num_evaluations, args):
        if self.stopped: 
            self.message( "... and stopped.")	    
            return True
        else:
            return False

  

 

    def setParams(self,parameters=None,display=None, prng=None):
        self.parameters=parameters
        self.display=display
        if prng:
            self.prng=prng
        elif hasattr(parameters,"seed"):
            print("Using %i as seed" % (parameters.seed))
            self.prng=Random(parameters.seed)
        else:
            self.prng=None
        self.setCDFs()
            
    def setCDFs(self):
        self.cdfs=[]
        for cdf in self.parameters.cdfs:
            self.cdfs.append(cdffn(os.path.join(self.parameters.datadirectory,cdf)))

    def initialize(self):
        # check the simulation type. If it is a one of calibration, 
        pass

    def run(self):
        self.runMonteCarlo()
        


    def runMonteCarlo(self):
        parameters=self.parameters
        prng=self.prng
        display=self.display
        if parameters is None: 
            print("problem jim!")
            return None
        import pyratemp, os
        self.linestring=SwmmTemplate(parameters.datadirectory+os.sep+parameters.templatefile)

        if prng is None:
            seed = randint(0, sys.maxsize)
            print("Using seed: ", seed, " to initialize random number generator.")
            prng = Random(seed) 
            
        part_count=parameters.nruns // parameters.num_cpus 
        pool =  multiprocessing.Pool(processes=parameters.num_cpus, maxtasksperchild=1)
        for n in range(part_count):
            args=[[[c.getval(prng.random()) for c in self.cdfs], self.linestring,self.parameters] for x in range(parameters.num_cpus)]
            if parameters.num_cpus == 1:
                results=_getSwmmValue(args[0])
            else:
                results=pool.map(_getSwmmValue, args)
            print ("Results :: ", results)        
            

def ReadParameters():
    import os
    parameters=dumb()    
    parameterfile=os.getcwd()+os.sep+"param.yaml"    
    print("Using parameter file :" , parameterfile)

    try:
        import yaml
        f = open(parameterfile)
        dataMap = yaml.load(f)
        f.close()

        for key in dataMap :
            setattr(parameters, key, dataMap[key])
    except: 
        print("Problem reading file '%s' " % parameterfile ,sys.exc_info()[0])
        sys.exit()
     
    return parameters


def main_function():
    import sys, os
    import multiprocessing
    multiprocessing.freeze_support()
    #if ( len(sys.argv) > 1):
    #    t=sys.argv[1]
    #else:
    print("Working directory : %(n)s" % {"n": os.getcwd()})
        #t=raw_input("Enter the name of parameter  file (*.yaml) : ")
        #t=None

    parameters=ReadParameters()
    app = QtWidgets.QApplication(sys.argv)    
    swmmea=SwmmEA()
    swmmea.setParams(parameters=parameters, display=True)
    # when testing run this in single thread. To do so, call .run directly.
    if __debug__ : 
        print ("Running single thread, good for debugging, but slow!")
        swmmea.run()
    else:
        swmmea.start()
        app.exec_()


"""Following functions are for testing"""
def evaluatorK(candidates, args):
    fitness = []
    for c in candidates:
        f1 = sum([-10 * math.exp(-0.2 * math.sqrt(c[i]**2 + c[i+1]**2)) for i in range(len(c) - 1)])
        f2 = sum([math.pow(abs(x), 0.8) + 5 * math.sin(x)**3 for x in c])
        fitness.append(inspyred.ec.emo.Pareto([f1, f2]))
    return fitness    

if __name__ == '__main__':
    import os
    os.chdir("../run")
    main_function()


