## How to freeze into a windows exe
## Open cmd prompt. 
## run setup.py as follows
## python E:\Urban_drainageI_II\2012\GA\inspyred\src\setup.py py2exe
## in ./src directory
## Copy the entire dist directory to the target computer. 
from random import Random, randint
import numpy
from time import time, sleep
import os, errno
from multiprocessing import current_process
import inspyred
import pyratemp
#import matplotlib
#matplotlib.use('GTKAgg')
#import matplotlib.pyplot as plt
#from subprocess import Popen,PIPE  
import math
import sys
import shutil
import traceback
#import swmmout
import swmm_ea_controller
from PyQt4 import QtCore, QtGui

from swmm5 import swmm5 as sw #old
from swmm5.swmm5tools import SWMM5Simulation #new # leave this both here until fully migrated to new swmm5 interface. 
 


class dumb(object):
    def __init__(self):
        pass  


def evaluatorf(candidates, args):
    fitness = []
    # linestring=args.get('linestring','foo')
    parameters=args.get('parameters',None)
    linestring=parameters.linestring
    #print "thread: ", QtCore.QThread.currentThreadId()
    for cs in candidates:
        sys.stdout.write("|")
        tf=cs[0:parameters.num_inputs] # this is important when evolution strategy is implemented. 
        # above subsetting of the array [0:len(parameters.power_for_inputs)] is important when evolution strategy is used.     
        fitness.append(getFitness(tf,linestring,parameters))
        sys.stdout.write("-")
    return fitness

def getFitness(fillers, linestring,parameters):

    fitness=0.0
    try:
        scaled=scale(fillers,parameters)
        filename = parameters.projectdirectory+os.sep+"tmp"+os.sep+("%07d" % (current_process().pid))+".inp"
	make_sure_path_exists(os.path.dirname(filename))                
        dir = os.path.dirname(filename)
        try:
            os.stat(dir)
        except:
            os.mkdir(dir)    

        #print "\tFlood : ", flood, sum(map(lambda fil: fil,scaled))
        if parameters.swmmouttype[0]==swmm_ea_controller.SWMMREULTSTYPE_FLOOD or parameters.swmmouttype[0]==swmm_ea_controller.SWMMREULTSTYPE_STAGE :
	    costf=pyratemp.Template("@!"+parameters.cost_function+"!@")
	    pp=parse_parameters(scale(fillers,parameters))
	    pp['discount_rate']=parameters.discount_rate
	    pp['stage_size']=parameters.stage_size
	    pp['stages']=parameters.stages
	    cost1=float(costf(**(pp)))
	    costf=pyratemp.Template("@!"+parameters.swmmout_cost_function+"!@")
	    if parameters.swmmouttype[0]==swmm_ea_controller.SWMMREULTSTYPE_STAGE:
		cost2=0.0
		for (i,ls) in enumerate(swmm_ea_controller.extractSWMMmultiplefiles(linestring)):
		    cost = swmmCost(scaled, ls, filename,parameters)
		    pp={"f": cost}
		    t1=float(costf(**(pp)))*(1-(1+parameters.discount_rate)**(-1*parameters.stage_size))/parameters.discount_rate
		    # t1 is the NPV at the start of the ith stage. (http://www.investopedia.com/articles/03/101503.asp#axzz2KWaBIiKB)
		    # now bring it to present value. 
		    cost2+=t1/(1+parameters.discount_rate)**(i*parameters.stage_size)
		    #/(1+parameters.discount_rate)**(parameters.stage_size*i)    
		#cost2*=parameters.stage_size
	    else:
		cost = swmmCost(scaled, linestring, filename,parameters)
		pp={"f": cost}
		cost2=float(costf(**(pp)))
	    if parameters.multiObjective:
		fitness=inspyred.ec.emo.Pareto([cost1,cost2])
	    else:
		fitness=cost1+cost2
	elif parameters.swmmouttype[0]==swmm_ea_controller.SWMMREULTSTYPE_CALIB:
	    fitness=swmmCost(scaled, linestring, filename,parameters)
	else:
	    print "I don't know the calculation type!"
	    raise	
        
    except:
        
        print "\nError here: !!!!\n\n"
        try:
            tb = traceback.format_exc()  
            print tb
        except:
            print "\grave error!!!\n\n"
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
        print "\nProblem scaling with valuerange array. Check it !"

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
	    print "I don't know the calculation type! (", parameters.swmmouttype, ")."
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
        print e, "Error!" #sw.ENgeterror(e,25)

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
    
    def swmm_best_observer(self,population, num_generations, num_evaluations, args):
        parameters=args["parameters"]
	if parameters.multiObjective:
	    print "MOO - Generation: %i " %(num_generations)
	    for i,p in enumerate(sorted(population,key=lambda p: p.fitness[0])):
		self.write_swmm_for_saving("Latest_gen_individual_no-%(#)003i", i, args, p.candidate)
	else:
	    best=max(population)
	    worst=min(population)

	    p=best.candidate[0:parameters.num_inputs] # this is essential when handling evolution strategy in inspyred (due to double internal length of the array)
	    self.write_swmm_for_saving("Best_of_gen_%(#)03i", num_generations, args, p)
	    strb=map(lambda s: "{0:.3e}".format(s),scale(p,parameters))
	    print '\nBest fitness %(fit).3e for values %(ind)s ' % {"fit": best.fitness,"ind": strb}

	while(self.paused):
	    QtCore.QThread.msleep(1000)
	    if not  self.paused_finally:
		self.paused_finally=True
		self.message("..and paused.")

    def write_swmm_for_saving(self, name, number, args, p):
	parameters=args["parameters"]
	linestring=args.get('parameters',None).linestring
	if self.parameters.swmmouttype[0]== swmm_ea_controller.SWMMREULTSTYPE_STAGE:
		for (i,ls) in enumerate(swmm_ea_controller.extractSWMMmultiplefiles(linestring)):
		    swmmfile=(parameters.projectdirectory+os.sep+parameters.resultsdirectory+os.sep+(name+"_stage%(x)03i") % {"#":number,"x": i})+".inp"
		    swmmWrite(scale(p,parameters),ls,swmmfile)
	elif self.parameters.swmmouttype[0]== swmm_ea_controller.SWMMREULTSTYPE_FLOOD:
		swmmfile=(parameters.projectdirectory+os.sep+parameters.resultsdirectory+os.sep+name % {"#":number})+".inp"
		swmmWrite(scale(p,parameters),linestring,swmmfile)
	elif self.parameters.swmmouttype[0]== swmm_ea_controller.SWMMREULTSTYPE_CALIB:
		swmmfile=(parameters.projectdirectory+os.sep+parameters.resultsdirectory+os.sep+name % {"#":number})+".inp"
		swmmWrite(scale(p,parameters),linestring,swmmfile)	    
		# if calibration write a small ini file with calibration data file name in it. 
		shutil.copy2(self.parameters.calINITEMPLATE, swmmfile[:-3]+"ini")
	else: 
		print "I don't know this type of analysis: ", self.parameters.swmmouttype, " !!!"

    def observer_function(self,population, num_generations, num_evaluations, args):
        import time
        if num_generations % 1 != 0 : 
            return 
        parameters=args["parameters"]
        popn=sorted(population)
        #parameters.bestlist[0].append(num_generations)
        result=[num_generations,[]]
        for i in reversed(popn):
            result[1].append(i.fitness)
        self.emit( QtCore.SIGNAL('nextGeneration(PyQt_PyObject)'), result )
        if  parameters.num_cpus < 2:
            # otherwise this thread will starve the gui thread. However, when multiprocessing, python multiprocessing module will take care of this?
            self.msleep(500)
        

    def stop(self):
	with QtCore.QMutexLocker(self.mutex):
	    self.stopped    = True
	    sys.stdout.write( " Signelled to stop. Waiting this cycle to finish....")
	    
    

    def pause(self, theBool=True):
	self.paused_finally=False
	if(theBool):
	    sys.stdout.write( " Signelled to pause. Waiting this cycle to finish....")
	    self.paused=True
	    self.paused_finally=False
	else:
	    sys.stdout.write( " Resuming...")
	    self.paused=False
	    self.paused_finally=True

	#if theBool == True:  # pause task
		#try:
			#self.mutex.lock()
			#self.paused     = True
		#finally:
			#self.mutex.unlock()

	#else: # resume processing
		#try:
			#self.mutex.lock()
			#self.paused     = False
			
		#finally:
			#self.mutex.unlock()
				
    def setParams(self,parameters=None,display=None, prng=None):
        self.parameters=parameters
        self.display=display
        if prng:
	    self.prng=prng
	elif hasattr(parameters,"seed"):
	    print "Using %i as seed" % (parameters.seed)
	    self.prng=Random(parameters.seed)
	else:
	    self.prng=None
	
    def initialize(self):
	# check the simulation type. If it is a one of calibration, 
	pass

    def run(self):
        self.runOptimization()

    def runOptimization(self):
	parameters=self.parameters
	prng=self.prng
	display=self.display
	if parameters is None: 
	    print "problem jim!"
	    return None
	import pyratemp, os
	parameters.linestring=SwmmTemplate(parameters.projectdirectory+os.sep+parameters.datadirectory+os.sep+parameters.templatefile)

	if prng is None:
	    seed = randint(0, sys.maxint)
	    print "Using seed: ", seed, " to initialize random number generator."
	    prng = Random(seed) 
	
	if (parameters.multiObjective):
	    print "Initiating NSGA2 for multi-objective optimization"	    
	    return self.runOptimization_MOO(parameters,prng)
	else:
	    print "Initiating ES for single-objective optimization"	
	    return self.runOptimization_SOO(parameters,prng)

	
    def runOptimization_SOO(self, parameters, prng):


	@inspyred.ec.generators.strategize    
	def generatorf(random, args):
	    bounds=args.get('bounds',[-1,1])
	    size = args.get('num_inputs', 10)
	    return [random.uniform(bounds[0],bounds[1]) for i in range(size)] 
	

	self.log(parameters.projectdirectory+os.sep+'swmm_ea.log')
	self.my_ec = inspyred.ec.EvolutionaryComputation(prng)
	self.my_ec.selector = inspyred.ec.selectors.tournament_selection
	

	self.my_ec.variator = [inspyred.ec.variators.arithmetic_crossover, inspyred.ec.variators.gaussian_mutation]
	self.my_ec.replacer = inspyred.ec.replacers.generational_replacement
	self.my_ec.observer = [self.observer_function,inspyred.ec.observers.file_observer, self.swmm_best_observer]
	self.my_ec.terminator = [inspyred.ec.terminators.evaluation_termination, 
	                    inspyred.ec.terminators.diversity_termination,
	                    self.stopterminator]

	staf=open(parameters.projectdirectory+os.sep+u"stats.csv","w")
	indf=open(parameters.projectdirectory+os.sep+u"indis.csv","w")
	# parallel processing would not work if you pass these to the evolve method (see the log file, serializing these fail!)

	#linestring = SwmmTemplate()



	mp=False
	if parameters.num_cpus > 1: 
	    print "Setting parallel processing because num_cpus =", parameters.num_cpus
	    mp=True

	final_pop = self.my_ec.evolve(generator=generatorf, 
                             parameters=parameters,
                             evaluator=mp 
                             and inspyred.ec.evaluators.parallel_evaluation_mp
                             or evaluatorf,
                             mp_evaluator=evaluatorf, 
                             mp_nprocs=parameters.num_cpus, # inspyred doc is wrong. 
                             pop_size=parameters.pop_size, 
                             statistics_file=staf,
                             individuals_file=indf,

                             bounder=inspyred.ec.Bounder(-1,1),
                             bounds=[-1,1],
                             maximize=parameters.maximize,
                             max_evaluations=parameters.max_evaluations, 
                             crossover_rate=parameters.crossover_rate,
                             num_crossover_points=parameters.num_crossover_points,
                             mutation_rate=parameters.mutation_rate,
                             #individuals_file=indf,
                             #statistics_file=staf,
                             num_inputs=parameters.num_inputs,
                             num_selected=parameters.pop_size,
                             num_elites =parameters.num_elites
                             )

	self.stopped    = True
	return self.my_ec

    def runOptimization_MOO(self, parameters,prng):
	
 
	def generatorf(random, args):
	    bounds=args.get('bounds',[-1,1])
	    size = args.get('num_inputs', 10)
	    return [random.uniform(bounds[0],bounds[1]) for i in range(size)] 
	""" following function is for testing"""
	def generatorK(random, args):
	      return [random.uniform(-5.0, 5.0) for _ in range(self.parameters.num_inputs)]	

	self.log(parameters.projectdirectory+os.sep+'swmm_ea.log')
	self.my_ec=inspyred.ec.emo.NSGA2(prng)
	self.my_ec.variator = [inspyred.ec.variators.blend_crossover, inspyred.ec.variators.gaussian_mutation]
	self.my_ec.observer = [self.observer_function,inspyred.ec.observers.file_observer, self.swmm_best_observer]
	self.my_ec.terminator = [inspyred.ec.terminators.evaluation_termination, 
	                    inspyred.ec.terminators.diversity_termination,
	                    self.stopterminator]

	staf=open(parameters.projectdirectory+os.sep+u"stats.csv","w")
	indf=open(parameters.projectdirectory+os.sep+u"indis.csv","w")
	# parallel processing would not work if you pass these to the evolve method (see the log file, serializing these fail!)
	mp=False
	if parameters.num_cpus > 1: 
	    print "Setting parallel processing because num_cpus =", parameters.num_cpus
	    mp=True
	final_pop = self.my_ec.evolve(generator=generatorf, #K, #generatorf, 
	                         parameters=parameters,
	                         evaluator=mp 
	                         and inspyred.ec.evaluators.parallel_evaluation_mp
	                         or evaluatorf,#K, #evaluatorf,
	                         mp_evaluator=evaluatorf,#K, #evaluatorf, 
	                         mp_nprocs=parameters.num_cpus, # inspyred doc is wrong. 
	                         pop_size=parameters.pop_size, 
	                         statistics_file=staf,
	                         individuals_file=indf,
	                         bounder=inspyred.ec.Bounder(-1,1),
	                         bounds=[-1,1],
	                         maximize=parameters.maximize,
	                         max_evaluations=parameters.max_evaluations, 
	                         num_inputs=parameters.num_inputs, 
	                         crossover_rate=parameters.crossover_rate,
	                         num_crossover_points=parameters.num_crossover_points,
	                         mutation_rate=parameters.mutation_rate,	                         
	                         )
	self.stopped    = True
	return self.my_ec


  

def ReadParameters():
    import os
    parameters=dumb()
    startfile=os.getcwd()+os.sep+'start.yaml'
    try:
        import yaml
        f = open(startfile)
        dataMap = yaml.load(f)
        f.close()

        for key in dataMap :
            setattr(parameters, key, dataMap[key])
    except: 
        print "Problem reading file '%s' " % startfile,sys.exc_info()[0]
        sys.exit()        

    parameterfile=os.getcwd()+os.sep+parameters.projectdirectory+os.sep+"param.yaml"    
    print "Using parameter file :" , parameterfile

    try:
        import yaml
        f = open(parameterfile)
        dataMap = yaml.load(f)
        f.close()

        for key in dataMap :
            setattr(parameters, key, dataMap[key])
    except: 
        print "Problem reading file '%s' " % parameterfile ,sys.exc_info()[0]
        sys.exit()

    parameters.bestlist=[]
    for i in range(parameters.pop_size+1):
        parameters.bestlist.append([])     
    return parameters


def main_function():
    import sys, os
    import multiprocessing
    multiprocessing.freeze_support()
    #if ( len(sys.argv) > 1):
    #    t=sys.argv[1]
    #else:
    print "Working directory : %(n)s" % {"n": os.getcwd()}
        #t=raw_input("Enter the name of parameter  file (*.yaml) : ")
        #t=None

    parameters=ReadParameters()
    
    if not os.path.exists(parameters.projectdirectory):
        print "There is no directory named: "+parameters.projectdirectory+" under "+ prjroot+os.sep + " directory.\n"
        print "Please create one using the template (myproject) provided.\n"
        sys.exit() 
    if not os.path.exists(parameters.projectdirectory+os.sep+parameters.resultsdirectory):
        os.makedirs(parameters.projectdirectory+os.sep+parameters.resultsdirectory)    
    if not os.path.exists(parameters.projectdirectory+os.sep+parameters.datadirectory):
        print "Hell! there's no directory named : " +  parameters.projectdirectory+os.sep+parameters.datadirectory +"\n"
        print "I quit. Check param.yaml file and try again"
        sys.exit()
    import cProfile, pstats


    app = QtGui.QApplication(sys.argv)    
    swmmea=SwmmEA()
    swmmea.setParams(parameters=parameters, display=True)
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
    main_function()


