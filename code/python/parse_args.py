import sys
import datetime

timestamp = datetime.datetime.now().strftime("%m-%d-%YT%H:%M:%S")

if '--segment_length' in sys.argv:
    segment_length = int(sys.argv[sys.argv.index('--segment_length')+1])
else:
    segment_length = 8*60

if '--step' in sys.argv:
    step = int(sys.argv[sys.argv.index('--step')+1])
else:
    step = 60

if '--epochs' in sys.argv:
    epochs = int(sys.argv[sys.argv.index('--epochs')+1])
else:
    epochs = 40

if '--batch_size' in sys.argv:
    batch_size = int(sys.argv[sys.argv.index('--batch_size')+1])
else:
    batch_size = 100

if '--model_path' in sys.argv:
    model_path = sys.argv[sys.argv.index('--model_path')+1]
else:
    model_path = None

if '--history_graph' in sys.argv:
    history_graph = sys.argv[sys.argv.index('--history_graph')+1]
else:
    history_graph = None

if '--dropout' in sys.argv:
    dropout = float(sys.argv[sys.argv.index('--dropout')+1])
else:
    dropout = 0.5

madrs = '-m' in sys.argv or '--madrs' in sys.argv
verbose = '-v' in sys.argv or '--verbose' in sys.argv

identifier = f'Conv1D{madrs and '-MADRS' or ''}_{timestamp}_{segment_length}_{step}_{epochs}_{batch_size}'

if '--logfile' in sys.argv:
    logfile = sys.argv[sys.argv.index('--logfile')+1]
elif '--log' in sys.argv:
    logfile = f'../logs/python/{identifier}'
else:
    logfile = None

if verbose:
    print('Segment length:', segment_length) 
    print('Step:', step)
