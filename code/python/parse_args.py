import sys

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

verbose = '-v' in sys.argv or '--verbose' in sys.argv
