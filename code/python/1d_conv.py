from setup_1d_conv import *

if '--segment_length' in sys.argv:
    segment_length = int(sys.argv[sys.argv.index('--segment_length')+1])
else:
    segment_length = 8*60

if '--step' in sys.argv:
    step = int(sys.argv[sys.argv.index('--step')+1])
else:
    step = 60

if verbose:
    print('Segment length:', segment_length) 
    print('Step:', step)

N_FEATURES = 1
BATCH_SIZE = 100
EPOCHS = 40

segments, labels, num_sensors, input_shape = create_segments_and_labels(N_FEATURES, segment_length, step)
X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2)

model = create_model(segment_length, num_sensors, input_shape)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3),
]

history = train(model, X_train, y_train, BATCH_SIZE, EPOCHS, callbacks, validation_split=0.3)

max_y_test, max_y_pred_test = predict(model, X_test, y_test)

make_confusion_matrix(max_y_test, max_y_pred_test, ouput_file='img/confusion_matrix.png', print_stdout=True)