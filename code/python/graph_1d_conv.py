from setup_1d_conv import *

N_FEATURES = 1

if not model_path:
    print('Usage: python3 graph_1d_conv.py [options] --model_path <path_to_models>')
    exit()

models = []
datasets = []

for f in os.listdir(model_path):
    if '.h5' in f and 'Conv1D' in f:
        models.append(load_model(f))

        t, ts, seg, step, epochs, batch = f.split('_')

        print(t,ts,seg,step,epochs,batch)
        break

        segments, labels, num_sensors, input_shape = create_segments_and_labels(N_FEATURES, segment_length, step)

        datasets.append(train_test_split(segments, labels, test_size=0.2))

if verbose:
    print(f'Loading models from {model_path}...')





for model in models:
    loss, acc = model.evaluate(X_test, y_test)

    if verbose:
        print('Accuracy: {:5.2f}%'.format(100 * acc))
        print('Loss: {:5.2f}%'.format(100 * loss))

    max_y_test, max_y_pred_test = predict(model, X_test, y_test)

    if not model_path:
        timestamp = datetime.datetime.now().strftime("%m-%d-%YT%H:%M:%S")
        save_label = f'Conv1D_{timestamp}_{segment_length}_{step}_{epochs}_{batch_size}'

        model.save(f'../models/{save_label}.h5')

        make_confusion_matrix(max_y_test, max_y_pred_test, output_file=f'../img/confusion_matrix/{save_label}.png', print_stdout=True)
    else:
        make_confusion_matrix(max_y_test, max_y_pred_test, print_stdout=True)