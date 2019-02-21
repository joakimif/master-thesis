from setup_1d_conv import *

N_FEATURES = 1

if not model_path:
    print('Usage: python3 graph_1d_conv.py [options] --model_path <path_to_models>')
    exit()

results = []

losses = []
accuracies = []

for f in os.listdir(model_path):
    if '.h5' in f and 'Conv1D' in f:
        t, ts, seg, step, epochs, batch = f.split('_')
        
        seg = int(seg)
        step = int(step)
        epochs = int(epochs)
        batch = int(batch.replace('.h5', ''))

        segments, labels, num_sensors, input_shape = create_segments_and_labels(N_FEATURES, seg, step)
        X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2)

        model = load_model(f'{model_path}/{f}')
        loss, acc = model.evaluate(X_test, y_test)
        max_y_test, max_y_pred_test = predict(model, X_test, y_test)

        """ results.append({
            'parameters': (t, ts, seg, step, epochs, batch),
            'evaluation': (loss, acc),
            'predictions': (max_y_test, max_y_pred_test)
        }) """

        losses.append(loss)
        accuracies.append(acc)

        # print('Accuracy: {:5.2f}%'.format(100 * acc))
        # print('Loss: {:5.2f}%'.format(100 * loss))
        # make_confusion_matrix(max_y_test, max_y_pred_test, print_stdout=True)

plt.plot([losses, accuracies])
plt.savefig('../img/plot.png')

