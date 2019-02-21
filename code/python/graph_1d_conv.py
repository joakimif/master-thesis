from setup_1d_conv import *

N_FEATURES = 1

callbacks = [
    EarlyStopping(monitor='val_loss', patience=2),
]

if not model_path:
    print('Usage: python3 graph_1d_conv.py [options] --model_path <path_to_models>')
    exit()

histories = []

for f in reversed(os.listdir(model_path)):
    if '.h5' in f and 'Conv1D' in f:
        t, ts, seg, step, epochs, batch = f.split('_')
        
        seg = int(seg)
        step = int(step)
        epochs = int(epochs)
        batch = int(batch.replace('.h5', ''))

        segments, labels, num_sensors, input_shape = create_segments_and_labels(N_FEATURES, seg, step)
        X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2)

        model = create_model(seg, num_sensors, input_shape)
        history = train(model, X_train, y_train, batch_size, epochs, callbacks, validation_split=0.4)

        df = pd.DataFrame(h.history, index=h.epoch)
        histories.append(df)

        df.plot()

        break

plt.savefig('plot.png')

