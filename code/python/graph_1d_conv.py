from setup_1d_conv import *

callbacks = [
    EarlyStopping(monitor='val_loss', patience=2),
]

output_classes = 2
filename_prefix = 'Conv1D'

if madrs:
    create_segments_and_labels = create_segments_and_labels_madrs
    output_classes = 4
    filename_prefix = 'Conv1D-MADRS'

img_path = f'../img/{filename_prefix}_{datetime.datetime.now().strftime("%m-%d-%YT%H:%M:%S")}'
model_path = f'../models/{filename_prefix}_{datetime.datetime.now().strftime("%m-%d-%YT%H:%M:%S")}'

os.mkdir(img_path)
os.mkdir(model_path)

histories = []
loss_list = []
acc_list = []

hours_list = [1, 2, 4, 8, 16, 24]
test_hours = len(hours_list)

for hours in hours_list:
    seg = hours * 60

    segments, labels, num_sensors, input_shape = create_segments_and_labels(1, seg, step)
    X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2)
    model = create_model(seg, num_sensors, input_shape, output_classes=output_classes)
    
    h = train(model, X_train, y_train, batch_size, epochs, callbacks, validation_split=0.4)
    model.save(f'{model_path}/{seg}_{step}_{epochs}_{batch_size}.h5')

    loss, acc = model.evaluate(X_test, y_test)
    max_y_test, max_y_pred_test = predict(model, X_test, y_test)

    make_confusion_matrix(max_y_test, max_y_pred_test, 
                            output_file=f'{img_path}/conf_{seg}_{step}_{epochs}_{batch_size}.png',
                            print_stdout=False, 
                            xticklabels=MADRS_LABLES, 
                            yticklabels=MADRS_LABLES)

    histories.append(pd.DataFrame(h.history, index=h.epoch))
    loss_list.append(loss)
    acc_list.append(acc)

    if len(histories) > test_hours - 1:
        break

historydf = pd.concat(histories, axis=1)

metrics_reported = histories[0].columns
historydf.columns = pd.MultiIndex.from_product([hours_list[:test_hours], metrics_reported], names=['hours', 'metric'])

ax = plt.subplot(211)
historydf.xs('loss', axis=1, level='metric').plot(ax=ax)
plt.title('Loss')

ax = plt.subplot(212)
historydf.xs('acc', axis=1, level='metric').plot(ax=ax)
plt.title('Accuracy')

plt.xlabel('Epochs')

plt.savefig(f'{img_path}/plot.png')

plt.clf()
plt.plot(hours_list[:test_hours], loss_list)
plt.xlabel('Hours')
plt.ylabel('Loss')
plt.savefig(f'{img_path}/plot_loss_eval.png')

plt.clf()
plt.plot(hours_list[:test_hours], acc_list)
plt.xlabel('Hours')
plt.ylabel('Accuracy')
plt.savefig(f'{img_path}/plot_acc_eval.png')