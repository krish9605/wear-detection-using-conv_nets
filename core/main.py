from process_func import *
from model_upconv import Model
import sys
from preprocess import resize_img, imshow
import cv2
from __get__test import get_testdata_from_server


sys.path.append("../utils/")
from logger import Logger
import pickle

# Variables
LOG_DIR = 'log_files/'
SAVE_DIR = 'saved_models/'
RUN_NAME = 'simple_model'
MODEL_NAME = 'latest_model2'
SAVE_PATH = SAVE_DIR + MODEL_NAME
SAVE_HEAT_PATH = './heat_maps2/'
LOAD_MODEL = True
SAVE_FREQ = 156

BATCH_SIZE = 28
NUM_WORKERS = 1
LEARNING_RATE = 1e-3

START_EPOCH = 0
NUM_EPOCHS = 20
VAL_FREQUENCY = 100
TRAIN_PERCENT = 1

NUM_TRAIN_DIR = 1
NUM_TEST_DIR = 1

CROP_SIZE = (128, 128)
EDGE_GAP = (40, 40)
TRANSFORM = False

if TRANSFORM:
    IM_SIZE = CROP_SIZE + EDGE_GAP
else:
    IM_SIZE = CROP_SIZE

TEST_IM_SIZE = CROP_SIZE
THRESHOLD = 0.05  # Threshold to find minimum wear
IS_CUDA = True

NOCUT_SELECT = 0.99  # % of regions selected with no wear


## Delete the tensorboard directory.
def del_log():
    try:
        import shutil
        shutil.rmtree(LOG_DIR, ignore_errors=True)
    except:
        print("couldnt delete log dir")


del_log()

model = Model()

if LOAD_MODEL is True:
    load_path = SAVE_DIR + MODEL_NAME
    model.load_state_dict(torch.load(load_path + '.pt'))


def main():
    if (IS_CUDA):
        model.cuda()

    print('Loading TrainVal Data')
    tensor_logs = Logger(LOG_DIR, name="abc")
    train_loader, val_loader = get_trainval_data(batch_size=BATCH_SIZE, train_percent=TRAIN_PERCENT,
                                                 num_workers=NUM_WORKERS, num_dir=NUM_TRAIN_DIR,
                                                 threshold=THRESHOLD, IM_SIZE=IM_SIZE, target_im_size=TEST_IM_SIZE,
                                                 transform=TRANSFORM, no_cut_select=NOCUT_SELECT, data_dir='../data/')
    val_loader = []
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    print('Started training...')

    for epoch in range(START_EPOCH, START_EPOCH + NUM_EPOCHS):
        train(train_loader, model, criterion, optimizer, epoch, tensor_logs, IS_CUDA=IS_CUDA)

        if epoch % VAL_FREQUENCY == 0 and epoch != 0:
            validate(val_loader, model, epoch, tensor_logs, IS_CUDA=IS_CUDA)

        if (SAVE_FREQ is not None and epoch % SAVE_FREQ == 0 and epoch != 0) or \
                (epoch is (START_EPOCH + NUM_EPOCHS - 1)):
            print ('Saving model')
            torch.save(model.state_dict(), SAVE_PATH + '.pt')

    print('Training done, throwing away training data')
    train_loader, val_loader = [], []

    print('Loading Test Data')
    img_pos_list = get_testdata_from_server(camera=1,test_server_link=False)
    test_dataset, preprocessed_cutouts, img_shapes = get_test_data(data_dir='../test/', num_dir=NUM_TEST_DIR,
                                                                   threshold=THRESHOLD,
                                                                   IM_SIZE=TEST_IM_SIZE)
    pos_list = [x[1] for x in img_pos_list]

    print('Testing model...')
    for i, (test_data, prep_cutout, img_shape, pos_name) in enumerate(zip(test_dataset, preprocessed_cutouts, img_shapes, pos_list)):
        # inside one image
        predictions = test(test_data, model, IS_CUDA=True)

        location_data = [x[1] for x in prep_cutout]
        heat_map = process_test(img_shape, location_data, predictions, TEST_IM_SIZE[0])
        cv2.imwrite(SAVE_HEAT_PATH + str(pos_name) + '.png', heat_map)


if __name__ == '__main__':
    main()
