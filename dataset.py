import os
import tensorflow as tf
import random
import numpy as np
from coviar import get_num_frames
from coviar import load
from PIL import Image

GOP_SIZE = 12
DATA_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 1, 3))
DATA_STD  = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 1, 3))
CROP_SIZE = 224
CHANNEL_SIZE = 3

def getPathAndLabel(data_list, data_path):
    path_list = []
    with open(data_list, 'r') as f:
        lines = f.read().splitlines()
        short_path_list, _, label_list_string = zip(*[i.split(' ') for i in lines])

    for video_path in short_path_list :
        path = os.path.join(data_path, video_path[:-4] + '.mp4')
        path_list.append( path )

    label_list =  list(map(int, label_list_string))

    return path_list, label_list

def getGopPos(frame_idx, representation_idx):
    gop_index = frame_idx // GOP_SIZE
    gop_pos = frame_idx % GOP_SIZE
    if representation_idx in [1, 2]:
        if gop_pos == 0:
            gop_index -= 1
            gop_pos = GOP_SIZE - 1
    else:
        gop_pos = 0
    return gop_index, gop_pos

def getSegRange(nFrames, num_segments, seg, representation_idx):
    if representation_idx in [1, 2]:
        nFrames -= 1

    seg_size = float(nFrames - 1) / num_segments
    seg_begin = int(np.round(seg_size * seg))
    seg_end = int(np.round(seg_size * (seg+1)))
    if seg_end == seg_begin:
        seg_end = seg_begin + 1

    if representation_idx in [1, 2]:
        return seg_begin + 1, seg_end + 1

    return seg_begin, seg_end

def getTestFrameIndex(nFrames, seg_idx, nSegments, representation_idx):
        if representation_idx in [1, 2]:
            nFrames -= 1

        seg_size = float(nFrames - 1) / nSegments
        v_frame_idx = int(np.round(seg_size * (seg_idx + 0.5)))

        if representation_idx in [1, 2]:
            v_frame_idx += 1

        return getGopPos(v_frame_idx, representation_idx)


def getTrainFrameIndex(nFrames, seg_idx, nSegments, representation_idx):
    seg_begin, seg_end = getSegRange(nFrames, nSegments, seg_idx, representation_idx)

    v_frame_idx = random.randint(seg_begin, seg_end - 1)
    return getGopPos(v_frame_idx, representation_idx)

def _flip_left_right(filename, label):
    print(filename)

    filename = tf.image.random_flip_left_right(filename, 0)

    return filename, label

def _parse_function(filename, label, nSegments):

    reps_np = []

    for representation_idx in range (0, 3):

        frames = []
        for seg_idx in range(0, nSegments):
            nFrames = get_num_frames(filename.decode())
            gop_index, gop_pos = getTestFrameIndex (nFrames, seg_idx, nSegments, representation_idx)
            img = load(filename.decode(), gop_index, gop_pos, representation_idx, True)

            if img is None:
                print('Error: loading video %s failed.' % filename.decode())
                img =  np.zeros((256, 256, 3))
            else:
                if representation_idx == 1:
                    img = (img * (127.5 / 20)).astype(np.int32)
                    img += 128
                    img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)
                    img = np.append(img, np.zeros_like(img[...,0,None]), axis=-1)
                elif representation_idx == 2:
                    img += 128
                    img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)
                else:
                    img = img[..., ::-1] #flipping to RGB

            frames.append(img)

        #np_frames = np.transpose(np.array(frames).astype(np.float32), (0, 3, 1, 2)) / 255.0
        np_frames = np.array(frames).astype(np.float32) / 255.0

        if representation_idx == 0:
            np_frames = (np_frames - DATA_MEAN) / DATA_STD
        elif representation_idx == 2:
            np_frames = (np_frames - 0.5) / DATA_STD
        elif representation_idx == 1:
            np_frames = (np_frames - 0.5)
        np_frames = np_frames [:,16:240, 52:276, :].astype(np.float32)
        reps_np.append(np_frames)

    return reps_np[0], reps_np[1], reps_np[2], label

def _parse_function_v2(filename, label, nSegments):

    reps_np = []

    for representation_idx in range (0, 3):

        frames = []
        for seg_idx in range(0, nSegments):
            nFrames = get_num_frames(filename.decode())
            gop_index, gop_pos = getTrainFrameIndex (nFrames, seg_idx, nSegments, representation_idx)
            img = load(filename.decode(), gop_index, gop_pos, representation_idx, True)

            if img is None:
                print('Error: loading video %s failed.' % filename.decode())
                img =  np.zeros((256, 256, 3))
            else:
                if representation_idx == 1:
                    img = (img * (127.5 / 20)).astype(np.int32)
                    img += 128
                    img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)
                    img = np.append(img, np.zeros_like(img[...,0,None]), axis=-1)
                elif representation_idx == 2:
                    img += 128
                    img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)
                else:
                    img = img[..., ::-1] #flipping to RGB

            frames.append(img)

        #np_frames = np.transpose(np.array(frames).astype(np.float32), (0, 3, 1, 2)) / 255.0
        np_frames = np.array(frames).astype(np.float32) / 255.0

        if representation_idx == 0:
            np_frames = (np_frames - DATA_MEAN) / DATA_STD
        elif representation_idx == 2:
            np_frames = (np_frames - 0.5) / DATA_STD
        elif representation_idx == 1:
            np_frames = (np_frames - 0.5)
        np_frames = np_frames [:,16:240, 52:276, :].astype(np.float32)
        reps_np.append(np_frames)

    return reps_np[0], reps_np[1], reps_np[2], label

def buildTestDataset(test_list, data_path, nSegments, batch_size = 16, num_threads=2, buffer=30):

    test_path_list, test_label_list = getPathAndLabel(test_list, data_path)

    dataset = tf.data.Dataset.from_tensor_slices((test_path_list, test_label_list))

    dataset = dataset.map(lambda filename, label:
                          tuple( tf.py_func( _parse_function, [filename, label, nSegments],
                          [tf.float32, tf.float32, tf.float32, tf.int32]))).prefetch(buffer)

    #if augment:
    #    dataset = dataset.map(_flip_left_right).prefetch(buffer)
    #    dataset = dataset.map(lambda filename, label: _random_crop(filename, label, nSegments)).prefetch(buffer)

    #dataset = dataset.map(_transpose).prefetch(buffer)
    
    dataset = dataset.batch(batch_size)

    return dataset


def buildTrainDataset_v2(train_list, data_path, nSegments, batch_size = 16, augment = True, shuffle = True, num_threads=2, buffer=30):

    train_path_list, train_label_list = getPathAndLabel(train_list, data_path)

    dataset = tf.data.Dataset.from_tensor_slices((train_path_list, train_label_list)).repeat()

    if shuffle:
        dataset = dataset.shuffle(10000)

    dataset = dataset.map(lambda filename, label:
                         tuple( tf.py_func( _parse_function_v2, [filename, label, nSegments],
                          [tf.float32, tf.float32, tf.float32, tf.int32]))).prefetch(buffer)

    #if augment:
    #    dataset = dataset.map(_flip_left_right).prefetch(buffer)
    #    dataset = dataset.map(lambda filename, label: _random_crop(filename, label, nSegments)).prefetch(buffer)

    #dataset = dataset.map(_transpose).prefetch(buffer)
    
    dataset = dataset.batch(batch_size)

    return dataset