#使用模型部分，用于测试图片等等
import os
import tensorflow as tf
import cv2
import numpy as np
import csv
import tqdm

data_folder_name = '..\\Facical-Expression-Recognition'
data_path_name = 'data'
pic_path_name = 'pic'
cv_path_name = 'fer2013'
csv_file_name = 'fer2013.csv'
ckpt_name = 'cnn_emotion_classifier_g.ckpt'
model_path_name = 'cnn_inuse'
casc_name = 'haarcascade_frontalface_alt.xml'
cv_path = os.path.join(data_folder_name, data_path_name, cv_path_name)
csv_path = os.path.join(cv_path, csv_file_name)
ckpt_path = os.path.join(data_folder_name, data_path_name, model_path_name, ckpt_name)
casc_path = os.path.join(data_folder_name, data_path_name, casc_name)
pic_path = os.path.join(data_folder_name, data_path_name, pic_path_name)

channel = 1  #单通道
default_height = 48  #高为48
default_width = 48   #宽为48
confusion_matrix = False
use_advanced_method = True
emotion_labels = ['angry', 'disgust:', 'fear', 'happy', 'sad', 'surprise', 'neutral']
num_class = len(emotion_labels)


# config=tf.ConfigProto(log_device_placement=True)
sess = tf.Session()
#载入已经训练好的模型
saver = tf.train.import_meta_graph(ckpt_path+'.meta')
saver.restore(sess, ckpt_path)
graph = tf.get_default_graph()#获取计算图
name = [n.name for n in graph.as_graph_def().node]
print(name)
#使用graph.get_tensor_by_name()方法来获得已经保存的操作（operations）和placeholder variables。
x_input = graph.get_tensor_by_name('x_input:0')
dropout = graph.get_tensor_by_name('dropout:0')
logits = graph.get_tensor_by_name('project/output/logits:0')


def advance_image(images_):
    rsz_img = []
    rsz_imgs = []
    for image_ in images_:
        rsz_img.append(image_)
    rsz_imgs.append(np.array(rsz_img))
    rsz_img = []
    for image_ in images_:
        rsz_img.append(np.reshape(cv2.resize(image_[2:45, :], (default_height, default_width)),
                                  [default_height, default_width, channel]))
    rsz_imgs.append(np.array(rsz_img))
    rsz_img = []
    for image_ in images_:
        rsz_img.append(np.reshape(cv2.resize(cv2.flip(image_, 1), (default_height, default_width)),
                                  [default_height, default_width, channel]))
    rsz_imgs.append(np.array(rsz_img))
    return rsz_imgs


def produce_result(images_):
    images_ = np.multiply(np.array(images_), 1. / 255)
    if use_advanced_method:
        rsz_imgs = advance_image(images_)
    else:
        rsz_imgs = [images_]
    pred_logits_ = []
    for rsz_img in rsz_imgs:
        pred_logits_.append(sess.run(tf.nn.softmax(logits), {x_input: rsz_img, dropout: 1.0}))
    return np.sum(pred_logits_, axis=0)


def produce_results(images_):
    results = []
    pred_logits_ = produce_result(images_)
    pred_logits_list_ = np.array(np.reshape(np.argmax(pred_logits_, axis=1), [-1])).tolist()
    for num in range(num_class):
        results.append(pred_logits_list_.count(num))
    result_decimals = np.around(np.array(results) / len(images_), decimals=3)#返回四舍五入后的值，可指定精度
    return results, result_decimals


def produce_confusion_matrix(images_list_, total_num_):
    total = []
    total_decimals = []
    for ii, images_ in enumerate(images_list_):
        results, result_decimals = produce_results(images_)
        total.append(results)
        total_decimals.append(result_decimals)
        print(results, ii, ":", result_decimals[ii])
        print(result_decimals)
    sum = 0
    for i_ in range(num_class):
        sum += total[i_][i_]
    print('acc: {:.3f} %'.format(sum * 100. / total_num_))
    print('Using ', ckpt_name)


def predict_emotion(image_):
    image_ = cv2.resize(image_, (default_height, default_width))
    image_ = np.reshape(image_, [-1, default_height, default_width, channel])
    return produce_result(image_)[0]


def face_detect(image_path, casc_path_=casc_path):
    if os.path.isfile(casc_path_):
        face_casccade_ = cv2.CascadeClassifier(casc_path_)#检测人脸
        img_ = cv2.imread(image_path)
        img_gray_ = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)   #使用函数cv2.cvtColor(input_image ，flag)，flag是转换类型
        #BGR和灰度图的转换使用 cv2.COLOR_BGR2GRAY ,BGR和HSV的转换使用 cv2.COLOR_BGR2HSV
        # face detection
        #detectMultiScale函数。它可以检测出图片中所有的人脸，并将人脸用vector保存各个人脸的坐标、大小（用矩形表示）
        faces = face_casccade_.detectMultiScale(
            img_gray_,
            scaleFactor=1.1,
            minNeighbors=1,
            minSize=(30, 30),
        )
        return faces, img_gray_, img_
    else:
        print("There is no {} in {}".format(casc_name, casc_path_))

#对图片进行读取识别
if __name__ == '__main__':
    if not confusion_matrix:#如果不进行训练而是测试模型
        images_path = []
        files = os.listdir(pic_path)
        for file in files:
            if file.lower().endswith('jpg') or file.endswith('png'):
                images_path.append(os.path.join(pic_path, file))
        for image in images_path:
            faces, img_gray, img = face_detect(image)
            spb = img.shape
            sp = img_gray.shape
            height = sp[0]
            width = sp[1]
            size = 600
            emotion_pre_dict = {}
            face_exists = 0
            for (x, y, w, h) in faces:
                face_exists = 1
                face_img_gray = img_gray[y:y + h, x:x + w]
                results_sum = predict_emotion(face_img_gray)  # face_img_gray
                for i, emotion_pre in enumerate(results_sum):
                    emotion_pre_dict[emotion_labels[i]] = emotion_pre
                # 输出所有情绪的概率
                print(emotion_pre_dict)
                label = np.argmax(results_sum)
                emo = emotion_labels[int(label)]
                print('Emotion : ', emo)
                # 输出最大概率的情绪
                # 使框的大小适应各种像素的照片
                t_size = 2
                ww = int(spb[0] * t_size / 300)
                www = int((w + 10) * t_size / 100)
                www_s = int((w + 20) * t_size / 100) * 2 / 5
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), ww)
                cv2.putText(img, emo, (x + 2, y + h - 2), cv2.FONT_HERSHEY_SIMPLEX,
                            www_s, (255, 0, 255), thickness=www, lineType=1)
                # img_gray full face     face_img_gray part of face
            if face_exists:
                cv2.namedWindow('Emotion_classifier', 0)
                cent = int((height * 1.0 / width) * size)
                cv2.resizeWindow('Emotion_classifier', size, cent)
                cv2.imshow('Emotion_classifier', img)
                k = cv2.waitKey(0)#waitKey()函数的功能是不断刷新图像，频率时间为delay，单位为ms。
                cv2.destroyAllWindows()
                 if k & 0xFF == ord('q'):
                     break
    if confusion_matrix:
        with open(csv_path, 'r') as f:
            csvr = csv.reader(f)
            header = next(csvr)
            rows = [row for row in csvr]
            val = [row[:-1] for row in rows if row[-1] == 'PublicTest']
            tst = [row[:-1] for row in rows if row[-1] == 'PrivateTest']
        confusion_images_total = []
        confusion_images = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
        test_set = tst
        total_num = len(test_set)
        for label_image_ in test_set:
            label_ = int(label_image_[0])
            image_ = np.reshape(np.asarray([int(p) for p in label_image_[-1].split()]), [default_height, default_width, 1])
            confusion_images[label_].append(image_)
        produce_confusion_matrix(confusion_images.values(), total_num)



#对视频文件进行读取并识别
# if __name__ == '__main__':
#     if not confusion_matrix:  # 如果不进行训练而是测试模型
#         window_name = 'Emotion_classifier'
#         cv2.namedWindow(window_name)
#         vp = './v1.mp4'
#         cap = cv2.VideoCapture(vp)
#
#         classifier = cv2.CascadeClassifier(
#             'D:/Anaconda3/envs/tensorflow/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
#
#         color = (0, 255, 0)
#         while cap.isOpened():
#             ok, frame = cap.read()  # 读取数据
#             if not ok:
#                 break
#
#             img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#             faceRects = classifier.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
#             spb = frame.shape
#             sp = img_gray.shape
#             height = sp[0]
#             width = sp[1]
#             size = 600
#             cent = int((height * 1.0 / width) * size)
#             emotion_pre_dict = {}
#             if len(faceRects) > 0:
#                 for faceRect in faceRects:
#                     x, y, w, h = faceRect
#                     face_img_gray = img_gray[y:y + h, x:x + w]
#                     results_sum = predict_emotion(face_img_gray)  # face_img_gray
#                     for i, emotion_pre in enumerate(results_sum):
#                         emotion_pre_dict[emotion_labels[i]] = emotion_pre
#                         # 输出所有情绪的概率
#                     print(emotion_pre_dict)
#                     label = np.argmax(results_sum)
#                     emo = emotion_labels[int(label)]
#                     print('Emotion : ', emo)
#                     # 输出最大概率的情绪
#                     # 使框的大小适应各种像素的照片
#                     t_size = 2
#                     ww = int(spb[0] * t_size / 300)
#                     www = int((w + 10) * t_size / 100)
#                     www_s = int((w + 20) * t_size / 100) * 2 / 5
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), ww)
#                     cv2.putText(frame, emo, (x + 2, y + h - 2), cv2.FONT_HERSHEY_SIMPLEX,
#                                 www_s, (255, 0, 255), thickness=www, lineType=1)
#
#             # cv2.resizeWindow('Emotion_classifier', size, cent)
#             cv2.imshow('Emotion_classifier', frame)
#             k = cv2.waitKey(10)  # waitKey()函数的功能是不断刷新图像，频率时间为delay，单位为ms。
#             if k & 0xFF == ord('q'):
#                 break
#         cv2.destroyAllWindows()
#
#     if confusion_matrix:
#         with open(csv_path, 'r') as f:
#             csvr = csv.reader(f)
#             header = next(csvr)
#             rows = [row for row in csvr]
#             val = [row[:-1] for row in rows if row[-1] == 'PublicTest']
#             tst = [row[:-1] for row in rows if row[-1] == 'PrivateTest']
#         confusion_images_total = []
#         confusion_images = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
#         test_set = tst
#         total_num = len(test_set)
#         for label_image_ in test_set:
#             label_ = int(label_image_[0])
#             image_ = np.reshape(np.asarray([int(p) for p in label_image_[-1].split()]),
#                                 [default_height, default_width, 1])
#             confusion_images[label_].append(image_)
#         produce_confusion_matrix(confusion_images.values(), total_num)