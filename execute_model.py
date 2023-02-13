import os
import tensorflow as tf
from pathlib import Path
from collections import Counter
import operator
import numpy as np
import glob
import shutil


class ExecuteModel:

    def __init__(self, path):
        self.path = path

    # dectect if file has picture loading by app
    def detect_picture(self):
        initial_count = 0
        for file in os.listdir(self.path):
            if os.path.isfile(os.path.join(self.path, file)):
                initial_count += 1
        return initial_count

    # delete all files in dir
    def remove_file(self,subpath):
        for f in Path(subpath).glob('*.jpg'):
            try:
                f.unlink()
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))

    def remove_dir(self, subpath):
        try:
            dir = subpath
            os.rmdir(dir)

        except OSError as e:
            print("Error: %s" % (e.strerror))

    def countphoto(self):
        count = 0
        for fileNames in os.walk(self.path):
            for i in range(len(fileNames[2])):
                progress1 = fileNames[2][i].split(".")[0]
                progress2 = progress1[0:36]  # 抓前面名字的處理

                final = self.path +'/' + progress2 + "*.jpg"
                result = glob.glob(final)
                count = len(result)

                if count == 16:
                    print("true")
                    userpath = self.path + '/' + progress2
                    os.makedirs(userpath)

                    for img in os.listdir(self.path + '/'):
                        img_path = self.path + '/' + img

                        if os.path.isdir(img_path):
                            continue

                        if progress2 in img_path:
                            shutil.move(img_path, userpath + '/' + img)
                    # count = 0
                    return progress2

    # run model
    def run_model(self, subpath):
        test_model = tf.keras.models.load_model(r'./model')
        test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            subpath,
            labels=None,
            label_mode=None,
            class_names=None,
            color_mode="rgb",
            batch_size=32,
            image_size=(224, 224),
            shuffle=False,
            seed=0,
            interpolation="bilinear",
            follow_links=False,
        )

        # piexl 0~255 -> 0~1
        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
        test_dataset = test_dataset.map(lambda x: (normalization_layer(x)))

        # predict class
        y_predict = test_model.predict(test_dataset)
        return y_predict

    # calculate return confidence level vector
    def get_c_v(self ,p_s='', class_num= 11):
        c_v = []  # confidence level vector
        for label in range(class_num):
            y_predict_lable = np.argmax(p_s, 1)
            y_index = np.argwhere(y_predict_lable == label).reshape(-1)
            p_temp = p_s[y_index]
            p_temp = np.array(p_temp)
            if len(np.shape(p_temp)) <= 1 or np.size(p_temp) <= 1:
                c = 1.0
            else:
                IQR = np.percentile(p_temp[:, label], 75) - np.percentile(p_temp[:, label], 25)
                c = np.percentile(p_temp[:, label], 25) - 1.5 * IQR
            c_v.append(c)
        return c_v

    # put final classify
    def p_to_label(self ,p_s='', c_v=''):
        y_predict_lable = np.argmax(p_s, 1)
        y_out = np.zeros(y_predict_lable.shape)
        for label in range(p_s.shape[1]):
            y_index = np.argwhere(y_predict_lable == label).reshape(-1)
            p_temp = p_s[y_index][:, label]
            p_temp = np.where(p_temp < c_v[label], -1, label)
            y_out[y_index] = p_temp
        return y_out.astype(int)

    def most_high_result(self, y_out):
        value = Counter(y_out).most_common()
        cat_dict = dict(value)
        max_key = max(cat_dict.items(), key=operator.itemgetter(1))[0]
        return max_key

    def ExecuteIdentify(self,uuid):

        result_str = ""
        try:
            subpath = self.path +'/'+ uuid
            y_predict = self.run_model(subpath)
            print("model running...")
            c = self.get_c_v(y_predict, 11)
            y_out = self.p_to_label(y_predict, c)
            print(y_out)
            result = self.most_high_result(y_out)
            # print(result)

            if result!= None:
                result_str = str(result)

            # self.remove_file()  # delete images in the cameraDatas dir
            #
            # if os.path.isdir(subpath):
            #     self.remove_file(subpath)
            #     self.remove_dir(subpath)

            id_and_result = uuid + result_str
            print(id_and_result)

            return id_and_result

        except Exception as e:
            print("Exception on running model: ")
            print(e)







