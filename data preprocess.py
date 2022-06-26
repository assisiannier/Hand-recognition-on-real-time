import os
import tensorflow as tf
import imageio

def resize_img():
    dirs = os.listdir("split_pic//9")
    for filename in dirs:
        im = tf.io.gfile.GFile("split_pic//9//{}".format(filename), 'rb').read()
        # print("正在处理第%d张照片"%counter)
        with tf.compat.v1.Session() as sess:
            img_data = tf.image.decode_jpeg(im)
            image_float = tf.image.convert_image_dtype(img_data, tf.float32)
            resized = tf.compat.v1.image.resize_images(image_float, [100, 100], method=3)
            resized_im = resized.eval()
            # new_mat = np.asarray(resized_im).reshape(1, 100, 100, 3)
            imageio.imwrite("resized_img9//{}".format(filename),resized_im)




resize_img()