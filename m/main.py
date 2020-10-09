import cv2
import numpy as np


def color_transfer():
    src = cv2.imread("scotland_house.jpg")
    cv2.imshow("src", src)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

    target = cv2.imread("scotland_plain.jpg")
    cv2.imshow("target", target)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    h, w, cc = src.shape

    img_s = np.reshape(src / 255, (-1, 3), order='F')
    img_t = np.reshape(target / 255, (-1, 3), order='F')

    a = np.array([[0.3811, 0.5783, 0.0402], [0.1967, 0.7244, 0.0782], [0.0241, 0.1288, 0.8444]])
    b = np.array([[1 / np.sqrt(3), 0, 0], [0, 1 / np.sqrt(6), 0], [0, 0, 1 / np.sqrt(2)]])
    c = np.array([[1, 1, 1], [1, 1, -2], [1, -1, 0]])
    b2 = np.array([[np.sqrt(3) / 3, 0, 0], [0, np.sqrt(6) / 6, 0], [0, 0, np.sqrt(2) / 2]])
    c2 = np.array([[1, 1, 1], [1, 1, -1], [1, -2, 0]])

    img_s[img_s < 1.0 / 255] = 1.0 / 255
    img_t[img_t < 1.0 / 255] = 1.0 / 255

    LMS_s = np.dot(a, np.transpose(img_s))
    LMS_t = np.dot(a, np.transpose(img_t))

    LMS_s = np.log10(LMS_s)
    LMS_t = np.log10(LMS_t)

    lab_s = np.dot(b, np.dot(c, LMS_s))
    lab_t = np.dot(b, np.dot(c, LMS_t))

    mean_s = np.transpose(np.mean(lab_s, axis=-1))  # 1行3列  不同
    std_s = np.transpose(np.std(lab_s, axis=-1))
    mean_t = np.transpose(np.mean(lab_t, axis=-1))  # 1行3列  不同
    std_t = np.transpose(np.std(lab_t, axis=-1))

    res_lab = np.zeros([3, h * w])

    sf = std_t / std_s

    for i in range(cc):
        res_lab[i, :] = (lab_s[i, :] - mean_s[i]) * sf[i] + mean_t[i]

    LMS_res = np.dot(c2, np.dot(b2, res_lab))
    for i in range(cc):
        LMS_res[i, :] = 10 ** LMS_res[i, :]

    mt = np.array([[4.4679, -3.5873, 0.1193], [-1.2186, 2.3809, -0.1624], [0.0497, -0.2439, 1.2045]])
    est_im = np.transpose(np.dot(mt, LMS_res))

    est_im = np.reshape(est_im, (h, w, cc), order='F')
    est_im = est_im.astype(np.float32)
    est_im = cv2.cvtColor(est_im, cv2.COLOR_RGB2BGR)

    cv2.imshow("est", est_im)

    cv2.waitKey(0)


def color_transfer_2(source, target):
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    h, w, ch = source.shape

    img_s = np.reshape(source / 255, (-1, 3), order='F')
    img_t = np.reshape(target / 255, (-1, 3), order='F')

    a = np.array([[0.3811, 0.5783, 0.0402], [0.1967, 0.7244, 0.0782], [0.0241, 0.1288, 0.8444]])
    b = np.array([[1 / np.sqrt(3), 0, 0], [0, 1 / np.sqrt(6), 0], [0, 0, 1 / np.sqrt(2)]])
    c = np.array([[1, 1, 1], [1, 1, -2], [1, -1, 0]])
    b2 = np.array([[np.sqrt(3) / 3, 0, 0], [0, np.sqrt(6) / 6, 0], [0, 0, np.sqrt(2) / 2]])
    c2 = np.array([[1, 1, 1], [1, 1, -1], [1, -2, 0]])

    img_s[img_s < 1.0 / 255] = 1.0 / 255
    img_t[img_t < 1.0 / 255] = 1.0 / 255

    LMS_s = np.dot(a, np.transpose(img_s))
    LMS_t = np.dot(a, np.transpose(img_t))

    LMS_s = np.log10(LMS_s)
    LMS_t = np.log10(LMS_t)

    lab_s = np.dot(b, np.dot(c, LMS_s))
    lab_t = np.dot(b, np.dot(c, LMS_t))

    mean_s = np.transpose(np.mean(lab_s, axis=-1))
    std_s = np.transpose(np.std(lab_s, axis=-1))
    mean_t = np.transpose(np.mean(lab_t, axis=-1))
    std_t = np.transpose(np.std(lab_t, axis=-1))

    res_lab = np.zeros([3, h * w])

    sf = std_t / std_s

    for i in range(ch):
        res_lab[i, :] = (lab_s[i, :] - mean_s[i]) * sf[i] + mean_t[i]

    LMS_res = np.dot(c2, np.dot(b2, res_lab))
    for i in range(ch):
        LMS_res[i, :] = 10 ** LMS_res[i, :]

    mt = np.array([[4.4679, -3.5873, 0.1193], [-1.2186, 2.3809, -0.1624], [0.0497, -0.2439, 1.2045]])
    est_im = np.transpose(np.dot(mt, LMS_res))

    est_im = np.reshape(est_im, (h, w, ch), order='F')
    est_im = est_im.astype(np.float32)
    est_im = cv2.cvtColor(est_im, cv2.COLOR_RGB2BGR)

    return est_im


if __name__ == '__main__':
    # color_transfer()

    src = cv2.imread("scotland_house.jpg")
    target = cv2.imread("scotland_plain.jpg")
    result = color_transfer_2(src, target)
    cv2.imshow("result", result)
    cv2.waitKey(0)
