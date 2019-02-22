import numpy as np

def greyscale(img):
    grey_copy = img.copy()
    for x in grey_copy:
        for y in x:
            rata = np.average(y)
            y[0] = rata
            y[1] = rata
            y[2] = rata
    return grey_copy
def big(img):
    new_img_x = len(img[:, 1]) * 2
    new_img_y = len(img[1, :]) * 2
    new_img = np.zeros((new_img_x, new_img_y, 3))
    for i in range(new_img_x):
        for j in range(new_img_y):
            halfi = int(i / 2)
            halfj = int(j / 2)
            new_img[i, j] = img[halfi, halfj]
    return new_img
#Copas dan kurang paham
def small(img):
    old_img = np.asarray(img)

    old_img_size = old_img.shape
    new_img_x = old_img.shape[0] * 2
    new_img_y = old_img.shape[1] * 2

    new_array = np.full((new_img_x, new_img_y, 3), 255)
    new_array.setflags(write=1)

    for row in range(old_img_size[0]):
        for col in range(old_img_size[1]):
            pix_1, pix_2, pix_3 = old_img[row, col, 0], old_img[row, col, 1], old_img[row, col, 2]
            new_array[row, col, 0], new_array[row + 1, col, 0], new_array[row, col + 1, 0], new_array[
                row + 1, col + 1, 0] = pix_1, pix_1, pix_1, pix_1
            new_array[row, col, 1], new_array[row + 1, col, 1], new_array[row, col + 1, 1], new_array[
                row + 1, col + 1, 1] = pix_2, pix_2, pix_2, pix_2
            new_array[row, col, 2], new_array[row + 1, col, 2], new_array[row, col + 1, 2], new_array[
                row + 1, col + 1, 2] = pix_3, pix_3, pix_3, pix_3

    return new_array
def brightertambah(img):
    img_arr = np.asfarray(img)
    new_arr = img_arr + 10
    new_arr = np.clip(new_arr, 0, 255)
    return new_arr
def darkerkurang(img):
    img_arr = np.asfarray(img)
    new_arr = img_arr - 10
    new_arr = np.clip(new_arr, 0, 255)
    return new_arr
def brighterkali(img):
    img_arr = np.asfarray(img)
    new_arr = img_arr * 1.1
    new_arr = np.clip(new_arr, 0, 255)
    return new_arr
def darkerbagi(img):
    img_arr = np.asfarray(img)
    new_arr = img_arr * 0.8
    new_arr = np.clip(new_arr, 0, 255)
    return new_arr
def geserkanan(img):
    new_arr = np.zeros_like(img)
    for x in range(len(new_arr)):
        for y in range(len(new_arr[x]) - 10):
            new_arr[x][y + 10] = img[x][y]
    return new_arr
def geserkiri(img):
    new_arr = np.zeros_like(img)
    for x in range(len(img)):
        for y in range(len(img[x]) - 10):
            new_arr[x][y] = img[x][y + 10]
    return new_arr
def geseratas(img):
    new_arr = np.zeros_like(img)
    for x in range(len(img) - 10):
        for y in range(len(img[x])):
            new_arr[x][y] = img[x + 10][y]
    return new_arr


def geserbawah(img):
    new_arr = np.zeros_like(img)
    for x in range(len(img) - 10):
        for y in range(len(img[x])):
            new_arr[x + 10][y] = img[x][y]
    return new_arr


def histogram(img):
    from collections import Counter
    hoho = greyscale(img).flatten()
    recounted = Counter(hoho)
    dictlist = np.zeros(256)
    for key, value in recounted.items():
        dictlist[key] = value
    return dictlist


def equalization(img):
    from collections import Counter
    #Jadikan flatten imagenya
    hoho = img.flatten()
    recounted = Counter(hoho)
    dictlist = np.zeros(256)
    for key, value in recounted.items():
        dictlist[key] = value
    cdf = [0] * len(dictlist)
    cdf[0] = dictlist[0]
    for i in range(1, len(dictlist)):
        cdf[i] = cdf[i - 1] + dictlist[i]

    # Normalisasi nilai CDFnya
    my_cdf = [ele * 255 / cdf[-1] for ele in cdf]

    # Interpolasi matrix menggunakan cdf
    image_equalized = np.interp(img, range(0, 256), my_cdf)
    return image_equalized


def blur(img):
    img_arr = np.asfarray(img)

    h, w, _ = img_arr.shape

    temp = np.zeros_like(img_arr)
    ker = np.full((3, 3), 1 / 9)
    print(ker)

    print(img_arr)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            temp[i, j, 0] = img_arr[i - 1, j - 1, 0] * ker[0, 0] + img_arr[i - 1, j, 0] * ker[0, 1] + img_arr[
                i - 1, j + 1, 0] * ker[0, 2] + img_arr[i, j - 1, 0] * ker[1, 0] + \
                            img_arr[i, j, 0] * ker[1, 1] + img_arr[i, j + 1, 0] * ker[1, 2] + img_arr[i + 1, j - 1,
                                                                                                      0] * ker[
                                2, 0] + img_arr[i + 1, j, 0] * ker[2, 1] + img_arr[i + 1, j + 1, 0] * ker[2, 2]
            temp[i, j, 1] = img_arr[i - 1, j - 1, 1] * ker[0, 0] + img_arr[i - 1, j, 1] * ker[0, 1] + img_arr[
                i - 1, j + 1, 1] * ker[0, 2] + img_arr[i, j - 1, 1] * ker[1, 0] + \
                            img_arr[i, j, 1] * ker[1, 1] + img_arr[i, j + 1, 1] * ker[1, 2] + img_arr[i + 1, j - 1,
                                                                                                      1] * ker[
                                2, 0] + img_arr[i + 1, j, 1] * ker[2, 1] + img_arr[i + 1, j + 1, 1] * ker[2, 2]
            temp[i, j, 2] = img_arr[i - 1, j - 1, 2] * ker[0, 0] + img_arr[i - 1, j, 2] * ker[0, 1] + img_arr[
                i - 1, j + 1, 2] * ker[0, 2] + img_arr[i, j - 1, 2] * ker[1, 0] + \
                            img_arr[i, j, 2] * ker[1, 1] + img_arr[i, j + 1, 2] * ker[1, 2] + img_arr[i + 1, j - 1,
                                                                                                      2] * ker[
                                2, 0] + img_arr[i + 1, j, 2] * ker[2, 1] + img_arr[i + 1, j + 1, 2] * ker[2, 2]

    #di set kalau yang besarnya lebih dari 255 dijadikan 255 dan yang kurang dari 0 dijadikan 0
    new_arr = np.clip(temp, 0, 255)
    return new_arr


def edge(img):
    
    img_arr = np.asfarray(img)

    h, w, _ = img_arr.shape

    temp = np.zeros_like(img_arr)
    laplacian = np.array((
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]), dtype="int")
    ker = laplacian
    print(ker)

    print(img_arr)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            temp[i, j, 0] = img_arr[i - 1, j - 1, 0] * ker[0, 0] + img_arr[i - 1, j, 0] * ker[0, 1] + img_arr[
                i - 1, j + 1, 0] * ker[0, 2] + img_arr[i, j - 1, 0] * ker[1, 0] + \
                            img_arr[i, j, 0] * ker[1, 1] + img_arr[i, j + 1, 0] * ker[1, 2] + img_arr[i + 1, j - 1,
                                                                                                      0] * ker[
                                2, 0] + img_arr[i + 1, j, 0] * ker[2, 1] + img_arr[i + 1, j + 1, 0] * ker[2, 2]
            temp[i, j, 1] = img_arr[i - 1, j - 1, 1] * ker[0, 0] + img_arr[i - 1, j, 1] * ker[0, 1] + img_arr[
                i - 1, j + 1, 1] * ker[0, 2] + img_arr[i, j - 1, 1] * ker[1, 0] + \
                            img_arr[i, j, 1] * ker[1, 1] + img_arr[i, j + 1, 1] * ker[1, 2] + img_arr[i + 1, j - 1,
                                                                                                      1] * ker[
                                2, 0] + img_arr[i + 1, j, 1] * ker[2, 1] + img_arr[i + 1, j + 1, 1] * ker[2, 2]
            temp[i, j, 2] = img_arr[i - 1, j - 1, 2] * ker[0, 0] + img_arr[i - 1, j, 2] * ker[0, 1] + img_arr[
                i - 1, j + 1, 2] * ker[0, 2] + img_arr[i, j - 1, 2] * ker[1, 0] + \
                            img_arr[i, j, 2] * ker[1, 1] + img_arr[i, j + 1, 2] * ker[1, 2] + img_arr[i + 1, j - 1,
                                                                                                      2] * ker[
                                2, 0] + img_arr[i + 1, j, 2] * ker[2, 1] + img_arr[i + 1, j + 1, 2] * ker[2, 2]

    new_arr = np.clip(temp, 0, 255)
    return new_arr


def sharpen(img):
    img_arr = np.asfarray(img)

    h, w, _ = img_arr.shape

    temp = np.zeros_like(img_arr)

    sharpen = np.array((
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]), dtype="int")
    ker = sharpen
    print(ker)

    print(img_arr)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            temp[i, j, 0] = img_arr[i - 1, j - 1, 0] * ker[0, 0] + img_arr[i - 1, j, 0] * ker[0, 1] + img_arr[
                i - 1, j + 1, 0] * ker[0, 2] + img_arr[i, j - 1, 0] * ker[1, 0] + \
                            img_arr[i, j, 0] * ker[1, 1] + img_arr[i, j + 1, 0] * ker[1, 2] + img_arr[i + 1, j - 1,
                                                                                                      0] * ker[
                                2, 0] + img_arr[i + 1, j, 0] * ker[2, 1] + img_arr[i + 1, j + 1, 0] * ker[2, 2]
            temp[i, j, 1] = img_arr[i - 1, j - 1, 1] * ker[0, 0] + img_arr[i - 1, j, 1] * ker[0, 1] + img_arr[
                i - 1, j + 1, 1] * ker[0, 2] + img_arr[i, j - 1, 1] * ker[1, 0] + \
                            img_arr[i, j, 1] * ker[1, 1] + img_arr[i, j + 1, 1] * ker[1, 2] + img_arr[i + 1, j - 1,
                                                                                                      1] * ker[
                                2, 0] + img_arr[i + 1, j, 1] * ker[2, 1] + img_arr[i + 1, j + 1, 1] * ker[2, 2]
            temp[i, j, 2] = img_arr[i - 1, j - 1, 2] * ker[0, 0] + img_arr[i - 1, j, 2] * ker[0, 1] + img_arr[
                i - 1, j + 1, 2] * ker[0, 2] + img_arr[i, j - 1, 2] * ker[1, 0] + \
                            img_arr[i, j, 2] * ker[1, 1] + img_arr[i, j + 1, 2] * ker[1, 2] + img_arr[i + 1, j - 1,
                                                                                                      2] * ker[
                                2, 0] + img_arr[i + 1, j, 2] * ker[2, 1] + img_arr[i + 1, j + 1, 2] * ker[2, 2]

    new_arr = np.clip(temp, 0, 255)
    return new_arr


def threshold(img, minval, maxval):
    grey_img = greyscale(img)
    new_img = np.zeros_like(grey_img)
    for x in range(len(img)):
        for y in range(len(img[x])):
            color = grey_img[x][y][0]
            if (color >= minval and color <= maxval):
                new_img[x][y] = [255, 255, 255]
    return new_img


def __backtrackGrowing(img_new, temp, x, y, color, seed):
    try:
        cek = img_new[y,x,0]
        cek2 = temp[y,x,0]
        if (cek - color < seed and cek2 == 0):
            temp[y,x] = [255,255,255]
            #JEJERAN ATAS
            __backtrackGrowing(img_new, temp, x - 1, y - 1, color, seed)
            __backtrackGrowing(img_new, temp, x, y - 1, color, seed)
            __backtrackGrowing(img_new, temp, x + 1, y - 1, color, seed)
            #JEJERAN TENGAH
            __backtrackGrowing(img_new, temp, x - 1, y, color, seed)
            __backtrackGrowing(img_new, temp, x + 1, y, color, seed)
            #JEJERAN BAWAH
            __backtrackGrowing(img_new, temp, x - 1, y + 1, color, seed)
            __backtrackGrowing(img_new, temp, x, y + 1, color, seed)
            __backtrackGrowing(img_new, temp, x + 1, y + 1, color, seed)
    except:
        print("Batas")

def growth(img,x,y,seed):
    img_new = greyscale(img)
    temp = np.zeros_like(img_new)
    __backtrackGrowing(img_new, temp, x, y, img_new[y, x, 0], seed)
    return temp