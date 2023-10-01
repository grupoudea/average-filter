import argparse
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def generate_kernel(n):
    kernel = np.ones((n, n), np.float32) / (n * n)
    # print(f'Kenel:\n{kernel}\nSize: {n}')
    return kernel


def draw_plot(img, title):
    plt.figure(figsize=(
        img.shape[1] / 100, img.shape[0] / 100))
    plt.imshow(img)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])


def read_rgb_img(imgpath):
    img = cv.imread(imgpath)  # la imagen se lee como BGR
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # con esto la paso a RGB
    return img_rgb


def main():
    parser = argparse.ArgumentParser(description='Average filter')

    parser.add_argument('filter', choices=['avglib', 'avgown', 'avgboth'], help='Filter to apply')
    parser.add_argument('a', type=int, help='Kernel size')
    parser.add_argument('impath', type=str, help='image path')

    args = parser.parse_args()
    kernel = generate_kernel(args.a)
    print(f'=================\n'
          f'Filter: {args.filter}\n'
          f'Kernel:\n'
          f'({args.a}x{args.a})\n'
          f'=================')


    if args.filter == 'avglib':
        img = read_rgb_img(args.impath)
        result = cv.filter2D(img, -1, kernel)

    elif args.filter == 'avgown':
        print("Implement code for own average filter function")
    elif args.filter == 'avgboth':
        #  TODO
        print("TODO")

    draw_plot(img, 'original')
    draw_plot(result, 'result')
    plt.show()


if __name__ == '__main__':
    main()
