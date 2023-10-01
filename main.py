import argparse
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def generate_kernel(n):
    kernel = np.ones((n, n), np.float32) / (n * n)
    print(f'Kenel:\n{kernel}\nSize: {n}')
    return kernel


def draw_plot(img, title):
    plt.figure(figsize=(
        img.shape[1] / 100, img.shape[0] / 100))
    plt.imshow(img)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])


def main():
    parser = argparse.ArgumentParser(description='Average filter')

    parser.add_argument('filter', choices=['avglib', 'avgown', 'avgboth'], help='Filter to apply')
    parser.add_argument('a', type=int, help='Kernel size')
    parser.add_argument('impath', type=str, help='image path')

    args = parser.parse_args()
    print("args", args)

    if args.filter == 'avglib':
        kernel = generate_kernel(args.a)
        img = cv.imread(args.impath)  # la imagen se lee como BGR
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # con esto la paso a RGB
        draw_plot(img_rgb, 'original')
        result = cv.filter2D(img_rgb, -1, kernel)
        draw_plot(result, 'result')
        plt.grid(True)

        plt.show()


    elif args.filter == 'avgown':
        kernel = generate_kernel(args.a)
    elif args.filter == 'avgboth':
        kernel = generate_kernel(args.a)

    print(f'Resultado de {args.filter}: {kernel}')


if __name__ == '__main__':
    main()
