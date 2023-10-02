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
        img.shape[1]*12 / 100, img.shape[0]*12 / 100))
    plt.imshow(img)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])


def read_rgb_img(imgpath):
    img = cv.imread(imgpath)  # la imagen se lee como BGR
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # con esto la paso a RGB
    return img_rgb


def get_channel(channel_index):
    if channel_index == 0:
        return 'RED'
    elif channel_index == 1:
        return 'GREEN'
    elif channel_index == 2:
        return 'GREEN'

def avgown(img, kernel):
    print("hola")


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
        print("shape original: ", img.shape)
        print("img original: ", img[1][0])
        result = cv.filter2D(img, -1, kernel)
        #print("shape result : ", result.shape)
        #print("img result: ", result[6][9])


    elif args.filter == 'avgown':
        print("Implement code for own average filter function")
        n=args.a
        img = read_rgb_img(args.impath)
        height, width, _ = img.shape
        result = np.zeros((height, width, 3), dtype=np.uint8)
        # refactorizar
        print(f"alto={height} - ancho={width}")
        for i in range(height): # en el excel  i es alto = 20
            for j in range(width): # j es ancho = 18 (para el ejemplo de la imagen 20x18)
                print(f'i,j {i},{j}')
                for c in range(3):  # por cada canal
                    print(f'pixel colr in chanel={get_channel(c)}: [{img[i,j,c]}]')
                    print(f'pixel: [{img[i,j]}]')
                    pixel_value = 0
                    for ki in range(n): # para iterar kernel
                        for kj in range(n):
                            y = i - n // 2 + ki # y = alto
                            x = j - n // 2 + kj # x = ancho (quizzá aquí este la diferencia)
                            if 0 <= x < width and 0 <= y < height:
                                print(f'Inside image: {img[y,x, c]}')
                                pixel_value += img[y,x, c] * kernel[ki, kj]
                    result[i, j, c] = int(pixel_value)

        print(result)

    elif args.filter == 'avgboth':
        #  TODO
        print("TODO")

    draw_plot(img, f'{args.filter} original ({args.a}x{args.a})')
    draw_plot(result, f'{args.filter} result ({args.a}x{args.a})')
    plt.show()



if __name__ == '__main__':
    main()
