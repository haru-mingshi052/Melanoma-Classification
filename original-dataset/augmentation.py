import numpy as np
import random
import cv2
import os

"""
transformerに使うaugmentation
"""

#====================================
# AdvancedHairAugmentation
#====================================
class AdvancedHairAugmentation:
    def __init__(self, hairs = 5, hairs_folder = '/kaggle/input/melanoma-hairs'):
        self.hairs = hairs
        self.hairs_folder = hairs_folder

    def __call__(self, img):
        n_hairs = random.randint(0, self.hairs)

        if not n_hairs:
            return img

        height, width, _ = img.shape
        hair_images = [im for im in os.listdir(self.hairs_folder) if 'png' in im]

        for _ in range(n_hairs):
            hair = cv2.imread(os.path.join(self.hairs_folder, random.choice(hair_images)))
            hair = cv2.flip(hair, random.choice([-1, 0, 1]))
            hair = cv2.rotate(hair, random.choice([0, 1, 2]))

            h_height, h_width, _ = hair.shape
            roi_ho = random.randint(0, img.shape[0] - hair.shape[0])
            roi_wo = random.randint(0, img.shape[1] - hair.shape[1])
            roi = img[roi_ho : roi_ho + h_height, roi_wo : roi_wo + h_width]

            img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            img_bg = cv2.bitwise_and(roi, roi, mask = mask_inv)

            hair_fg = cv2.bitwise_and(hair, hair, mask = mask)

            dst = cv2.add(img_bg, hair_fg)

            img[roi_ho : roi_ho + h_height, roi_wo : roi_wo + h_width] = dst

        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(hairs = {self.hairs}, hairs_folder = "{self.hairs_folder}")'

#============================
# Microscope
#============================
class Microscope:
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            circle = cv2.circle(
                (np.ones(img.shape) * 255).astype(np.uint8),
                (img.shape[0] // 2, img.shape[1] // 2),
                random.randint(img.shape[0] // 2 - 3, img.shape[0] // 2 + 15),
                (0, 0, 0),
                -1
            )

            mask = circle - 255
            img = np.multiply(img, mask)

        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(p = {self.p})'

#=============================
# DrawHair
#=============================
class DrawHair:
    def __init__(self, hairs = 4, width = (1,2)):
        self.hairs = hairs
        self.width = width

    def __call__(self, img):
        if not self.hairs:
            return img

        width, height, _ = img.shape

        for _ in range(random.randint(0, self.hairs)):
            origin = (random.randint(0, width), random.randint(0, height // 2))
            end = (random.randint(0, width), random.randint(0, height))
            color = (0, 0, 0)
            cv2.line(img, origin, end, color, random.randint(self.width[0], self.width[1]))

        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(hairs = {self.hairs}, width = {self.width})'