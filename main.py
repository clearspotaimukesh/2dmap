import sys
import cv2
import numpy as np
import imutils
from imutils import paths
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import deque


class Orthomosaic:
    def __init__(self, debug):
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        self.no_raw_images = []
        self.temp_image = []
        self.final_image = []
        self.debug = debug

    def load_image(self, imagePath):
        image_temp = cv2.imread(imagePath)
        scale_percent = 20  # percent of original size, reduced to 20% to avoid large images
        width = int(image_temp.shape[1] * scale_percent / 100)
        height = int(image_temp.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        image = cv2.resize(image_temp, dim)
        return image

    def load_dataset(self):
        self.ap = argparse.ArgumentParser()
        self.ap.add_argument("-i", "--images", type=str, required=True,
                             help="path to input directory of images to stitch")
        self.ap.add_argument("-o", "--output", type=str, required=True,
                             help="path to the output image")
        self.args = vars(self.ap.parse_args())

        # grab the paths to the input images and initialize our images list
        if self.debug:
            print("[INFO] Importing Images...")
        self.imagePaths = sorted(list(paths.list_images(self.args["images"])))

        # Parallelize the image loading process
        with ThreadPoolExecutor() as executor:
            self.images = list(executor.map(self.load_image, self.imagePaths))

        if self.debug:
            print("[INFO] Importing Complete")

    def mixer(self):
        self.no_raw_images = len(self.images)
        if self.debug:
            print(f"[INFO] {self.no_raw_images} Images have been loaded")

        # Use deque for efficient pop and append operations
        image_queue = deque(self.images)
        self.temp_image = image_queue.popleft()

        while image_queue:
            self.temp_image = self.sticher(self.temp_image, image_queue.popleft())

        self.final_image = self.temp_image
        
        if self.debug:
            print(f"[INFO] Saving final image to {self.args['output']}")

        output_path = self.args["output"]
        if not output_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
            print("[WARNING] Output path does not have a valid extension. Defaulting to 'output.png'.")
            output_path = "output.png"

        cv2.imshow("output", self.final_image)
        cv2.imwrite(output_path, self.final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def sticher(self, image1, image2):
        orb = cv2.ORB_create(nfeatures=5000)  # Increased the number of features
        if self.debug:
            print(image1.shape)

        # Find the key points and descriptors with ORB
        keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

        # Use FLANN-based matcher for faster and more accurate matching
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        MIN_MATCH_COUNT = 10  # Set minimum match condition

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            result = self.wrap_images(image2, image1, M)
            return result
        else:
            print("Error: Not enough matches found")
            return image1  # Return the original image if not enough matches

    def wrap_images(self, image1, image2, H):
        rows1, cols1 = image1.shape[:2]
        rows2, cols2 = image2.shape[:2]
        list_of_points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
        temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)
        list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
        list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)
        [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

        translation_dist = [-x_min, -y_min]

        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
        output_img = cv2.warpPerspective(image2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
        output_img[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0]] = image1
        return output_img


if __name__ == "__main__":
    tester = Orthomosaic(debug=True)
    tester.load_dataset()
    tester.mixer()
else:
    pass
