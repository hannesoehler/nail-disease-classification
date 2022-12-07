import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image
import glob
from tqdm import tqdm

class Image_preprocessing:
    
    def extract_rows(self, image, one_image=False):
    
        rows = []
        for h in range(image.shape[0]):
            if np.sum(image[h-1,:,0]) + np.sum(image[h-1,:,1]) + np.sum(image[h-1,:,2]) == image.shape[1] * 3 \
            and np.sum(image[h,:,0]) + np.sum(image[h,:,1]) + np.sum(image[h,:,2]) != image.shape[1] * 3:
                row = image[h:,:,:]
                h1 = h
            elif np.sum(image[h-1,:,0]) + np.sum(image[h-1,:,1]) + np.sum(image[h-1,:,2]) != image.shape[1] * 3 \
            and np.sum(image[h,:,0]) + np.sum(image[h,:,1]) + np.sum(image[h,:,2]) == image.shape[1] * 3:
                row = row[:h-h1,:,:]
                rows.append(row)
        if one_image == True:
            if len(rows) > 0:
                return rows[0]
            else:
                return image
        else:
            return rows
    
    def extract_single_nail_images(self, row):
        nail_images = []
        for w in range(row.shape[1]):
            if np.sum(row[:,w-1,0]) + np.sum(row[:,w-1,1]) + np.sum(row[:,w-1,2]) == row.shape[0] * 3 \
            and np.sum(row[:,w,0]) + np.sum(row[:,w,1]) + np.sum(row[:,w,2]) != row.shape[0] * 3:
                nail_image = row[:,w:,:]
                w1 = w
            elif np.sum(row[:,w-1,0]) + np.sum(row[:,w-1,1]) + np.sum(row[:,w-1,2]) != row.shape[0] * 3 and \
            np.sum(row[:,w,0]) + np.sum(row[:,w,1]) + np.sum(row[:,w,2]) == row.shape[0] * 3:
                nail_image = nail_image[:,:w-w1,:]
                nail_image = self.extract_rows(nail_image, one_image=True)
                nail_images.append(nail_image)
        return nail_images        
    
    def __call__(self, project_path, input_path, output_path):
        
        if not os.path.exists(f'{project_path}/{output_path}'):
            os.mkdir(output_path)
            
        for path in tqdm(glob.glob(f'{input_path}/*')):
    
            image = plt.imread(path)
            image_name = path.split('\\')[-1].split('.')[0]
            
            if not os.path.exists(f"{project_path}/{output_path}/{image_name}"):
                os.mkdir(f"{output_path}/{image_name}")
            
                rows = self.extract_rows(image)
                
                for i, row in enumerate(rows):
                    nail_images = self.extract_single_nail_images(row)
                    for j, nail_image in enumerate(nail_images):
                        nail_image = Image.fromarray((nail_image*255).astype(np.uint8))
                        nail_image.save(f'{project_path}/{output_path}/{image_name}/{image_name}-{i:03}-{j:03}.png')