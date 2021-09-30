from header_imports import *

class edge_detection_analysis(object):
    def __init__(self, input_auguments):
        
        # Path to ad it to
        self.path = input_auguments[1]

        # Import paths
        self.input_auguments = input_auguments[2]
        self.image_path = self.path + "/" + self.input_auguments + "/"

        # Personal Kernel Creation 
        self.kernel_2 = np.array([[1, -3, -1],
                                  [3, 0, 3],
                                  [1, -3, -1]]) 
        

        # Original edge detection kernel
        self.edge_detector = np.array([[-1, -1, -1],
                                        [-1, 8, -1],
                                        [-1, -1, -1]])
    

        # Count from the standard
        self.count = 0
        
        # Start analysing
        self.image_looping()
        


    def save_image(self, img, subdir, file_name, image_to_save):
        
        # Determine the name of the file we want to output
        subdir = subdir[int(len(self.image_path)):]
        
        if image_to_save == 'vision_object_edge_1':
            image_output = self.path +"/vision_object_edge_1/" + subdir
        elif image_to_save == 'vision_object_edge_2':
            image_output = self.path + "/vision_object_edge_2/" + subdir
        
        
        for i in range(len(subdir)):
            cv2.imwrite(os.path.join(image_output, str(file_name)), img)
            #cv2.waitKey(0)
        

    
    def image_looping(self):

        # Loop throught nested files 
        for subdir, dirs, files in os.walk(self.image_path):

            if self.count != 0:

                for image_path in os.listdir(subdir):
                    
                    image = os.path.join(subdir, image_path)

                    print(self.count)

                    img = cv2.imread(image, -1)
                    file_name = basename(image)
                    
                    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    img_my_kernel_2 = cv2.filter2D(gray_scale, -1, self.kernel_2)
                    edge_detector_known = cv2.filter2D(gray_scale, -1, self.edge_detector)
                    
                    self.save_image(img_my_kernel_2, subdir, file_name, image_to_save = self.input_auguments + "_edge_1")
                    self.save_image(edge_detector_known, subdir, file_name, image_to_save = self.input_auguments + "_edge_2")

            self.count += 1
