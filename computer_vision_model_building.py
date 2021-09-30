from header_imports import *

class computer_vision_building(object):

    def __init__(self, model_type, image_type):


        self.images = []
        self.filename = []
        self.image_file = []
        # 0 for False and 1 for True for label name
        self.label_name = []
        self.number_classes = 20
        self.image_size = 240
        self.path = "Data/"
        self.true_path  = "split_images_folders/"
        self.image_type = image_type


        # Determine
        if self.image_type == "normal":
            self.true_path = self.true_path + "vision_object/"
        elif self.image_type == "edge_1":
            self.true_path = self.true_path + "vision_object_edge_1/"
        elif self.image_type == "edge_2":
            self.true_path = self.true_path + "vision_object_edge_2/"
            
        self.valid_images = [".jpg",".png"]
        self.input_shape = None
        self.advanced_categories = ["aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog","horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

        # Split training data variables
        self.X_train = None
        self.X_test = None
        self.Y_train_vec = None
        self.Y_test_vec = None

        # model informations
        self.model = None
        
        # model summary path 
        self.model_summary = "model_summary/"

        self.optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        self.create_model_type = model_type
        
        # Check validity
        self.check_valid(self.advanced_categories[0])
        self.check_valid(self.advanced_categories[1])
        self.check_valid(self.advanced_categories[2])
        self.check_valid(self.advanced_categories[3])
        self.check_valid(self.advanced_categories[4])
        self.check_valid(self.advanced_categories[5])
        self.check_valid(self.advanced_categories[6])
        self.check_valid(self.advanced_categories[7])
        self.check_valid(self.advanced_categories[8])
        self.check_valid(self.advanced_categories[9])
        self.check_valid(self.advanced_categories[10])
        self.check_valid(self.advanced_categories[11])
        self.check_valid(self.advanced_categories[12])
        self.check_valid(self.advanced_categories[13])
        self.check_valid(self.advanced_categories[14])
        self.check_valid(self.advanced_categories[15])
        self.check_valid(self.advanced_categories[16])
        self.check_valid(self.advanced_categories[17])
        self.check_valid(self.advanced_categories[18])
        self.check_valid(self.advanced_categories[19])
            
        # Resize image
        self.resize_image_and_label_image(self.advanced_categories[0])
        self.resize_image_and_label_image(self.advanced_categories[1])
        self.resize_image_and_label_image(self.advanced_categories[2])
        self.resize_image_and_label_image(self.advanced_categories[3])
        self.resize_image_and_label_image(self.advanced_categories[4])
        self.resize_image_and_label_image(self.advanced_categories[5])
        self.resize_image_and_label_image(self.advanced_categories[6])
        self.resize_image_and_label_image(self.advanced_categories[7])
        self.resize_image_and_label_image(self.advanced_categories[8])
        self.resize_image_and_label_image(self.advanced_categories[9])
        self.resize_image_and_label_image(self.advanced_categories[10])
        self.resize_image_and_label_image(self.advanced_categories[11])
        self.resize_image_and_label_image(self.advanced_categories[12])
        self.resize_image_and_label_image(self.advanced_categories[13])
        self.resize_image_and_label_image(self.advanced_categories[14])
        self.resize_image_and_label_image(self.advanced_categories[15])
        self.resize_image_and_label_image(self.advanced_categories[16])
        self.resize_image_and_label_image(self.advanced_categories[17])
        self.resize_image_and_label_image(self.advanced_categories[18])
        self.resize_image_and_label_image(self.advanced_categories[19])

        
        # Numpy array
        self.image_file = np.array(self.image_file)
        self.label_name = np.array(self.label_name)
        self.label_name = self.label_name.reshape((len(self.image_file),1))

        self.splitting_data_normalize()

        if self.create_model_type == "model1":
            self.create_models_1()
        elif self.create_model_type == "model2":
            self.create_models_2()
        elif self.create_model_type == "model3":
            self.create_model_3()

        # Saving model summary
        self.save_model_summary()
        
        print("finished")



    # Checks to see if the image is valid or not
    def check_valid(self, input_file):
        for img in os.listdir(self.true_path + input_file):
            ext = os.path.splitext(img)[1]
            if ext.lower() not in self.valid_images:
                continue
    

    # Resize images
    def resize_image_and_label_image(self, input_file):
        for image in os.listdir(self.true_path + input_file):
            
            image_resized = cv2.imread(os.path.join(self.true_path + input_file,image))
            image_resized = cv2.resize(image_resized,(self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
            self.image_file.append(image_resized)

            if input_file == "aeroplane":
                self.label_name.append(0)
            elif input_file == "bicycle":
                self.label_name.append(1)
            elif input_file == "bird":
                self.label_name.append(2)
            elif input_file == "boat":
                self.label_name.append(3)
            elif input_file == "bottle":
                self.label_name.append(4)
            elif input_file == "bus":
                self.label_name.append(5)
            elif input_file == "car":
                self.label_name.append(6)
            elif input_file == "cat":
                self.label_name.append(7)
            elif input_file == "chair":
                self.label_name.append(8)
            elif input_file == "cow":
                self.label_name.append(9)
            elif input_file == "diningtable":
                self.label_name.append(10)
            elif input_file == "dog":
                self.label_name.append(11)
            elif input_file == "horse":
                self.label_name.append(12)
            elif input_file == "motorbike":
                self.label_name.append(13)
            elif input_file == "person":
                self.label_name.append(14)
            elif input_file == "pottedplant":
                self.label_name.append(15)
            elif input_file == "sheep":
                self.label_name.append(16)
            elif input_file == "sofa":
                self.label_name.append(17)
            elif input_file == "train":
                self.label_name.append(18)
            elif input_file == "tvmonitor":
                self.label_name.append(19)
            else:
                print("error")



    # Split training data and testing Data and makes it random and normalized it
    def splitting_data_normalize(self):
        self.X_train, self.X_test, self.Y_train_vec, self.Y_test_vec = train_test_split(self.image_file, self.label_name, test_size = 0.10, random_state = 42)

        self.input_shape = self.X_train.shape[1:]
        
        self.Y_train = tf.keras.utils.to_categorical(self.Y_train_vec, self.number_classes)
        self.Y_test = tf.keras.utils.to_categorical(self.Y_test_vec, self.number_classes)

        # Normalize
        self.X_train = self.X_train.astype("float32")
        self.X_train /= 255
        self.X_test = self.X_test.astype("float32")
        self.X_test /= 255


    def get_model(self):
        return self.model

    def get_data(self):
        return self.X_train , self.Y_train, self.X_test, self.Y_test, self.Y_test_vec

    def get_categories(self):
        # Number of categories
        return self.advanced_categories


    def create_models_1(self):

        self.model = Sequential()

        # First Hitten Layer with 64, 7, 7
        self.model.add(Conv2D(64,(7,7), strides = (1,1), padding="same", input_shape = self.input_shape, activation = "relu"))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size = (4,4)))
        self.model.add(Dropout(0.25))
    
        # Second Hitten Layer 32, 7, 7
        self.model.add(Conv2D(32,(7,7), strides = (1,1), padding="same", activation = "relu"))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size = (2,2)))
        self.model.add(Dropout(0.25))
    
        # Third Hitten Layer 32, 7, 7
        self.model.add(Conv2D(16,(7,7), strides = (1,1), padding="same", activation = "relu"))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size = (1,1)))
        self.model.add(Dropout(0.25))
    
        # last layer, output Layer
        self.model.add(Flatten())
        self.model.add(Dense(units = self.number_classes, activation = 'softmax', input_dim=2))

        self.model.compile(loss = "binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        return self.model


    
    def create_models_2(self):

        self.model = Sequential()

        self.model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu", input_shape = self.input_shape))
        self.model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu"))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.25))

        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.25))
        self.model.add(Flatten())

        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Dense(units = self.number_classes, activation="softmax"))
        
        self.model.compile(loss = "binary_crossentropy", optimizer="adam", metrics=["accuracy"])
	
        return self.model


    def create_model_3(self):

        self.model = Sequential()
        
        self.MyConv(first = True)
        self.MyConv()
        self.MyConv()
        self.MyConv()

        self.model.add(Flatten())
        self.model.add(Dense(units = self.number_classes, activation = 'softmax', input_dim=2))

        self.model.compile(loss = 'binary_crossentropy', optimizer ='adam', metrics= ['accuracy'])
        
        return self.model

        

    def MyConv(self, first = False):

        if first == False:
            self.model.add(Conv2D(64, (4, 4),strides = (1,1), padding='same',
                input_shape = self.input_shape))
        else:
            self.model.add(Conv2D(64, (4, 4),strides = (1,1), padding='same',
                 input_shape = self.input_shape))
    
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Conv2D(32, (4, 4),strides = (1,1),padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.25))



    # Save the model summery as a txt file
    def save_model_summary(self):

        with open(self.model_summary + self.create_model_type +"_summary_architecture_" + str(self.number_classes) +".txt", "w+") as model:
            with redirect_stdout(model):
                self.model.summary()


    



    
