import torch.nn as nn

def load_settings(dataset):
    
    if dataset=="cloth":
        FILENAME = 'Womens Clothing E-Commerce Reviews.csv'
        categorical_var = ['Division Name', 'Department Name', 'Class Name']
        numerical_var = ['Age', 'Positive Feedback Count']
        text_var = 'Review'
        MAX_LEN_QUANTILE = 0.9
        N_CLASSES = 5 # number of classes in classification task
        WEIGHT_DECAY = 1e-5 # optimizer's weight decay
        FACTOR = 0.9 # LR scheduler factor
        N_EPOCHS = 100 # max number of epochs
        split_val = 0.2 # split share for validation and for test
        CRITERION = nn.CrossEntropyLoss()
        N_SEED = 5
        DROPOUT = 0.1
        
    if dataset=="wine_10":
        FILENAME = 'winemag-data-130k-v2.csv'
        categorical_var = ['country', 'year']
        numerical_var = ['points', 'price']
        text_var = 'description'
        MAX_LEN_QUANTILE = 0.9
        N_CLASSES = 10 # number of classes in classification task
        WEIGHT_DECAY = 1e-5 # optimizer's weight decay
        FACTOR = 0.9 # LR scheduler factor
        N_EPOCHS = 100 # max number of epochs
        split_val = 0.2 # split share for validation and for test
        CRITERION = nn.CrossEntropyLoss()
        N_SEED = 5
        DROPOUT = 0.1
        

    if dataset=="wine_100":
        FILENAME = 'winemag-data-130k-v2.csv'
        categorical_var = ['country', 'year']
        numerical_var = ['points', 'price']
        text_var = 'description'
        MAX_LEN_QUANTILE = 0.9
        N_CLASSES = 100 # number of classes in classification task
        WEIGHT_DECAY = 1e-5 # optimizer's weight decay
        FACTOR = 0.9 # LR scheduler factor
        N_EPOCHS = 100 # max number of epochs
        split_val = 0.2 # split share for validation and for test
        CRITERION = nn.CrossEntropyLoss()
        N_SEED = 5
        DROPOUT = 0.1
        
    if dataset=="kick":
        FILENAME = 'kickstarter_train.csv'
        categorical_var = ['country', 'currency', 'disable_communication']
        numerical_var = ['log_goal', 'backers_count', 'duration']
        text_var = 'description'
        MAX_LEN_QUANTILE = 0.9
        N_CLASSES = 2 # number of classes in classification task
        WEIGHT_DECAY = 1e-5 # optimizer's weight decay
        FACTOR = 0.9 # LR scheduler factor
        N_EPOCHS = 100 # max number of epochs
        split_val = 0.2 # split share for validation and for test
        CRITERION = nn.CrossEntropyLoss()
        N_SEED = 5
        DROPOUT = 0.1

        
    if dataset=="airbnb":
        FILENAME = 'cleansed_listings_dec18.csv'
        categorical_var = ['host_location', 'host_since_year','host_is_superhost', 'host_neighborhood', 'host_has_profile_pic', 'host_identity_verified', 
        'neighborhood', 'city', 'smart_location', 'suburb', 'state', 'is_location_exact', 'property_type', 'room_type', 'bed_type', 'instant_bookable', 
        'cancellation_policy', 'require_guest_profile_picture', 'require_guest_phone_verification', 'host_response_time', 'calendar_updated', 'host_verifications',
        'last_review_year']
        numerical_var = ['host_response_rate', 'latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'security_deposit', 'cleaning_fee', 'guests_included',
        'extra_people', 'minimum_nights', 'maximum_nights', 'availability_30', 'availability_60', 'availability_90', 'availability_365', 'number_of_reviews', 'review_scores_rating',
        'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value',
        'calculated_host_listings_count', 'reviews_per_month']
        text_var = 'description'
        MAX_LEN_QUANTILE = 0.9
        N_CLASSES = 10 # number of classes in classification task
        WEIGHT_DECAY = 1e-5 # optimizer's weight decay
        FACTOR = 0.9 # LR scheduler factor
        N_EPOCHS = 100 # max number of epochs
        split_val = 0.2 # split share for validation and for test
        CRITERION = nn.CrossEntropyLoss()
        N_SEED = 5
        DROPOUT = 0.1 
        
    if dataset=="pet":
        FILENAME = 'petfinder_train.csv'
        categorical_var = ["Type", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3", "MaturitySize", "FurLength", "Vaccinated", "Dewormed", "Sterilized", "Health", "State"]
        numerical_var = ["Age", "Quantity", "Fee", "VideoAmt", "PhotoAmt"]
        text_var = 'Description'
        MAX_LEN_QUANTILE = 0.9
        N_CLASSES = 5 # number of classes in classification task
        WEIGHT_DECAY = 1e-5 # optimizer's weight decay
        FACTOR = 0.9 # LR scheduler factor
        N_EPOCHS = 100 # max number of epochs
        split_val = 0.2 # split share for validation and for test
        CRITERION = nn.CrossEntropyLoss()
        N_SEED = 5
        DROPOUT = 0.1
        

    if dataset=="salary":
        FILENAME = 'Data_Scientist_Salary_Train.csv'
        categorical_var = ['location', 'company_name_encoded']
        numerical_var = ['experience_int']
        text_var = 'description'
        MAX_LEN_QUANTILE = 0.9
        N_CLASSES = 6 # number of classes in classification task
        WEIGHT_DECAY = 1e-5 # optimizer's weight decay
        FACTOR = 0.9 # LR scheduler factor
        N_EPOCHS = 100 # max number of epochs
        split_val = 0.2 # split share for validation and for test
        CRITERION = nn.CrossEntropyLoss()
        N_SEED = 5
        DROPOUT = 0.1
        
    if dataset=="jigsaw":
        FILENAME = 'jigsaw_train_100k.csv'
        categorical_var = ['rating']
        numerical_var = ['funny', 'wow', 'sad', 'likes', 'disagree']
        text_var = 'comment_text'
        MAX_LEN_QUANTILE = 0.9
        N_CLASSES = 2 # number of classes in classification task
        WEIGHT_DECAY = 1e-5 # optimizer's weight decay
        FACTOR = 0.9 # LR scheduler factor
        N_EPOCHS = 100 # max number of epochs
        split_val = 0.2 # split share for validation and for test
        CRITERION = nn.CrossEntropyLoss()
        N_SEED = 5
        DROPOUT = 0.1
        
    return FILENAME, categorical_var, numerical_var, text_var, MAX_LEN_QUANTILE, N_CLASSES, WEIGHT_DECAY, FACTOR, N_EPOCHS, split_val, CRITERION, N_SEED, DROPOUT

def load_pretrained_settings():
    """
    Load hyperparameters for pretrained models
    """
    LR = 5e-5
    BATCH_SIZE = 32
    D_FC = 32
    N_EPOCHS = 100
    N_HEADS = 8
    N_LAYERS = 3

    return LR, BATCH_SIZE, D_FC, N_EPOCHS, N_HEADS, N_LAYERS 