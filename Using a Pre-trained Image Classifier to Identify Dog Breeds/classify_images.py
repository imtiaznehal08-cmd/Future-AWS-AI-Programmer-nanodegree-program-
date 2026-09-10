#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/classify_images.py
#                                                                             
# PROGRAMMER: Nehal Imtiaz
# DATE CREATED: 25 August 2026                                 
# REVISED DATE: 
# PURPOSE: Create a function classify_images that uses the classifier function 
#          to create the classifier labels and then compares the classifier 
#          labels to the pet image labels. This function inputs:
#            -The Image Folder as image_dir within classify_images and function 
#             and as in_arg.dir for function call within main. 
#            -The results dictionary as results_dic within classify_images 
#             function and results for the functin call within main.
#            -The CNN model architecture as model wihtin classify_images function
#             and in_arg.arch for the function call within main. 
#           This function uses the extend function to add items to the list 
#           that's the 'value' of the results dictionary. You will be adding the
#           classifier label as the item at index 1 of the list and the comparison 
#           of the pet and classifier labels as the item at index 2 of the list.
#
##
# Imports classifier function for using CNN to classify images 
from classifier import classifier 

def classify_images(images_dir, results_dic, model):
    """
    Creates classifier labels with classifier function, compares pet labels to 
    the classifier labels, and adds the classifier label and the comparison of 
    the labels to the results dictionary using the extend function. Be sure to
    format the classifier labels so that they will match your pet image labels.
    The format will include putting the classifier labels in all lower case 
    letters and strip the leading and trailing whitespace characters from them.
    Parameters: 
      images_dir - The (full) path to the folder of images that are to be
                   classified by the classifier function (string)
      results_dic - Results Dictionary with 'key' as image filename and 'value'
                    as a List. Where the list will contain the following items: 
                  index 0 = pet image label (string)
                --- where index 1 & index 2 are added by this function ---
                  NEW - index 1 = classifier label (string)
                  NEW - index 2 = 1/0 (int)  where 1 = match between pet image
                    and classifer labels and 0 = no match between labels
      model - Indicates which CNN model architecture will be used by the 
              classifier function to classify the pet images,
              values must be either: resnet alexnet vgg (string)
     Returns:
           None - results_dic is mutable data type so no return needed.         
    """
    # Process all files in the results_dic
    for key in results_dic:
        
        # 1. Create the full image path (e.g., "pet_images/Basenji_00963.jpg")
        # Ensure we add a slash between the directory and the filename if missing
        if images_dir.endswith('/'):
            img_path = images_dir + key
        else:
            img_path = images_dir + '/' + key
            
        # 2. Run the classifier function on the image
        # This returns a string of the predicted labels
        model_label = classifier(img_path, model)
        
        # 3. Format the model_label (lowercase and strip whitespace)
        model_label = model_label.lower().strip()
        
        # 4. Compare the truth (pet label) to the model_label
        truth = results_dic[key][0]
        
        # If the true pet label is found inside the model's output label string
        if truth in model_label:
            # Append the model label and a 1 (indicating a match)
            results_dic[key].extend([model_label, 1])
        else:
            # Append the model label and a 0 (indicating no match)
            results_dic[key].extend([model_label, 0])