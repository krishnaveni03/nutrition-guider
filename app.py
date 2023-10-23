# Import Dependencies
import os
import numpy as np
import pandas as pd
from six import reraise
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Create a Flask App
app = Flask(__name__)

# Load the model
Model_path = "model_inceptionV3.h5"
model = load_model(Model_path)

# Define a list of food item descriptions, fats, and nutritional information
food_info = [
    {
        "name": "Burger",
        "description": "A typical fast-food burger contains approximately 20-30 grams of fat, with variations depending on the size and toppings. It typically provides around 12-15 grams of protein. However, these values can vary widely depending on the type of burger and ingredients used.",
        "fat_content": "20-30 grams of fat",
        "nutritional_info": "12-15 grams of protein",
    },
    {
        "name": "Butter Naan",
        "description": "Butternut squash is low in fat, containing approximately 0.2 grams of fat per 100 grams. It is a good source of dietary fiber and provides around 1 gram of protein per 100 grams, making it a healthy, low-fat addition to a balanced diet.",
        "fat_content": "0.2 grams of fat per 100 grams",
        "nutritional_info": "1 gram of protein per 100 grams",
    },
    {
        "name": "Chai",
        "description": "Chai is a popular Indian tea made with a combination of black tea, milk, sugar, and spices. The fat content can vary depending on the type of milk and sugar used. It typically contains minimal protein.",
        "fat_content": "Varies depending on ingredients",
        "nutritional_info": "Low protein content",
    },
    {
        "name": "Chapati",
        "description": "Chapati is a type of unleavened bread popular in South Asia. It is typically low in fat and a good source of carbohydrates, providing moderate protein content.",
        "fat_content": "Low in fat",
        "nutritional_info": "Moderate protein content",
    },
    {
        "name": "Chole Bhature",
        "description": "Chole Bhature is a North Indian dish. The bhature (deep-fried bread) can be high in fat, while the chole (chickpea curry) can be a source of protein.",
        "fat_content": "Varies depending on preparation",
        "nutritional_info": "Protein from chickpeas",
    },
    {
        "name": "Dal Makhani",
        "description": "Dal Makhani is a rich, creamy lentil dish. It contains a moderate amount of fat due to butter and cream and provides protein from lentils.",
        "fat_content": "Moderate fat content",
        "nutritional_info": "Protein from lentils",
    },
    {
        "name": "Dhokla",
        "description": "Dhokla is a steamed Indian snack made from fermented rice and chickpea flour. It is low in fat and provides some protein.",
        "fat_content": "Low in fat",
        "nutritional_info": "Moderate protein content",
    },
    {
        "name": "Fried Rice",
        "description": "Fried rice is a popular Chinese dish with variations. It can contain variable fat content depending on ingredients, and it may provide some protein from meat or tofu.",
        "fat_content": "Varies depending on preparation",
        "nutritional_info": "Protein from meat or tofu",
    },
    {
        "name": "Idli",
        "description": "Idli is a South Indian steamed rice cake. It is low in fat and a source of carbohydrates. The protein content is relatively low.",
        "fat_content": "Low in fat",
        "nutritional_info": "Low protein content",
    },
    {
        "name": "Jalebi",
        "description": "Jalebi is a sweet Indian dessert made by deep-frying batter and soaking it in sugar syrup. It is high in fat and sugar, with minimal protein content.",
        "fat_content": "High fat content",
        "nutritional_info": "High sugar content, low protein",
    },
    {
        "name": "Kaathi Rolls",
        "description": "Kaathi Rolls are a popular street food in India. They are typically made with flatbreads stuffed with various fillings, which can vary in fat content depending on ingredients. The protein content comes from the filling.",
        "fat_content": "Varies depending on ingredients",
        "nutritional_info": "Protein from the filling",
    },
    {
        "name": "Kadai Paneer",
        "description": "Kadai Paneer is a North Indian dish made with paneer (Indian cheese), bell peppers, and spices. It contains a moderate amount of fat from the cheese and cream and provides protein from paneer.",
        "fat_content": "Moderate fat content",
        "nutritional_info": "Protein from paneer",
    },
    {
        "name": "Kulfi",
        "description": "Kulfi is a traditional Indian ice cream made with milk, sugar, and flavorings. It contains a moderate amount of fat and sugar and provides some protein from milk.",
        "fat_content": "Moderate fat content",
        "nutritional_info": "Protein from milk",
    },
    {
        "name": "Masala Dosa",
        "description": "Masala Dosa is a South Indian dish consisting of a crispy fermented rice and lentil crepe filled with spiced potatoes. It is low in fat and a source of carbohydrates and protein from lentils and rice.",
        "fat_content": "Low in fat",
        "nutritional_info": "Protein from lentils and rice",
    },
    {
        "name": "Momos",
        "description": "Momos are dumplings popular in South Asian cuisine. They are steamed or fried and may vary in fat content based on the cooking method and filling. Protein comes from the filling.",
        "fat_content": "Varies depending on preparation",
        "nutritional_info": "Protein from the filling",
    },
    {
        "name": "Paani Puri",
        "description": "Paani Puri is a popular Indian street food snack consisting of hollow, crispy balls filled with spicy flavored water, tamarind chutney, and potato filling. It is low in fat and provides carbohydrates from the shell and filling.",
        "fat_content": "Low in fat",
        "nutritional_info": "Carbohydrates from shell and filling",
    },
    {
        "name": "Pakode",
        "description": "Pakode are deep-fried fritters made from chickpea flour and various ingredients. They are high in fat due to deep frying and provide some protein from chickpea flour and ingredients.",
        "fat_content": "High fat content",
        "nutritional_info": "Protein from chickpea flour and ingredients",
    },
    {
        "name": "Pav Bhaji",
        "description": "Pav Bhaji is a popular Indian street food dish consisting of a spicy vegetable curry (bhaji) served with soft bread rolls (pav). The fat content can vary depending on the use of butter, and it provides some protein from vegetables and legumes.",
        "fat_content": "Varies depending on butter use",
        "nutritional_info": "Protein from vegetables and legumes",
    },
    {
        "name": "Pizza",
        "description": "Pizza is a versatile dish with variations in ingredients. It can be high in fat due to cheese and meat toppings, and it provides protein from cheese and meat. The nutritional content can vary widely depending on the type of pizza.",
        "fat_content": "Varies depending on ingredients",
        "nutritional_info": "Protein from cheese and meat",
    },
    {
        "name": "Samosa",
        "description": "Samosa is a popular Indian snack consisting of a deep-fried pastry shell filled with spiced potatoes, peas, and sometimes meat. It is high in fat due to deep frying and provides some protein from the filling.",
        "fat_content": "High fat content",
        "nutritional_info": "Protein from the filling",
    }
    # Add descriptions, fats, and nutritional info for the remaining food items
]

# Create a function to take an image and predict the class
def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)

    # Use the food_info list to get the description, fat content, and nutritional info
    food = food_info[preds[0]]

    result = "This item is {name}.\n\nDescription:\n{description}\n\nFat Content:\n{fat_content}\n\nNutritional Info:\n{nutritional_info}".format(
        name=food['name'],
        description=food['description'],
        fat_content=food['fat_content'],
        nutritional_info=food['nutritional_info']
    )

    return result

@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def uploads():
    if request.method == 'POST':
        # Get the File from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make Prediction
        preds = model_predict(file_path, model)

        return preds

    return "No image uploaded."

if __name__ == '__main__':
    app.run()
