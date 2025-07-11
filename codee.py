import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import random
import matplotlib.pyplot as plt

# Declare knn and nb as global variables
knn = None
nb = None

# Function to train machine learning models
def train_models():
    global knn, nb  # Use global variables to store the trained models

    # Create a random dataset for now (Can be replaced by actual user input)
    df = create_sample_dataset(100)

    # Map categorical features to numerical values
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['Activity Level'] = df['Activity Level'].map({'Low': 0, 'Moderate': 1, 'High': 2})

    # Define features (X) and target variable (y)
    X = df[['Age', 'Weight', 'Height', 'Gender', 'Activity Level', 'BMI', 'Sleep Hours', 'Smoking/Alcohol']]
    y = df['Activity Level']

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # K-Nearest Neighbors Model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Naive Bayes Model
    nb = GaussianNB()
    nb.fit(X_train, y_train)

    # Show results in message box
    messagebox.showinfo("Model Results", f"KNN Model Accuracy: {knn.score(X_test, y_test):.2f}\nNaive Bayes Accuracy: {nb.score(X_test, y_test):.2f}")

    # Display accuracy comparison chart
    plt.bar(["KNN", "Naive Bayes"], [knn.score(X_test, y_test), nb.score(X_test, y_test)], color=['blue', 'green'])
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.show()

# Create a sample dataset (this would be replaced by actual user input in a real-world case)
def create_sample_dataset(n_samples=100):
    data = []
    for _ in range(n_samples):
        age = random.randint(18, 60)
        weight = random.randint(50, 100)
        height = random.randint(150, 200)
        gender = random.choice(["Male", "Female"])
        activity_level = random.choice(["Low", "Moderate", "High"])
        bmi = round(weight / ((height / 100) ** 2), 2)  # BMI calculation
        sleep_hours = random.randint(4, 10)
        smoking_alcohol = random.choice([0, 1])  # 0 = No, 1 = Yes
        data.append({"Age": age, "Weight": weight, "Height": height, "Gender": gender, 
                     "Activity Level": activity_level, "BMI": bmi, "Sleep Hours": sleep_hours, 
                     "Smoking/Alcohol": smoking_alcohol})
    
    return pd.DataFrame(data)

# Function to submit the user input data
def submit_data():
    global knn  # Use the global knn model

    if knn is None:  # Check if the model has been trained
        messagebox.showerror("Model Not Trained", "Please train the model first before making predictions.")
        return

    try:
        # Collect user input
        age = int(age_entry.get())
        weight = float(weight_entry.get())
        height = float(height_entry.get())
        gender = gender_var.get()
        activity_level = activity_level_var.get()
        sleep_hours = int(sleep_entry.get())
        smoking_alcohol = 1 if smoking_var.get() == "Yes" else 0

        # Calculate BMI
        bmi = round(weight / ((height / 100) ** 2), 2)

        # Prepare user data for prediction
        user_data = pd.DataFrame([[age, weight, height, 0 if gender == "Male" else 1, 
                                   0 if activity_level == "Low" else (1 if activity_level == "Moderate" else 2), 
                                   bmi, sleep_hours, smoking_alcohol]], 
                                 columns=['Age', 'Weight', 'Height', 'Gender', 'Activity Level', 'BMI', 'Sleep Hours', 'Smoking/Alcohol'])
        user_data = StandardScaler().fit_transform(user_data)

        # Make predictions using KNN
        prediction = knn.predict(user_data)
        predicted_level = "Low" if prediction == 0 else ("Moderate" if prediction == 1 else "High")

        # Generate personalized recommendations
        recommendations = {
            "Low": "Start with light activities like walking or yoga. Aim for 30 minutes of exercise daily.",
            "Moderate": "Maintain a balanced routine with a mix of cardio and strength training. Ensure proper rest and hydration.",
            "High": "Great job! Focus on consistency and incorporate advanced techniques like HIIT or endurance training."
        }

        # Generate BMI-related health tips
        bmi_tip = ""
        if bmi < 18.5:
            bmi_tip = "Your BMI indicates underweight. Consider increasing your calorie intake with nutrient-rich foods."
        elif 18.5 <= bmi < 24.9:
            bmi_tip = "Your BMI is in the normal range. Maintain a balanced diet and regular exercise."
        elif 25 <= bmi < 29.9:
            bmi_tip = "Your BMI indicates overweight. Incorporate more cardio and monitor your diet."
        else:
            bmi_tip = "Your BMI indicates obesity. Focus on low-impact exercises and consult a nutritionist."

        # Display the detailed output
        output_message = f"""
        **Prediction:**
        Predicted Activity Level: {predicted_level}

        **BMI:**
        Your BMI: {bmi} ({'Underweight' if bmi < 18.5 else 'Normal' if bmi < 24.9 else 'Overweight' if bmi < 29.9 else 'Obese'})

        **Fitness Recommendations:**
        {recommendations[predicted_level]}

        **Health Tip:**
        {bmi_tip}

        **Lifestyle Advice:**
        Sleep Hours: {sleep_hours} (Aim for 7-9 hours for optimal health)
        Smoking/Alcohol: {'Avoid smoking and alcohol for better fitness outcomes.' if smoking_alcohol else 'Good job avoiding harmful substances!'}
        """
        messagebox.showinfo("Fitness Trainer Output", output_message)

        # Update prediction label
        prediction_label.config(text=f"Predicted Activity Level: {predicted_level}")

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid data for all fields.")

# Create the main window
root = tk.Tk()
root.title("AI-Powered Personalized Fitness Trainer")
root.geometry("800x850")
root.configure(bg="#f0f8ff")

# Add a label
label = tk.Label(root, text="Welcome to the AI-Powered Personalized Fitness Trainer", font=("Arial", 18, "bold"), fg="#4b0082", bg="#f0f8ff")
label.pack(pady=20)

# Create the user input form
frame = tk.Frame(root, bg="#e6e6fa", bd=5, relief=tk.RIDGE)
frame.pack(pady=20, padx=20)

# Age
age_label = tk.Label(frame, text="Age:", font=("Arial", 14, "bold"), bg="#e6e6fa")
age_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
age_entry = tk.Entry(frame, font=("Arial", 12), width=30)
age_entry.grid(row=0, column=1, padx=10, pady=10)

# Weight
weight_label = tk.Label(frame, text="Weight (kg):", font=("Arial", 14, "bold"), bg="#e6e6fa")
weight_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
weight_entry = tk.Entry(frame, font=("Arial", 12), width=30)
weight_entry.grid(row=1, column=1, padx=10, pady=10)

# Height
height_label = tk.Label(frame, text="Height (cm):", font=("Arial", 14, "bold"), bg="#e6e6fa")
height_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")
height_entry = tk.Entry(frame, font=("Arial", 12), width=30)
height_entry.grid(row=2, column=1, padx=10, pady=10)

# Gender
gender_label = tk.Label(frame, text="Gender:", font=("Arial", 14, "bold"), bg="#e6e6fa")
gender_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")
gender_var = tk.StringVar()
gender_var.set("Male")
gender_male_radio = tk.Radiobutton(frame, text="Male", variable=gender_var, value="Male", font=("Arial", 12), bg="#e6e6fa")
gender_female_radio = tk.Radiobutton(frame, text="Female", variable=gender_var, value="Female", font=("Arial", 12), bg="#e6e6fa")
gender_male_radio.grid(row=3, column=1, padx=10, pady=10, sticky="w")
gender_female_radio.grid(row=3, column=1, padx=10, pady=10, sticky="e")

# Activity Level
activity_level_label = tk.Label(frame, text="Activity Level:", font=("Arial", 14, "bold"), bg="#e6e6fa")
activity_level_label.grid(row=4, column=0, padx=10, pady=10, sticky="w")
activity_level_var = tk.StringVar()
activity_level_var.set("Low")
activity_level_low_radio = tk.Radiobutton(frame, text="Low", variable=activity_level_var, value="Low", font=("Arial", 12), bg="#e6e6fa")
activity_level_moderate_radio = tk.Radiobutton(frame, text="Moderate", variable=activity_level_var, value="Moderate", font=("Arial", 12), bg="#e6e6fa")
activity_level_high_radio = tk.Radiobutton(frame, text="High", variable=activity_level_var, value="High", font=("Arial", 12), bg="#e6e6fa")
activity_level_low_radio.grid(row=4, column=1, padx=10, pady=10, sticky="w")
activity_level_moderate_radio.grid(row=4, column=1, padx=10, pady=10)
activity_level_high_radio.grid(row=4, column=1, padx=10, pady=10, sticky="e")

# Sleep Hours
sleep_label = tk.Label(frame, text="Sleep Hours:", font=("Arial", 14, "bold"), bg="#e6e6fa")
sleep_label.grid(row=5, column=0, padx=10, pady=10, sticky="w")
sleep_entry = tk.Entry(frame, font=("Arial", 12), width=30)
sleep_entry.grid(row=5, column=1, padx=10, pady=10)

# Smoking/Alcohol
smoking_label = tk.Label(frame, text="Smoking/Alcohol:", font=("Arial", 14, "bold"), bg="#e6e6fa")
smoking_label.grid(row=6, column=0, padx=10, pady=10, sticky="w")
smoking_var = tk.StringVar()
smoking_var.set("No")
smoking_yes_radio = tk.Radiobutton(frame, text="Yes", variable=smoking_var, value="Yes", font=("Arial", 12), bg="#e6e6fa")
smoking_no_radio = tk.Radiobutton(frame, text="No", variable=smoking_var, value="No", font=("Arial", 12), bg="#e6e6fa")
smoking_yes_radio.grid(row=6, column=1, padx=10, pady=10, sticky="w")
smoking_no_radio.grid(row=6, column=1, padx=10, pady=10, sticky="e")

# Submit Button
submit_button = tk.Button(root, text="Submit Data", font=("Arial", 14, "bold"), bg="#4caf50", fg="white", command=submit_data)
submit_button.pack(pady=20)

# Prediction Label
prediction_label = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#f0f8ff")
prediction_label.pack(pady=10)

# Train Model Button
train_button = tk.Button(root, text="Train Models and Show Accuracy", font=("Arial", 14, "bold"), bg="#2196f3", fg="white", command=train_models)
train_button.pack(pady=10)

# Run the main loop
root.mainloop()

