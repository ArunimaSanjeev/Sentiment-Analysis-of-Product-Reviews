# Sentiment Analysis of Product Review

Data Mining Project 
## Project Overview

This project performs **multimodal sentiment analysis** by combining image and text data to evaluate product quality and user experience.

The system integrates:

* **CNN (Convolutional Neural Network)** — to classify *Damaged vs. Undamaged* product images
* **Bi-LSTM (Bidirectional Long Short-Term Memory)** — to classify *Positive vs. Negative* text reviews
* **Rule-Based Decision Layer** — to combine both predictions and determine the *Overall Experience*

This approach demonstrates how image and textual cues can jointly enhance product quality assessment and customer sentiment understanding.


## Dataset Description

### Image Dataset

| Category         | Description                         | Source                                                                                                                   | Approx. Images |
| ---------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | -------------- |
| Damaged Plates   | Plates with visible cracks or chips | [Broken Plates – Roboflow Universe](https://universe.roboflow.com/asoiu-x4iyw/broken-plates/dataset/1)                   | ~250           |
| Undamaged Plates | Clean and intact plates             | [Sushi Mentai Plates – Roboflow Universe](https://universe.roboflow.com/sushi-mentai-plate/sushi-mentai-plate/dataset/2) | ~50            |

**Preprocessing:**

* Resize: 224×224 pixels
* Normalize pixel values to [0,1]
* Data augmentation (rotation, shift, zoom, flip)
* Validation split: 15%


### Text Review Dataset

| File                         | Description              | Label        | Samples |
| ---------------------------- | ------------------------ | ------------ | ------- |
| `positive_plate_reviews.csv` | Positive product reviews | 1 (Positive) | ~300    |
| `negative_plate_reviews.csv` | Negative product reviews | 0 (Negative) | ~300    |

**Preprocessing:**

* Remove duplicates and empty rows
* Text cleaning (lowercasing, punctuation, stopword removal)
* Tokenization and padding for uniform sequence length


##  Model Architecture

### CNN Model (Image Classification)

* Input: 224×224 RGB images
* Layers: Conv2D → MaxPooling → Flatten → Dense
* Output: Binary classification (Damaged / Undamaged)

### Bi-LSTM Model (Text Classification)

* Input: Tokenized text sequences
* Layers: Embedding → Bidirectional LSTM → Dense
* Output: Sentiment polarity (Positive / Negative)

### Rule-Based Decision Layer

| Image     | Sentiment | Final Decision      |
| --------- | --------- | ------------------- |
| Undamaged | Positive  | Positive Experience |
| Undamaged | Negative  | Negative Experience |
| Damaged   | Positive  | Negative Experience |
| Damaged   | Negative  | Negative Experience |


## Tools and Technologies

* **Python 3.x**
* **TensorFlow / Keras**
* **NumPy, Pandas, scikit-learn**
* **Google Colab / Jupyter Notebook**


## Results & Testing

* Trained CNN and Bi-LSTM models independently
* Combined results through rule-based logic
* Real-time demo accepts an image and user review comment, outputs predicted product condition, sentiment, and overall experience.


## How to Run

1. Upload datasets and review CSVs to your working directory.
2. Run the CNN training notebook to generate `plate_cnn_model.h5`.
3. Run the Bi-LSTM training notebook to generate `plate_nlp_model.h5`.
4. Use the simulation code block to test combined predictions interactively.
