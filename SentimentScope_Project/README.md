SentimentScope: Sentiment Analysis using TransformersThis repository contains the code and model weights for SentimentScope, a custom transformer-based sentiment analysis project. This project was completed as the final project for the Udacity Future AWS AI Programmer Nanodegree.   Developed from the perspective of a Machine Learning Engineer at Cinescope, the goal of this project is to enhance user recommendation systems by accurately classifying IMDB movie reviews as either positive (1) or negative (0).   📊 Project OverviewObjective: To build, train, and test a transformer model from scratch in PyTorch, specifically modifying it for a binary classification task.   Dataset: The model is trained and evaluated on the standard IMDB dataset, loaded into Pandas DataFrames consisting of 25,000 training reviews and 25,000 testing reviews.   Tokenization: Text preprocessing is handled by Hugging Face's bert-base-uncased tokenizer. This approach utilizes subword tokenization to efficiently handle vocabulary, capping sequence lengths at a maximum of 128 tokens.   🧠 Model Architecture: DemoGPTThe core of this project relies on a custom PyTorch model named DemoGPT. While transformers are traditionally used for generative tasks, this architecture has been explicitly adapted for classification.   Pooling Mechanism: Token-level embeddings are condensed into a single representation vector by applying a mean pooling operation (torch.mean) across the time dimension.   Classification Head: The aggregated vector is then passed into a linear classification output layer mapped to 2 classes (positive and negative) without a bias term.   Hyperparameters: To achieve higher accuracy, the model capacity was upgraded to include 6 stacked transformer blocks, 8 attention heads, a 256-dimensional embedding space, and a 0.2 dropout rate to prevent overfitting.   📈 Training and ResultsTraining Setup: Data was fed into the model using a custom IMDBDataset PyTorch class and a DataLoader with a batch size of 32. The model was optimized using AdamW (learning rate 3e-4) and CrossEntropyLoss over 6 epochs.   Evaluation Best Practices: Validation and testing loops explicitly utilized model.eval() and torch.no_grad() to disable dropout layers and prevent gradient tracking, ensuring accurate and memory-efficient results.   Final Performance: The trained model successfully achieved a final test accuracy of 76.10%, surpassing the initial project baseline requirement of 75.0%.   🚀 Usage & InferenceThe project provides a packaged SentimentPredictor interface class that easily loads the saved model checkpoint (sentimentscope_model.pt) to perform batch inference on new, raw text strings.   Pythonimport torch

# Initialize the predictor using the saved checkpoint
predictor = SentimentPredictor(
    model_path="sentimentscope_model.pt", 
    config=config, 
    tokenizer=tokenizer, 
    device=device
)

# Test it on a custom batch of raw text inputs
custom_reviews = [
    "This movie was an absolute masterpiece! The acting was phenomenal.",
    "Terrible script and awful acting. I wasted two hours of my life."
]

predictions = predictor.predict(custom_reviews)

for result in predictions:
    print(f"Sentiment: {result['sentiment']} | Review: {result['review']}")
