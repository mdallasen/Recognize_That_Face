import argparse
from train import Training
from inference import predict_attributes
from evaluate import evaluate_model
import os

def main():
    """Main function to handle training, inference, and evaluation."""
    
    parser = argparse.ArgumentParser(description="Facial Attribute Classification Pipeline")

    parser.add_argument("--task", type=str, required=True, choices=["train", "infer", "evaluate"],
                        help="Choose a task: 'train' to train the model, 'infer' for predictions, 'evaluate' for model evaluation.")

    parser.add_argument("--image", type=str, default=None, 
                        help="Path to an image for inference (required for --task infer).")

    args = parser.parse_args()

    if args.task == "train":
        print("🚀 Starting Training...")
        trainer = Training()
        trainer.train(batch_size=32, epochs=10)
    
    elif args.task == "infer":
        if not args.image or not os.path.exists(args.image):
            print("❌ Error: Please provide a valid image path for inference.")
            return
        
        print(f"🔍 Running inference on {args.image}...")
        from tensorflow.keras.models import load_model
        import pandas as pd
        
        # Load model
        model = load_model("models/attribute_model.h5")
        
        # Load attribute names
        attributes_list = list(pd.read_csv("list_attr_celeba.csv").columns[1:])
        
        # Run inference
        top_attrs = predict_attributes(args.image, model, attributes_list)
        print(f"🔹 **Top 5 Predicted Attributes for {args.image}:**")
        for attr, prob in top_attrs:
            print(f"- {attr}: {prob:.4f}")

    elif args.task == "evaluate":
        print("📊 Evaluating model performance...")
        evaluate_model()

if __name__ == "__main__":
    main()