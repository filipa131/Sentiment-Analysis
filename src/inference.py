from utils import predict_message, load_model_and_tokenizer

if __name__ == "__main__":
    # Load model and tokenizer
    tokenizer, model, device = load_model_and_tokenizer()

    while True:
        message = input("Enter a message (or type 'exit' to quit): ")
        if message.lower() == "exit":
            break
        
        # Make predictions 
        prediction = predict_message(message, tokenizer, model, device)
        print(f"Prediction: {prediction}")
