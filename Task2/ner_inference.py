from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer


def ner_inference(text):
    '''
    Named Entity Recognition (NER) inference on the text given by user.

    Parameters:
    text (str): The input text.

    Returns:
    str or None: The recognized entity if found (if only one), otherwise None.
    '''
    try:
        # Loading model and tokenizer
        model_path = "ner_model"
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Initialization NER pipeline
        pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        result = pipe(text)

        # Validate the output: Ensure only one entity is returned
        if len(result) > 1:
            raise ValueError("Please provide only one answer.")
        if len(result) == 0:
            raise ValueError("No entity found.")
        return result[0]['word']

    except ValueError as e:
        print(f"Error: {e}")
        return None

