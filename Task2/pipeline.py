import argparse
from cv_inference import cv_inference
from ner_inference import ner_inference


def pipeline(text: str, image_path: str) -> bool:
    '''
    Performs Named Entity Recognition (NER) and Computer Vision (CV)
    inference and checks if the results are matching.

    Args:
    text (str): Input text for the NER model.
    image_path (str): Path to the image for the CV model.

    Returns:
    bool: True if the NER and CV results match, otherwise False.
    '''
    try:
        # Inferencing
        ner_result = ner_inference(text)
        cv_result = cv_inference(image_path)

        # Check if both results are not None and if they match
        if ner_result is not None and cv_result is not None and ner_result == cv_result:
            return True
    except Exception as e:
        print(f"Error: {e}")
    return False


if __name__ == "__main__":
    # Parsing the arguments from the command line
    parser = argparse.ArgumentParser(description="Pipeline for NER and CV model inference.")
    parser.add_argument("text", type=str, help="Input text for NER model.")
    parser.add_argument("image_path", type=str, help="Path to image for CV model.")
    args = parser.parse_args()
    print("Processing...")

    # Call the pipeline function with the parsed arguments and print the result.
    result = pipeline(args.text, args.image_path)
    print(result)
