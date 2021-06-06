import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer

if __name__ == '__main__':

    text = str(sys.argv[0])

    model_name = 'garynguyen1174/disaster_tweet_bert'
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    cls_explainer = SequenceClassificationExplainer(
        model,
        tokenizer
        )
    word_attributions = cls_explainer(text)
    cls_explainer.visualize("viz.html")
    print(word_attributions)


