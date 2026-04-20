# Objective and Evaluation Metric

## Objective

The objective of this project is to develop an image captioning model that generates natural language descriptions from visual inputs. This is done by using a **convolutional neural network (CNN)** to extract meaningful features from images and a **long short-term memory (LSTM)** network to generate descriptive captions.

By connecting these two components, the model aims to recognize the context of an image and produce **accurate and meaningful descriptions**. This objective is clearly defined and directly aligns with the model architecture being used.

### Evaluation Metrics

This project is evaluated using two common metrics: **BLEU** and **METEOR**. Both metrics compare the generated captions to the ground truth captions, but they do so in different ways.

- **BLEU Score**  
  BLEU measures the similarity between the generated caption and the reference caption using **n-gram precision**. This means it checks how many sequences of words in the generated caption match those in the ground truth. BLEU is useful for quantitatively measuring how closely the generated caption matches the reference.

- **METEOR Score**  
  METEOR provides a more balanced evaluation by considering both **precision and recall**. It aligns words between the generated and reference captions and also uses **stemming**, which allows it to match words with similar meanings. This makes METEOR better at capturing semantic similarity.

### Justification

Both metrics are used because they complement each other:

- BLEU is strict and focuses on exact word matches  
- METEOR is more flexible and accounts for meaning and variation in wording  

This is important because a generated caption can be correct in meaning but use different phrasing than the ground truth, which BLEU alone may not reward.

### Success Criteria

The project is considered successful if the model performs similarly to other CNN-LSTM models on the Flickr8k dataset, with:

- **BLEU-4:** ~0.10 – 0.20  
- **METEOR:** ~0.20 – 0.30  

These ranges are based on previously reported results for similar models, making them a reasonable benchmark for evaluation.

### Conclusion

Overall, the objective and evaluation metrics are clearly stated and well motivated. The objective matches the model design, and the use of both BLEU and METEOR provides a more complete and fair evaluation of the generated captions.

## Dataset: `https://www.kaggle.com/datasets/adityajn105/flickr8k`

This dataset is needed to run the program. Please download it and put the contents in a directory named 'archive'. The captions should directly be in the archives directory and then make another directory inside named 'Images', which contains all of the images.
