# bac
Bachelor Thesis

Step 1: Command to train a model:

python retrain.py \
--bottleneck_dir=bottlenecks \
--how_many_training_steps=1000 \
--model_dir=mobilenet \
--summaries_dir=training_summaries/basic \
--output_graph=retrained_graph.pb \
--output_labels=retrained_labels.txt \
--image_dir=categories

Images have to be in a folder structure in "categories". For Example the folder categories with a folder named carrot and another named apple. In the carrot and apple folders should be images from apples and carrots. The result will be a model.

Step 2: To classify a image:

Start the "Klassifizieren"-Main method with the path to the image and the path to the model from Step 1.

ORCID: https://orcid.org/0000-0002-6975-1959
