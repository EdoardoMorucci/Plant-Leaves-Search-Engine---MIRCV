
# Plant Leaves Search Engine

Developed a search engine capable of retrieving similar images in a plant leaves dataset. Employed a fine-tuned DenseNet for feature
extraction.

Project completed for the “Computer Vision and Information Retrieval” exam.


## Short Description

The goal of this project is to develop a **leaf recognition system**, capable of identifying the plant species from an uploaded leaf image.

The system leverages a ***fine-tuned version of DenseNet121*** as the foundational model for feature extraction. These features are then used to construct a ***Vantage Point Tree (VPT) index*** for retrieval. The system can accurately identify leaves from 14 different plant species. In cases where non-leaf images are uploaded, the system primarily returns images from a distractor dataset.

Finally, a ***web user interface*** was built to allow the search engine to be used from a web browser.

The VPT index employs a precise **similarity search** method that segments the dataset into subsets using the ball partitioning technique. The VPT itself is a balanced binary tree with a fixed structure. Once created, it facilitates k-NN queries. The system's performance has yielded significant improvements, reducing the mean query time by 35% compared to the brute force approach, and also decreasing the computed distances by 49.4%.

The system operates with two distinct datasets. The first dataset comprises 72,000 leaf images categorized into 14 classes, while the second dataset contains 25,000 images meant to account for noise. This second dataset is especially useful when users upload non-leaf images.
